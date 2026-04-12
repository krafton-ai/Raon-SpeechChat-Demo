"""Worker entry point for fd-demo split deployment.

Starts a Ray head node (or joins an existing cluster), creates the RouterActor
and RaonWorkerActors, then blocks until shutdown signal.

Usage:
    # Head node (first worker, starts Ray head + RouterActor):
    python3 launch_worker.py --role head --gpu-ids 0,1

    # Additional worker node (joins existing head):
    python3 launch_worker.py --role worker-node --ray-address 10.0.0.1:6379 --gpu-ids 0,1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from proto.close_reasons import INTERNAL_ERROR

MODEL_PATH_DEFAULT = "/models/sglang-bundle"

logger = logging.getLogger("fd-demo.worker")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fd-demo worker: GPU inference backend")
    p.add_argument(
        "--role", choices=["head", "worker-node"], default="head",
        help="head=start Ray head node + RouterActor, worker-node=join existing cluster",
    )
    p.add_argument("--ray-address", default=None,
                   help="Ray head address for worker-node role (e.g. 10.0.0.1:6379)")
    p.add_argument("--ray-port", type=int, default=6379,
                   help="Ray GCS port for head node")
    p.add_argument("--ray-client-port", type=int, default=10001,
                   help="Ray client server port for head node")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", MODEL_PATH_DEFAULT),
                   help="Path to Raon-Speech model checkpoint (required)")
    p.add_argument("--gpu-ids", default=os.environ.get("FD_GPU_IDS", "0"),
                   help="Comma-separated GPU IDs")
    p.add_argument(
        "--max-sessions-per-gpu",
        type=int,
        default=int(os.environ.get("FD_MAX_SESSIONS_PER_GPU", "2")),
    )
    p.add_argument("--dtype", default="auto")
    p.add_argument("--mem-fraction", type=float, default=0.75)
    p.add_argument("--router-health-interval", type=float, default=5.0)
    p.add_argument("--router-reconcile-interval", type=float, default=7.5)
    p.add_argument("--router-health-timeout", type=float, default=20.0)
    p.add_argument("--router-reserved-ttl", type=float,
                   default=float(os.environ.get("FD_ROUTER_RESERVED_TTL", "15.0")))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    return args


def _parse_gpu_ids(raw: str) -> list[int]:
    parsed = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token))
    if not parsed:
        raise ValueError("No GPU IDs provided")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"Duplicate GPU IDs are not allowed: {raw!r}")
    return parsed


def _parse_visible_gpu_ids() -> set[int] | None:
    for env_name in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(env_name, "").strip()
        if not raw or raw.lower() in {"all", "void", "none"}:
            continue
        items = [part.strip() for part in raw.split(",") if part.strip()]
        if not items:
            continue
        try:
            return {int(item) for item in items}
        except ValueError:
            logger.info(
                "Skipping visible GPU validation for %s=%r because entries are not numeric IDs",
                env_name,
                raw,
            )
            return None
    return None


def _validate_gpu_ids(gpu_ids: list[int]) -> None:
    visible_gpu_ids = _parse_visible_gpu_ids()
    if visible_gpu_ids is None:
        return
    missing_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in visible_gpu_ids]
    if missing_gpu_ids:
        raise RuntimeError(
            f"Requested GPU IDs {missing_gpu_ids} are not visible in the container "
            f"(visible: {sorted(visible_gpu_ids)})"
        )


async def _run_worker(args: argparse.Namespace) -> None:
    import ray

    from proto.config import SGLangConfig, WorkerConfig
    from router.actor import make_router_actor_cls
    from worker.actor import get_raon_actor_cls

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    _validate_gpu_ids(gpu_ids)

    # Initialize Ray
    if args.role == "head":
        # Start Ray head with client server enabled via subprocess
        import subprocess
        subprocess.run([
            "ray", "start", "--head",
            "--num-gpus", str(len(gpu_ids)),
            "--dashboard-host", "0.0.0.0",
            "--ray-client-server-port", str(args.ray_client_port),
            "--port", str(args.ray_port),
        ], check=True)
        ray.init(
            address="auto",
            namespace="default",
        )
        logger.info("Ray head node started (GPUs: %s)", gpu_ids)

        # Create RouterActor on the head node
        RouterActor = make_router_actor_cls()
        router_actor = RouterActor.options(name="fd_router").remote(
            health_interval=args.router_health_interval,
            reconcile_interval=args.router_reconcile_interval,
            health_timeout=args.router_health_timeout,
            reserved_ttl_sec=args.router_reserved_ttl,
        )
        logger.info("RouterActor created")
    else:
        # Join existing Ray cluster
        if not args.ray_address:
            raise ValueError("--ray-address is required for worker-node role")
        ray.init(address=args.ray_address, namespace="default")
        logger.info("Joined Ray cluster at %s", args.ray_address)
        router_actor = ray.get_actor("fd_router")
        logger.info("Found existing RouterActor")

    # Create worker actors — first one downloads shared assets (speaker encoder),
    # then remaining GPUs load in parallel using the cached files.
    RaonWorkerActor = get_raon_actor_cls()

    def _make_config(gpu_id: int) -> WorkerConfig:
        return WorkerConfig(
            gpu_id=gpu_id,
            sglang=SGLangConfig(
                model_path=args.model_path,
                dtype=args.dtype,
                mem_fraction_static=args.mem_fraction,
            ),
            max_sessions=args.max_sessions_per_gpu,
        )

    def _dispose_failed_actor(actor: object, gpu_id: int) -> None:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            logger.debug("Failed to kill actor for GPU %d after init failure", gpu_id, exc_info=True)

    # First worker: initialize one GPU at a time until one succeeds so shared
    # caches are populated. This avoids hard-failing the entire service just
    # because gpu_ids[0] happens to be busy or fragmented.
    first_success: tuple[int, object] | None = None
    failed_gpus: dict[int, str] = {}
    remaining_gpu_ids: list[int] = []
    for idx, gpu_id in enumerate(gpu_ids):
        actor = RaonWorkerActor.remote()
        logger.info("Initializing candidate worker on GPU %d...", gpu_id)
        try:
            ray.get(actor.initialize.remote(_make_config(gpu_id)))
        except Exception as exc:
            failed_gpus[gpu_id] = str(exc)
            logger.exception("Worker initialization failed on GPU %d", gpu_id)
            _dispose_failed_actor(actor, gpu_id)
            continue

        first_success = (gpu_id, actor)
        remaining_gpu_ids = gpu_ids[:idx] + gpu_ids[idx + 1 :]
        logger.info("First worker on GPU %d ready, caches populated", gpu_id)
        break

    if first_success is None:
        raise RuntimeError(f"Failed to initialize any configured GPU: {failed_gpus}")

    # Register the first ready worker immediately so the service can start with
    # partial capacity while additional GPUs continue loading in the background.
    worker_handles: list[tuple[int, object]] = []
    loaded_gpu_ids: list[int] = []

    first_gpu_id, first_actor = first_success
    first_worker_id = f"worker-gpu{first_gpu_id}"
    await router_actor.register_worker.remote(
        first_worker_id, first_actor, first_gpu_id, args.max_sessions_per_gpu,
    )
    worker_handles.append((first_gpu_id, first_actor))
    loaded_gpu_ids.append(first_gpu_id)
    logger.info("Registered %s; continuing to load remaining GPUs...", first_worker_id)

    # Remaining workers: initialize in parallel. Failures are logged and skipped
    # so one bad GPU does not take down otherwise-healthy workers.
    actors_and_refs: list[tuple[int, object, object | None]] = [(first_gpu_id, first_actor, None)]
    pending_refs: dict[object, tuple[int, object]] = {}
    for gpu_id in remaining_gpu_ids:
        actor = RaonWorkerActor.remote()
        init_ref = actor.initialize.remote(_make_config(gpu_id))
        pending_refs[init_ref] = (gpu_id, actor)

    if pending_refs:
        logger.info("Loading model on remaining %d GPUs in parallel...", len(pending_refs))
        while pending_refs:
            ready_refs, _ = ray.wait(list(pending_refs.keys()), num_returns=1)
            ready_ref = ready_refs[0]
            gpu_id, actor = pending_refs.pop(ready_ref)
            try:
                ray.get(ready_ref)
            except Exception as exc:
                failed_gpus[gpu_id] = str(exc)
                logger.exception("Worker initialization failed on GPU %d", gpu_id)
                _dispose_failed_actor(actor, gpu_id)
                continue
            actors_and_refs.append((gpu_id, actor, None))
            worker_id = f"worker-gpu{gpu_id}"
            await router_actor.register_worker.remote(
                worker_id, actor, gpu_id, args.max_sessions_per_gpu,
            )
            worker_handles.append((gpu_id, actor))
            loaded_gpu_ids.append(gpu_id)
            logger.info("Worker on GPU %d ready and registered", gpu_id)

    logger.info("Loaded %d/%d GPUs: %s", len(loaded_gpu_ids), len(gpu_ids), loaded_gpu_ids)
    if failed_gpus:
        logger.warning("Skipped failed GPUs: %s", sorted(failed_gpus))

    logger.info("Registered %d workers. Waiting for gateway connections...", len(worker_handles))

    # Block until shutdown signal
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    await shutdown_event.wait()
    logger.info("Shutdown signal received, cleaning up...")

    # Graceful cleanup: close all sessions
    for gpu_id, actor in worker_handles:
        try:
            for sid in await actor.list_sessions.remote():
                await actor.close_session.remote(sid, INTERNAL_ERROR)
        except Exception:
            pass

    ray.shutdown()
    logger.info("Worker shutdown complete")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(_run_worker(args))


if __name__ == "__main__":
    main()

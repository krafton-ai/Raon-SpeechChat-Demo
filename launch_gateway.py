"""Gateway entry point for fd-demo split deployment.

Connects to an existing Ray cluster, discovers the RouterActor, and runs
the FastAPI WebSocket gateway with frontend static file serving.

Usage:
    python3 launch_gateway.py --ray-address ray://fd-worker:10001 --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time


logger = logging.getLogger("fd-demo.gateway")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fd-demo gateway: WebSocket frontend")
    p.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "ray://localhost:10001"),
        help="Ray client address (e.g. ray://fd-worker:10001)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--ws-backend",
        choices=["auto", "websockets", "wsproto"],
        default="wsproto",
        help="Uvicorn WebSocket backend. wsproto is generally proxy/tunnel friendly.",
    )
    p.add_argument(
        "--root-path", default=os.environ.get("FD_ROOT_PATH", ""),
        help="ASGI root_path for reverse proxy prefix (e.g. /eval/fd-demo)",
    )
    p.add_argument("--log-level", default="INFO")

    # Protocol selection: default HTTPS, --http degrades to HTTP
    proto_group = p.add_mutually_exclusive_group()
    proto_group.add_argument("--https", action="store_true", default=True,
                             help="Enable HTTPS (default, self-signed cert)")
    proto_group.add_argument("--http", action="store_true",
                             help="Degrade to HTTP (not recommended, browser APIs like microphone need HTTPS)")
    p.add_argument("--ssl-certfile", type=str, default="certs/cert.pem", help="SSL cert file path")
    p.add_argument("--ssl-keyfile", type=str, default="certs/key.pem", help="SSL key file path")

    return p.parse_args()


def _wait_for_router(max_attempts: int = 60, interval: float = 5.0):
    """Wait for the fd_router actor to be available in the Ray cluster."""
    import ray

    for attempt in range(max_attempts):
        try:
            router = ray.get_actor("fd_router", namespace="default")
            status = ray.get(router.status.remote())
            healthy_count = status.get("healthy_worker_count", 0)
            logger.info(
                "Router found. %d healthy worker(s) available.", healthy_count,
            )
            return router
        except Exception:
            if attempt % 6 == 0:
                logger.info(
                    "Waiting for router actor... (attempt %d/%d)",
                    attempt + 1,
                    max_attempts,
                )
            time.sleep(interval)

    raise RuntimeError(
        f"Could not find fd_router actor after {max_attempts * interval:.0f}s"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import ray
    import uvicorn

    from proto.config import GatewayConfig
    from gateway.server import create_app

    # Connect to existing Ray cluster (worker head node)
    logger.info("Connecting to Ray cluster at %s ...", args.ray_address)
    ray.init(address=args.ray_address, namespace="default")
    logger.info("Connected to Ray cluster")

    # Wait for router actor to be available
    _wait_for_router()

    # Create and run gateway
    gateway_config = GatewayConfig(host=args.host, port=args.port)
    app = create_app(gateway_config)

    use_https = not args.http
    ssl_kwargs = {}
    if use_https:
        cert = args.ssl_certfile
        key = args.ssl_keyfile
        if not os.path.exists(cert) or not os.path.exists(key):
            logger.error("SSL cert/key not found: %s, %s", cert, key)
            logger.error("Generate with: openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj '/CN=dev'")
            logger.error("Or use --http to start without HTTPS")
            return
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        logger.info("HTTPS enabled: cert=%s, key=%s", cert, key)
    else:
        logger.warning("Running in HTTP mode (no TLS). Browser microphone/camera APIs may not work.")

    uvi_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        ws=args.ws_backend,
        proxy_headers=True,
        forwarded_allow_ips="*",
        root_path=args.root_path,
        **ssl_kwargs,
    )
    server = uvicorn.Server(uvi_config)

    proto = "https" if use_https else "http"
    logger.info(
        "Starting fd-demo gateway on %s://%s:%d (ray=%s)",
        proto,
        args.host,
        args.port,
        args.ray_address,
    )
    server.run()


if __name__ == "__main__":
    main()

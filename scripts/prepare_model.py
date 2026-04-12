#!/usr/bin/env python3
"""Download and export a Raon-Speech model for the SpeechChat Demo.

Runs before the worker starts. Checks if an SGLang bundle already exists;
if not, downloads the HuggingFace checkpoint and exports it.

Environment variables:
    MODEL_PATH   — If set and points to a valid SGLang bundle, skip everything.
    HF_MODEL_ID  — HuggingFace model ID (default: KRAFTON/Raon-SpeechChat-9B).
    HF_TOKEN     — Optional HuggingFace token for gated models.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [prepare_model] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_model")

SGLANG_BUNDLE_DIR = "/models/sglang-bundle"
HF_CACHE_DIR = "/models/hf-cache"
MODEL_PATH_FILE = "/models/.model_path"

DEFAULT_HF_MODEL_ID = "KRAFTON/Raon-SpeechChat-9B"


def _is_valid_sglang_bundle(path: str | Path) -> bool:
    """Check if a directory looks like a complete SGLang bundle."""
    p = Path(path)
    return (
        (p / "text_model" / "config.json").is_file()
        and (p / "raon_runtime" / "config.json").is_file()
    )


def _write_model_path(path: str) -> None:
    """Write resolved MODEL_PATH to a file for entrypoint.sh to read."""
    Path(MODEL_PATH_FILE).write_text(path.strip())
    logger.info("MODEL_PATH written to %s → %s", MODEL_PATH_FILE, path)


def main() -> None:
    print("=" * 60)
    print("  Raon Model Preparation")
    print("=" * 60)

    model_path = os.environ.get("MODEL_PATH", "").strip()

    # Case 1: User provided a MODEL_PATH that already has a valid bundle
    if model_path and _is_valid_sglang_bundle(model_path):
        logger.info("Using user-provided MODEL_PATH=%s", model_path)
        _write_model_path(model_path)
        return

    # Case 2: User provided MODEL_PATH but it's not an SGLang bundle
    if model_path:
        logger.warning(
            "MODEL_PATH=%s is set but does not contain a valid SGLang bundle "
            "(expected text_model/ + raon_runtime/ subdirectories). "
            "Will attempt auto-download instead.",
            model_path,
        )

    # Case 3: Check if a previous run already produced a bundle
    if _is_valid_sglang_bundle(SGLANG_BUNDLE_DIR):
        logger.info(
            "SGLang bundle found at %s, skipping download.", SGLANG_BUNDLE_DIR
        )
        _write_model_path(SGLANG_BUNDLE_DIR)
        return

    # Case 4: Download and export
    hf_model_id = os.environ.get("HF_MODEL_ID", DEFAULT_HF_MODEL_ID).strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    logger.info("Downloading %s from HuggingFace...", hf_model_id)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Cannot download model. "
            "Either install it or provide MODEL_PATH manually."
        )
        sys.exit(1)

    hf_local_dir = os.path.join(HF_CACHE_DIR, hf_model_id.replace("/", "--"))
    os.makedirs(hf_local_dir, exist_ok=True)

    snapshot_download(
        hf_model_id,
        local_dir=hf_local_dir,
        token=hf_token,
    )
    logger.info("Download complete: %s", hf_local_dir)

    # Export HF → SGLang
    logger.info("Exporting HF checkpoint to SGLang bundle...")

    try:
        from raon.export import export_hf_to_sglang
    except ImportError:
        logger.error(
            "raon package not installed. Cannot export model. "
            "Install with: pip install 'raon @ git+https://github.com/krafton-ai/Raon-Speech.git'"
        )
        sys.exit(1)

    import torch

    export_hf_to_sglang(
        input_path=hf_local_dir,
        output_path=SGLANG_BUNDLE_DIR,
        dtype=torch.bfloat16,
    )

    if not _is_valid_sglang_bundle(SGLANG_BUNDLE_DIR):
        logger.error("Export completed but bundle validation failed at %s", SGLANG_BUNDLE_DIR)
        sys.exit(1)

    logger.info("Export complete! SGLang bundle at %s", SGLANG_BUNDLE_DIR)
    _write_model_path(SGLANG_BUNDLE_DIR)


if __name__ == "__main__":
    main()

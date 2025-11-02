#!/usr/bin/env python3
import argparse
import pathlib
from typing import Dict, Any

from syntheticdocs import (
    load_config,
    resolve_path,
    make_text_generator,
    generate_batch,
)


def main():
    p = argparse.ArgumentParser(description="Generate train/test synthetic docs split using syntheticdocs")
    p.add_argument("--config", required=True, help="Path to config.json (will also read augment/layout)")
    p.add_argument("--train-out", required=True, help="Output directory for train set")
    p.add_argument("--test-out", required=True, help="Output directory for test set")
    p.add_argument("--train-n", type=int, default=1000)
    p.add_argument("--test-n", type=int, default=100)
    args = p.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    llm_cfg = cfg.get("llm", {})
    gen_cfg = cfg.get("generator", {})
    augment_cfg = cfg.get("augment")
    layout_cfg = cfg.get("layout")

    # Build text generator
    gen_text_fn = make_text_generator(llm_cfg.get("mode", "vllm"), llm_cfg)
    batch_size = int(llm_cfg.get("batch_size", 8))

    # Common assets
    sig_dir = resolve_path(gen_cfg["signatures_dir"])
    stamp_dir = resolve_path(gen_cfg["stamps_dir"])
    render_workers = int(gen_cfg.get("render_workers", 0) or 0)

    # Train
    generate_batch(
        gen_text_fn,
        int(args.train_n),
        sig_dir,
        stamp_dir,
        resolve_path(args.train_out),
        augment_cfg=augment_cfg,
        layout_cfg=layout_cfg,
        batch_size=batch_size,
        render_workers=render_workers,
    )

    # Test
    generate_batch(
        gen_text_fn,
        int(args.test_n),
        sig_dir,
        stamp_dir,
        resolve_path(args.test_out),
        augment_cfg=augment_cfg,
        layout_cfg=layout_cfg,
        batch_size=batch_size,
        render_workers=render_workers,
    )


if __name__ == "__main__":
    main()



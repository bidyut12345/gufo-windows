#!/usr/bin/env python3
"""Inventory or verify the pinned MiniMax H3 FL2VA source checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.h3_manifest import (
    H3ManifestError,
    build_manifest,
    verify_manifest,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MiniMax H3 FL2VA without importing model-provided Python"
        )
    )
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument(
        "--variant",
        choices=("fl2va", "ref2va"),
        default="fl2va",
        help="MiniMax H3 variant to inventory (default: fl2va)",
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.verify:
            manifest = verify_manifest(
                args.model_root, args.verify, progress=not args.quiet
            )
            action = "verified"
        else:
            manifest = build_manifest(
                args.model_root, variant=args.variant, progress=not args.quiet
            )
            write_manifest(args.output, manifest)
            action = "written"
    except H3ManifestError as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"MiniMax H3 manifest error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": "ok",
        "action": action,
        "model_kind": manifest["model_kind"],
        "revision": manifest["revision"],
        "file_count": manifest["file_count"],
        "tensor_count": manifest["tensor_count"],
        "total_file_bytes": manifest["total_file_bytes"],
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            f"{action}: {summary['model_kind']} revision {summary['revision']} "
            f"({summary['file_count']} files, {summary['tensor_count']} tensors, "
            f"{summary['total_file_bytes']} bytes)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

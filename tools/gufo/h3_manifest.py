"""Deterministic MiniMax H3 FL2VA source inventory and validation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from . import control_tokens, safetensors

SCHEMA = "gufo.minimax-h3-source.v1"
REPOSITORY = "MiniMaxAI/MiniMax-H3"
REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
MODEL_KIND = "minimax-h3-fl2va-bf16"
REF2VA_MODEL_KIND = "minimax-h3-ref2va-bf16"
REFERENCE_REPOSITORY = "antirez/h3.c"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
LICENSE_SHA256 = (
    "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"
)

def _variant_dir(variant: str) -> str:
    if variant == "fl2va":
        return "FL2VA"
    if variant == "ref2va":
        return "Ref2VA"
    raise H3ManifestError(f"unsupported MiniMax H3 variant: {variant!r}")


def _required_files(variant: str) -> set[str]:
    v = _variant_dir(variant)
    return {
        "LICENSE",
        "README.md",
        "model_index.json",
        f"{v}/model_index.json",
        f"{v}/transformer/config.json",
        f"{v}/transformer/model.safetensors.index.json",
        f"{v}/text_encoder/config.json",
        f"{v}/text_encoder/model.safetensors.index.json",
        f"{v}/tokenizer/merges.txt",
        f"{v}/tokenizer/tokenizer_config.json",
        f"{v}/tokenizer/tokenizer.json",
        f"{v}/tokenizer/vocab.json",
        f"{v}/processor/chat_template.json",
        f"{v}/processor/preprocessor_config.json",
        f"{v}/processor/tokenizer_config.json",
        f"{v}/processor/tokenizer.json",
        f"{v}/processor/video_preprocessor_config.json",
        f"{v}/processor/vocab.json",
        f"{v}/video_vae/config.json",
        f"{v}/video_vae/source/config.json",
        f"{v}/video_vae/source/model.safetensors",
        f"{v}/audio_vae/config.json",
        f"{v}/audio_vae/metadata.json",
        f"{v}/audio_vae/model.safetensors",
    }


def _components(variant: str) -> tuple[dict, ...]:
    v = _variant_dir(variant)
    return (
        {
            "name": "text_encoder",
            "directory": f"{v}/text_encoder",
            "index": "model.safetensors.index.json",
            "allowed_dtypes": {"BF16"},
        },
        {
            "name": "transformer",
            "directory": f"{v}/transformer",
            "index": "model.safetensors.index.json",
            "allowed_dtypes": {"BF16", "F32"},
        },
        {
            "name": "video_vae",
            "directory": f"{v}/video_vae/source",
            "index": None,
            "allowed_dtypes": {"F32"},
        },
        {
            "name": "audio_vae",
            "directory": f"{v}/audio_vae",
            "index": None,
            "allowed_dtypes": {"F32"},
        },
    )


REQUIRED_FILES = _required_files("fl2va")
COMPONENTS = _components("fl2va")


class H3ManifestError(Exception):
    """Raised when an H3 source directory violates the pinned contract."""


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: Path):
    try:
        return safetensors.load_json_file(path)
    except safetensors.SafetensorsError as exc:
        raise H3ManifestError(str(exc)) from exc


def _require_equal(actual, expected, description: str) -> None:
    if actual != expected:
        raise H3ManifestError(
            f"{description}: expected {expected!r}, got {actual!r}"
        )


def _artifact_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise H3ManifestError(f"symbolic links are not allowed: {relative}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _validate_selected_tree(
    root: Path, files: list[Path], variant: str = "fl2va"
) -> None:
    v = _variant_dir(variant)
    relative_files = {path.relative_to(root).as_posix() for path in files}
    missing = _required_files(variant) - relative_files
    if missing:
        raise H3ManifestError(f"required files missing: {sorted(missing)}")

    disallowed_variant = "Ref2VA" if variant == "fl2va" else "FL2VA"
    unsupported_roots = (disallowed_variant, "H3-Regenerate-2K", "Regenerate-2K")
    for name in unsupported_roots:
        if (root / name).exists():
            raise H3ManifestError(f"unsupported H3 artifact is present: {name}")

    for relative in sorted(relative_files):
        lowered = relative.lower()
        if (
            "regenerate-2k" in lowered
            or f"/{disallowed_variant.lower()}/" in f"/{lowered}/"
        ):
            raise H3ManifestError(f"unsupported H3 artifact file: {relative}")
        if Path(relative).suffix.lower() in {".bin", ".pt", ".pth", ".ckpt"}:
            raise H3ManifestError(
                f"pickle/checkpoint payload is unsupported: {relative}"
            )


def _metadata_for(root: Path, relative: str) -> tuple[str, str]:
    metadata = (
        root / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    )
    if not metadata.is_file():
        raise H3ManifestError(f"missing Hugging Face revision metadata: {relative}")
    lines = metadata.read_text("utf-8").splitlines()
    if len(lines) < 2:
        raise H3ManifestError(f"truncated Hugging Face metadata: {relative}")
    revision, upstream_blob = lines[0].strip(), lines[1].strip()
    _require_equal(revision, REVISION, f"{relative} revision")
    if not upstream_blob:
        raise H3ManifestError(f"missing upstream blob identifier: {relative}")
    return revision, upstream_blob


def _validate_indexes_and_configs(
    root: Path, variant: str = "fl2va"
) -> dict:
    v = _variant_dir(variant)
    repository_index = _load_json(root / "model_index.json")
    variant_index = _load_json(root / v / "model_index.json")
    transformer = _load_json(root / v / "transformer" / "config.json")
    text_encoder = _load_json(root / v / "text_encoder" / "config.json")
    tokenizer = _load_json(
        root / v / "tokenizer" / "tokenizer_config.json"
    )
    audio_metadata = _load_json(
        root / v / "audio_vae" / "metadata.json"
    )

    _require_equal(
        repository_index.get("_class_name"),
        "MiniMaxH3ModularPipeline",
        "repository model class",
    )
    _require_equal(
        variant_index.get("_class_name"),
        "MiniMaxH3Pipeline",
        f"{v} model class",
    )
    minimax = variant_index.get("_minimax_h3")
    if not isinstance(minimax, dict):
        raise H3ManifestError(f"{v} model index lacks _minimax_h3 metadata")
    _require_equal(minimax.get("schema_version"), 1, f"{v} schema version")
    _require_equal(minimax.get("partition"), variant, f"{v} partition")
    _require_equal(
        minimax.get("sigma_shift_scales"),
        {"video": 12.0, "audio": 3.0},
        f"{v} sigma shift scales",
    )
    tasks = minimax.get("tasks")
    expected_task = "t2va" if variant == "fl2va" else "ref2va"
    if not isinstance(tasks, list) or expected_task not in tasks:
        raise H3ManifestError(f"{v} model index does not declare {expected_task}")

    expected_transformer = {
        "_class_name": "MiniMaxH3DiTModel",
        "hidden_size": 5376,
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "ffn_hidden_size": 14336,
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 5120,
    }
    for key, expected in expected_transformer.items():
        _require_equal(transformer.get(key), expected, f"transformer {key}")

    text_config = text_encoder.get("text_config")
    if not isinstance(text_config, dict):
        raise H3ManifestError("text encoder config lacks text_config")
    expected_text = {
        "model_type": "qwen3_vl_text",
        "dtype": "bfloat16",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
    }
    for key, expected in expected_text.items():
        _require_equal(text_config.get(key), expected, f"text encoder {key}")

    _require_equal(tokenizer.get("eos_token"), control_tokens.kImEnd, "tokenizer EOS")
    _require_equal(
        tokenizer.get("pad_token"), control_tokens.kEndOfText, "tokenizer padding token"
    )
    special_tokens = tokenizer.get("additional_special_tokens")
    if not isinstance(special_tokens, list):
        raise H3ManifestError("tokenizer special-token list is missing")
    required_special = set(control_tokens.H3_SPECIAL_TOKENS)
    missing_special = required_special - set(special_tokens)
    if missing_special:
        raise H3ManifestError(
            f"tokenizer special tokens missing: {sorted(missing_special)}"
        )

    try:
        audio_kwargs = audio_metadata["metadata"]["kwargs"]
    except (KeyError, TypeError) as exc:
        raise H3ManifestError("audio VAE metadata lacks metadata.kwargs") from exc
    _require_equal(audio_kwargs.get("sample_rate"), 32000, "audio sample rate")
    _require_equal(
        audio_kwargs.get("vae_latent_channels"), 32, "audio latent channels"
    )

    return {
        "repository_model_class": repository_index["_class_name"],
        f"{variant}_model_class": variant_index["_class_name"],
        "upstream_declared_tasks": tasks,
        "runtime_capabilities": (
            ["text-to-audio-video"]
            if variant == "fl2va"
            else ["reference-to-audio-video"]
        ),
        "text_encoder": {
            "architecture": "Qwen3-VL-32B",
            "source_layers": text_config["num_hidden_layers"],
            "output_hidden_layer": 50,
            "hidden_size": text_config["hidden_size"],
            "vocab_size": text_config["vocab_size"],
        },
        "transformer": {
            key: transformer[key] for key in expected_transformer
        },
        "tokenizer": {
            "implementation": "Qwen2TokenizerFast",
            "eos_token": tokenizer["eos_token"],
            "pad_token": tokenizer["pad_token"],
            "additional_special_tokens": special_tokens,
        },
        "audio": {
            "sample_rate": audio_kwargs["sample_rate"],
            "latent_channels": audio_kwargs["vae_latent_channels"],
        },
    }


def _component_inventory(
    root: Path, variant: str = "fl2va"
) -> tuple[list[dict], list[dict]]:
    components = []
    tensors = []
    for component in _components(variant):
        directory = root / component["directory"]
        index_path = (
            directory / component["index"] if component["index"] else None
        )
        try:
            tensor_map, shard_files = safetensors.inspect_snapshot(
                directory, index_path
            )
        except safetensors.SafetensorsError as exc:
            raise H3ManifestError(
                f"{component['name']} safetensors validation failed: {exc}"
            ) from exc

        dtype_counts = {}
        encoded_bytes = 0
        for tensor_name in sorted(tensor_map):
            shard, (dtype, shape, offsets) = tensor_map[tensor_name]
            if dtype not in component["allowed_dtypes"]:
                raise H3ManifestError(
                    f"{component['name']} tensor {tensor_name!r} uses "
                    f"unsupported dtype {dtype}"
                )
            byte_count = offsets[1] - offsets[0]
            encoded_bytes += byte_count
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            tensors.append(
                {
                    "component": component["name"],
                    "name": tensor_name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "shard": (
                        Path(component["directory"]) / shard
                    ).as_posix(),
                    "data_offsets": list(offsets),
                    "byte_count": byte_count,
                }
            )

        components.append(
            {
                "name": component["name"],
                "directory": component["directory"],
                "index": component["index"],
                "tensor_count": len(tensor_map),
                "encoded_tensor_bytes": encoded_bytes,
                "dtype_counts": dict(sorted(dtype_counts.items())),
                "shards": [
                    path.relative_to(root).as_posix()
                    for path in sorted(shard_files)
                ],
            }
        )
    return components, tensors


def build_manifest(
    root: Path, *, variant: str = "fl2va", progress: bool = False
) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise H3ManifestError(f"model root is not a directory: {root}")

    files = _artifact_files(root)
    _validate_selected_tree(root, files, variant=variant)
    contract = _validate_indexes_and_configs(root, variant=variant)
    components, tensors = _component_inventory(root, variant=variant)

    file_inventory = []
    total_bytes = 0
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        _revision, upstream_blob = _metadata_for(root, relative)
        if progress:
            print(
                f"[{index}/{len(files)}] sha256 {relative}",
                file=sys.stderr,
                flush=True,
            )
        size = path.stat().st_size
        total_bytes += size
        file_inventory.append(
            {
                "path": relative,
                "size": size,
                "sha256": _sha256_file(path),
                "upstream_blob": upstream_blob,
                "runtime_required": path.suffix != ".py",
            }
        )

    license_entry = next(
        entry for entry in file_inventory if entry["path"] == "LICENSE"
    )
    _require_equal(
        license_entry["sha256"], LICENSE_SHA256, "MiniMax H3 license hash"
    )

    model_kind = MODEL_KIND if variant == "fl2va" else REF2VA_MODEL_KIND
    other_variant = "Ref2VA" if variant == "fl2va" else "FL2VA"
    return {
        "schema": SCHEMA,
        "model_kind": model_kind,
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_storage": "mixed BF16/F32 safetensors",
        "license": {
            "name": "MiniMax H3 Community License Agreement",
            "date": "2026-08-02",
            "sha256": LICENSE_SHA256,
            "distribution": "operator-supplied; not distributed by gufo",
        },
        "reference_implementation": {
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
            "license": "MIT",
            "tensorops_ancestry": "ccv NAMatMul/NAInt8MatMul, BSD-3-Clause",
        },
        "contract": contract,
        "components": components,
        "file_count": len(file_inventory),
        "total_file_bytes": total_bytes,
        "files": file_inventory,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "unsupported": [
            other_variant,
            "H3-Regenerate-2K",
            "arbitrary revisions",
            "pickle checkpoints",
            "runtime execution of model-provided Python",
        ],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(manifest))


def verify_manifest(
    root: Path, manifest_path: Path, *, variant: str | None = None, progress: bool = False
):
    expected = _load_json(Path(manifest_path))
    if variant is None:
        variant = (
            "ref2va"
            if expected.get("model_kind") == REF2VA_MODEL_KIND
            else "fl2va"
        )
    actual = build_manifest(root, variant=variant, progress=progress)
    expected_bytes = _canonical_bytes(expected)
    actual_bytes = _canonical_bytes(actual)
    if expected_bytes != actual_bytes:
        raise H3ManifestError(
            f"source manifest differs from {manifest_path}; regenerate and review"
        )
    return actual

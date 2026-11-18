# Ornith-1.5-35B (qwen35moe)

Ornith-1.5-35B-A3B is a 35B-parameter (3B active) hybrid text model in the
Qwen3.5-MoE lineage: 40 layers alternating Gated DeltaNet linear attention and
full gated GQA attention (every fourth layer), 256 routed experts with 8 active
plus one shared expert, and a multi-token-prediction draft block stored inside
the model file (`nextn`). Gufo registers the architecture as `qwen35moe`.

## Acquisition

```sh
hf download ornith-ai/Ornith-1.5-35B-A3B-GGUF   Ornith-1.5-35B-Q4_K_M.gguf --local-dir models/ornith-1.5-35b
```

The Q8_0 artifact from the same repository runs natively end to end (raw
Q8_0 bytes on every path; see [BENCHMARKS.md](BENCHMARKS.md) for the
quant trade-off).

MIT licensed; the GGUF embeds the tokenizer and chat template. Optional vision
sidecar: `mmproj-Ornith-1.5-35B-BF16.gguf` (auto-discovered beside the model).

## Usage

```sh
gufo prompt -m models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf -p "Hello"
gufo serve llm -m models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf
gufo bench -m ... -p 2048 -n 128 -r 2
```

Speculative decoding (`--speculative mtp`) uses the in-file draft block; no
sidecar or `--mtp-model` is needed.

## DFlash draft decoding

`--speculative dflash2` runs the block-diffusion
[Ornith-1.5-35B-A3B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash)
drafter instead of the in-file MTP block. The draft is a 6-layer Qwen3-style
transformer (5 causal sliding-window layers + 1 bidirectional layer) that taps
8 trunk layer outputs, drafts 16-token blocks, and borrows the target's
embedding table and LM head. Convert the safetensors checkpoint to Gufo's
`dflash` GGUF:

```sh
python tools/ornith/convert_dflash_gguf.py \
  --checkpoint models/ornith-1.5-35b/dflash-src \
  --output models/ornith-1.5-35b/Ornith-1.5-35B-DFlash-Q8_0.gguf --q8
# Drop --q8 to keep the draft BF16 (larger, same acceptance).
```

```sh
gufo prompt -m models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf -p "Hello" \
  --speculative dflash2 \
  --dflash-model models/ornith-1.5-35b/Ornith-1.5-35B-DFlash-Q8_0.gguf
```

Verification stays lossless: greedy output is identical to AR (hash-checked on
Strix Halo), and sampled decoding uses target/draft rejection with residual
correction. Serving and benchmarks accept the same flags (`--draft-tokens`
caps proposals, `--draft-policy fixed|adaptive`; serve additionally requires
`--sessions 1`). The draft borrows the target tables in their native
quantized layout so shared weights stay bit-faithful.

When to prefer which backend on this model (single Strix Halo host,
Q4_K_M target, measured points in [BENCHMARKS.md](BENCHMARKS.md)):

- `--speculative mtp` is the default choice: +18–26% tg in every measured
  regime, and the draft block is quantization-aware (trained with the model).
- `--speculative dflash2` was trained against the BF16 target, so acceptance
  on the Q4_K_M target is modest on open-ended text (-14% tg at shallow
  depth). It wins when continuations are highly predictable: on the deep
  repetitive bench row it reaches 2.1× AR and 1.5× MTP at 100% acceptance.

## Notes

- Dense Q4_K/Q6_K projections are host-dequantized to Q8_0 at load; routed
  Q4_K experts keep their native layout on the MMQ/WMMA tiers. Q8_0
  artifacts stream through unchanged and skip the dequant pass entirely.
- Routed experts accept Q4_K/Q5_K/Q8_0/IQ3_S/IQ4_XS gate/up stacks and
  Q4_K/Q5_K/Q6_K/Q5_1/Q8_0/IQ3_S/IQ4_XS/IQ4_NL down stacks; other formats
  fail at load. Routers must be F32.
- The current Windows port qualifies this model on a single Strix Halo
  gfx1151 host; Linux numbers land in BENCHMARKS.md once measured.

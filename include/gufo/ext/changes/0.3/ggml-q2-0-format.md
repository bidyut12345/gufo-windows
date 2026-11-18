# Q2_0 block format support

Add `GGML_TYPE_Q2_0` (type `42`) storage and CPU dequantization (`gguf_reader.hpp`, `experimental/ggml_dequant.hpp`):
- `GgmlType` gains `kQ2_0 = 42` (`ToString` returns `"Q2_0"`).
- `block_q2_0` (`QK = 64`, 18 bytes: fp16 scale + 16 packed 2-bit bytes), `QuantizedBlockElements`, `QuantizedRowBytes`, `DequantizeQ2_0`, and `DequantizeRow` support `kQ2_0`.

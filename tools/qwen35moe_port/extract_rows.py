"""Extracts raw rows of one tensor from a GGUF file.

Usage: python extract_rows.py <model.gguf> <tensor.name> <rows> <out.bin>
Dumps the first `rows` rows of the tensor's raw (still-encoded) bytes so the
GPU vector-kernel validation can compare against the host dequantizer.
"""
import struct
import sys

path, tensor_name, n_rows, out_path = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
f = open(path, 'rb')


def rd(fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))


def rstr():
    (n,) = rd('<Q')
    return f.read(n).decode('utf8', 'replace')


magic, ver, n_tensors, n_kv = rd('<4sIQQ')
assert magic == b'GGUF', magic


def skip_value(t):
    if t == 0:
        rd('<c')
    elif t == 1:
        rd('<b')
    elif t == 2:
        rd('<H')
    elif t == 3:
        rd('<h')
    elif t == 4:
        rd('<I')
    elif t == 5:
        rd('<i')
    elif t == 6:
        rd('<f')
    elif t == 7:
        rd('<B')
    elif t == 8:
        rstr()
    elif t == 9:
        et, = rd('<I')
        n, = rd('<Q')
        for _ in range(n):
            skip_value(et)
    elif t == 10:
        rd('<Q')
    elif t == 11:
        rd('<q')
    elif t == 12:
        rd('<d')
    else:
        raise ValueError(t)


for _ in range(n_kv):
    rstr()
    t, = rd('<I')
    skip_value(t)

align = 32
for _ in range(n_tensors):
    name = rstr()
    nd, = rd('<I')
    dims = rd('<' + 'Q' * nd) if nd else ()
    tt, = rd('<I')
    off, = rd('<Q')
    if name == tensor_name:
        break
else:
    raise SystemExit(f'tensor {tensor_name} not found')

# The data section starts at the first aligned offset after the header; the
# tensor offset is relative to it. The header ends after the tensor infos.
end = f.tell()
data_start = (end + align - 1) // align * align
BB = {0: 4.0, 1: 2.0, 8: 34 / 32, 12: 144 / 256, 13: 176 / 256,
      14: 210 / 256, 20: 144 / 256, 23: 136 / 256, 21: 110 / 256, 30: 2.0}
row_bytes = int(round(dims[0] * BB[tt]))  # dims[0] is the row width (ne0)
assert abs(row_bytes - dims[0] * BB[tt]) < 1e-9, 'unaligned row'
f.seek(data_start + off)
remaining = n_rows
with open(out_path, 'wb') as out:
    while remaining:
        chunk = f.read(min(remaining, 1024) * row_bytes)
        assert len(chunk) % row_bytes == 0, 'truncated read'
        out.write(chunk)
        remaining -= len(chunk) // row_bytes
print(f'{tensor_name}: dims={dims} type={tt} row_bytes={row_bytes} '
      f'-> {out_path}')

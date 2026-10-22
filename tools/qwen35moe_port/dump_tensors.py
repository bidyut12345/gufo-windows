import struct, sys, re
from collections import defaultdict

path = sys.argv[1]
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
    if t == 0: rd('<c')[0]
    elif t == 1: rd('<b')[0]
    elif t == 2: rd('<H')[0]
    elif t == 3: rd('<h')[0]
    elif t == 4: rd('<I')[0]
    elif t == 5: rd('<i')[0]
    elif t == 6: rd('<f')[0]
    elif t == 7: rd('<B')[0]
    elif t == 8: rstr()
    elif t == 9:
        et, = rd('<I')
        n, = rd('<Q')
        for _ in range(n): skip_value(et)
    elif t == 10: rd('<Q')[0]
    elif t == 11: rd('<q')[0]
    elif t == 12: rd('<d')[0]
    else: raise ValueError(t)

for _ in range(n_kv):
    rstr(); t, = rd('<I'); skip_value(t)

# per-256-elem block bytes
BB = {0:None,1:None,8:None}
def bytes_per_elem(t, ne):
    if t == 0: return 4
    if t == 1: return 2
    if t == 8: return 34/32
    if t == 12: return 144/256
    if t == 13: return 176/256
    if t == 14: return 210/256
    if t == 20: return 144/256   # IQ4_NL 256-elem blocks
    if t == 23: return 136/256   # IQ4_XS 256-elem blocks
    if t == 21: return 110/256   # IQ3_S
    if t == 30: return 2
    raise ValueError(t)

TN = {0:'F32',1:'F16',8:'Q8_0',12:'Q4_K',13:'Q5_K',14:'Q6_K',20:'IQ4_NL',23:'IQ4_XS',21:'IQ3_S',30:'BF16'}

tensors = []
for _ in range(n_tensors):
    name = rstr()
    nd, = rd('<I')
    dims = rd('<' + 'Q' * nd) if nd else ()
    tt, = rd('<I')
    off, = rd('<Q')
    tensors.append((name, dims, tt))

def cat(name):
    if name == 'token_embd.weight': return 'embd'
    if name == 'output.weight': return 'head'
    if name.startswith('blk.40.'): return 'nextn'
    m = re.match(r'blk\.(\d+)\.(.+)', name)
    if not m: return 'other:' + name
    il = int(m.group(1)); rest = m.group(2)
    cls = 'gdn' if (il + 1) % 4 else 'gqa'
    return f'{cls}.{rest}'

rows = defaultdict(lambda: [0, 0.0, defaultdict(float)])
for name, dims, tt in tensors:
    ne = 1
    for d in dims: ne *= d
    b = ne * bytes_per_elem(tt, ne)
    c = cat(name)
    rows[c][0] += 1
    rows[c][1] += b
    rows[c][2][TN.get(tt, str(tt))] += b

total = sum(v[1] for v in rows.values())
print(f'{"category":44s} {"n":>3s} {"MB":>9s}  types')
for c in sorted(rows):
    n, b, types = rows[c]
    ts = ', '.join(f'{k}:{v/1e6:.1f}' for k, v in sorted(types.items(), key=lambda x: -x[1]))
    print(f'{c:44s} {n:3d} {b/1e6:9.1f}  {ts}')
print(f'{"TOTAL":44s}     {total/1e6:9.1f}')

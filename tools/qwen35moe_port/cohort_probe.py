import json, sys, threading, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18745"

PARA = ("The quick brown fox jumps over the lazy dog while the parser walks the "
        "token stream and the scheduler rebalances the cohort. Strix Halo executes "
        "the deterministic benchmark sequence on a unified memory bus. ")
text = PARA * 52  # ~2048 tokens (the driver's prompt length)

def post(payload, out, key):
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out[key] = json.load(r)

def fire(payloads):
    out = {}
    ts = [threading.Thread(target=post, args=(p, out, k)) for k, p in payloads.items()]
    for t in ts: t.start()
    for t in ts: t.join()
    return out

def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n

# 0) warmup round exactly like the driver: same prompt, 16 tokens, concurrent
fire({"a": {"prompt": text, "max_tokens": 16, "temperature": 0, "cache_prompt": True},
      "b": {"prompt": text, "max_tokens": 16, "temperature": 0, "cache_prompt": True}})

ref = fire({"k": {"prompt": text, "max_tokens": 128, "temperature": 0}})
ref = ref["k"]
ref_text = ref["choices"][0]["text"]
print("reference tokens:", ref["timings"]["predicted_n"], "prompt:", ref["timings"]["prompt_n"])
ref_text = ref_text

# 1) sequential preparation, batched release
fire({"a": {"prompt": text, "max_tokens": 1, "temperature": 0, "cache_prompt": True}})
fire({"b": {"prompt": text, "max_tokens": 1, "temperature": 0, "cache_prompt": True}})
rel = fire({"a": {"prompt": text, "max_tokens": 128, "temperature": 0, "cache_prompt": True},
            "b": {"prompt": text, "max_tokens": 128, "temperature": 0, "cache_prompt": True}})
ta, tb = rel["a"]["choices"][0]["text"], rel["b"]["choices"][0]["text"]
print("seq-prep batched-release: a==ref:", ta == ref_text, " b==ref:", tb == ref_text,
      " a==b:", ta == tb)
d = first_diff(ta, ref_text)
if d is not None:
    print("  first diff at char", d, repr(ref_text[max(0,d-20):d+20]), "|", repr(ta[max(0,d-20):d+20]))
print("  cached:", rel["a"]["timings"]["cache_n"], rel["b"]["timings"]["cache_n"])

# 2) concurrent preparation, batched release
conc = fire({"a": {"prompt": text, "max_tokens": 1, "temperature": 0, "cache_prompt": True},
             "b": {"prompt": text, "max_tokens": 1, "temperature": 0, "cache_prompt": True}})
rel = fire({"a": {"prompt": text, "max_tokens": 128, "temperature": 0, "cache_prompt": True},
            "b": {"prompt": text, "max_tokens": 128, "temperature": 0, "cache_prompt": True}})
ta, tb = rel["a"]["choices"][0]["text"], rel["b"]["choices"][0]["text"]
print("conc-prep batched-release: a==ref:", ta == ref_text, " b==ref:", tb == ref_text,
      " a==b:", ta == tb)
d = first_diff(ta, ref_text)
if d is not None:
    print("  first diff at char", d, repr(ref_text[max(0,d-20):d+20]), "|", repr(ta[max(0,d-20):d+20]))

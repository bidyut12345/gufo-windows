import hashlib, json, sys, threading, urllib.request

sys.path.insert(0, "D:/gufo/tools")
from gufo.model_bench.llm import synthetic_text, turn_prompt

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18745"
MODEL = "ornith"
KNOWN_C1 = "4f550c3b2ae69f58"  # single-ar-gufo C1 completion hash prefix


def stream(prompt, max_tokens, cache_prompt=None, client_id="c0"):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream",
                                          "X-Client-ID": client_id},
                                 method="POST")
    parts, usage = [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        data_lines = []
        for raw in r:
            line = raw.decode("utf-8").strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line == "" and data_lines:
                event = "\n".join(data_lines); data_lines = []
                if event == "[DONE]":
                    break
                chunk = json.loads(event)
                if chunk.get("choices"):
                    c = chunk["choices"][0].get("delta", {}).get("content")
                    if c:
                        parts.append(c)
                if chunk.get("usage"):
                    usage = chunk["usage"]
    return "".join(parts), usage


def fire(jobs):  # jobs: list of (key, kwargs)
    out = {}
    ts = [threading.Thread(target=lambda k=k, p=p: out.__setitem__(
        k, stream(**p))) for k, p in jobs]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


# calibrate the tokenizer like the driver
_, u = stream("Hi", 1)
overhead = u["prompt_tokens"] - 1
_, u = stream(synthetic_text(7777, 3000), 1)
ratio = (u["prompt_tokens"] - overhead) / 3000
prompt = turn_prompt(2048 - overhead, ratio, task="prose")
print(f"overhead={overhead} ratio={ratio:.3f} words={len(prompt.split())}")

ref, u = stream(prompt, 128)
ref_hash = hashlib.sha256(ref.encode()).hexdigest()
print("reference hash:", ref_hash[:16], "match single-ar C1:",
      ref_hash.startswith(KNOWN_C1), "prompt_n:", u["prompt_tokens"])

# driver sequence at C2: warmup(16) -> prep(1) -> release(128), all concurrent
fire([("a", dict(prompt=prompt, max_tokens=16, cache_prompt=True, client_id="wa")),
      ("b", dict(prompt=prompt, max_tokens=16, cache_prompt=True, client_id="wb"))])
fire([("a", dict(prompt=prompt, max_tokens=1, cache_prompt=True, client_id="pa")),
      ("b", dict(prompt=prompt, max_tokens=1, cache_prompt=True, client_id="pb"))])
rel = fire([("a", dict(prompt=prompt, max_tokens=128, cache_prompt=True, client_id="ra")),
            ("b", dict(prompt=prompt, max_tokens=128, cache_prompt=True, client_id="rb"))])
for k in ("a", "b"):
    text, u = rel[k]
    h = hashlib.sha256(text.encode()).hexdigest()
    print(f"release[{k}]: {h[:16]} match_ref={h == ref_hash} cached={u.get('cache_n')} prefill={u.get('prompt_n')}")

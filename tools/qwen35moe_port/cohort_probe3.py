import hashlib, json, sys, threading, urllib.request

sys.path.insert(0, "D:/gufo/tools")
from gufo.model_bench.llm import synthetic_text, turn_prompt

BASE = sys.argv[1]
STAGE = sys.argv[2]  # batched | seqprep | direct
KNOWN_C1 = "4f550c3b2ae69f58"
prompt = turn_prompt(2048 - 12, 1.167, task="prose")


def stream(prompt, max_tokens, cache_prompt=None, client_id="c0"):
    payload = {"model": "ornith", "messages": [{"role": "user", "content": prompt}],
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
    parts = []
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
    return "".join(parts)


def fire(jobs):
    out = {}
    ts = [threading.Thread(target=lambda k=k, p=p: out.__setitem__(k, stream(**p)))
          for k, p in jobs]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


def report(tag, text):
    h = hashlib.sha256(text.encode()).hexdigest()
    print(f"{tag}: {h[:16]} match_c1={h.startswith(KNOWN_C1)}")
    return h


if STAGE == "batched":       # driver-exact: concurrent prep, batched release
    fire([("a", dict(prompt=prompt, max_tokens=1, cache_prompt=True, client_id="pa")),
          ("b", dict(prompt=prompt, max_tokens=1, cache_prompt=True, client_id="pb"))])
    rel = fire([("a", dict(prompt=prompt, max_tokens=128, cache_prompt=True, client_id="ra")),
                ("b", dict(prompt=prompt, max_tokens=128, cache_prompt=True, client_id="rb"))])
    ha = report("release[a]", rel["a"]); hb = report("release[b]", rel["b"])
    print("a==b:", ha == hb)
elif STAGE == "seqprep":     # sequential prep, batched release
    stream(prompt, 1, cache_prompt=True, client_id="pa")
    stream(prompt, 1, cache_prompt=True, client_id="pb")
    rel = fire([("a", dict(prompt=prompt, max_tokens=128, cache_prompt=True, client_id="ra")),
                ("b", dict(prompt=prompt, max_tokens=128, cache_prompt=True, client_id="rb"))])
    ha = report("release[a]", rel["a"]); hb = report("release[b]", rel["b"])
    print("a==b:", ha == hb)
elif STAGE == "direct":      # no cache, two concurrent full requests
    rel = fire([("a", dict(prompt=prompt, max_tokens=128, client_id="da")),
                ("b", dict(prompt=prompt, max_tokens=128, client_id="db"))])
    ha = report("direct[a]", rel["a"]); hb = report("direct[b]", rel["b"])
    print("a==b:", ha == hb)

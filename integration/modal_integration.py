"""Integration test for rerankd against a real Ollama and llama-server, on Modal.

    ./integration/run.sh            # builds rerankd, then: modal run integration/modal_integration.py

What it checks, on a CPU container:
  1. Ollama (official release, checksum-pinned) serves; rerankd proxies it, including streaming.
  2. Two rerankers pulled through Ollama are served by llama-server (official llama.cpp release,
     checksum-pinned) via rerankd, in every response shape (/api/rerank, /v1/rerank, top_n,
     return_documents), with 0..1 scores.
  3. The same through Ollama's bundled llama-server (--use-ollama-engine).
  4. `rerankd check` accepts the good GGUFs and reports on a popular community upload.
  5. Rankings agree with the original Hugging Face models (reference image, CPU torch).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).parent
OLLAMA = ("v0.40.13", "c9490e0f520be21271d969e6703f012268b1d39f8b70dd58854e5bfef7a75bb6")
LLAMA = ("b11980", "589606b9cef85d05ab0d9db14c8307f2b1aef8621dfaaa0eea777492d66bf1e3")
LLAMA_CUDA = {  # CUDA 12.8 build plus its runtime libraries, for the GPU run
    "llama-b11980-bin-ubuntu-cuda-12.8-x64.tar.gz": "b8a992b3f63fcb47fe33f92aa4f587b2103ee85545fba06f1c94423e29afdabc",
    "cudart-llama-b11980-bin-ubuntu-cuda-12.8-x64.tar.gz": "e560a6efd043f7edac962c431423317f0e5a9cdb04bc44c2800ebb1565453f46",
}
MODELS = {
    "qwen3": {
        "ollama": "hf.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "hf": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    },
    "bge": {
        "ollama": "hf.co/gpustack/bge-reranker-v2-m3-GGUF:Q8_0",
        "hf": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    },
}
COMMUNITY = "dengcao/Qwen3-Reranker-0.6B:Q8_0"  # a widely shared upload, checked but not required to pass

volume = modal.Volume.from_name("rerankd-integration", create_if_missing=True)
app = modal.App("rerankd-integration")

def _image(llama_files: dict[str, str]) -> modal.Image:
    fetch = [
        f"curl -fsSL -o /tmp/{name} https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA[0]}/{name} && "
        f"echo '{sha}  /tmp/{name}' | sha256sum -c - && tar -xzf /tmp/{name} -C /opt/llama && rm /tmp/{name}"
        for name, sha in llama_files.items()
    ]
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl", "zstd", "ca-certificates", "libgomp1")
        .run_commands(
            f"curl -fsSL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/download/{OLLAMA[0]}/ollama-linux-amd64.tar.zst",
            f"echo '{OLLAMA[1]}  /tmp/ollama.tar.zst' | sha256sum -c -",
            "mkdir -p /opt/ollama && tar --zstd -xf /tmp/ollama.tar.zst -C /opt/ollama && rm /tmp/ollama.tar.zst",
            "mkdir -p /opt/llama", *fetch,
        )
        .env({"OLLAMA_MODELS": "/vol/ollama", "OLLAMA_HOST": "127.0.0.1:11434"})
        .add_local_file(HERE / "build" / "rerankd", "/usr/local/bin/rerankd")
        .add_local_file(HERE / "eval_pairs.json", "/root/eval_pairs.json")
    )


image = _image({f"llama-{LLAMA[0]}-bin-ubuntu-x64.tar.gz": LLAMA[1]})
gpu_image = _image(LLAMA_CUDA)

reference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(HERE / "requirements-reference.txt"))
    .env({"HF_HOME": "/vol/hf"})
    .add_local_file(HERE / "eval_pairs.json", "/root/eval_pairs.json")
)


def _http(method: str, url: str, body: dict | None = None, timeout: float = 600) -> tuple[int, dict | str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            try:
                return r.status, json.loads(raw)
            except ValueError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


def _wait(url: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up")


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail="") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
        return ok


def _rerankd(args: list[str], env: dict, log: Path) -> subprocess.Popen:
    p = subprocess.Popen(["rerankd", *args], env=env, stdout=log.open("ab"), stderr=subprocess.STDOUT)
    _wait("http://127.0.0.1:11435/rerankd/status", 30)
    return p


def _rerank_all(base: str, model: str, pairs: list[dict]) -> list[list[float]]:
    out = []
    for p in pairs:
        code, body = _http("POST", f"{base}/v1/rerank", {"model": model, "query": p["query"], "documents": p["documents"]})
        if code != 200:
            raise RuntimeError(f"rerank failed: {code} {body}")
        scores = [0.0] * len(p["documents"])
        for r in body["results"]:
            scores[r["index"]] = r["relevance_score"]
        out.append(scores)
    return out


@app.function(image=image, cpu=4, memory=8192, volumes={"/vol": volume}, timeout=45 * 60)
def integration() -> dict:
    return _integration(gpu=False)


@app.function(image=gpu_image, gpu="L4", volumes={"/vol": volume}, timeout=45 * 60)
def integration_gpu() -> dict:
    return _integration(gpu=True)


def _integration(gpu: bool) -> dict:
    checks = Checks()
    logs = Path("/tmp/logs")
    logs.mkdir(exist_ok=True)
    llama_server = next(Path("/opt/llama").rglob("llama-server"))
    lib_dirs = sorted({str(p.parent) for p in Path("/opt/llama").rglob("*.so*")} | {str(llama_server.parent)})
    env = dict(os.environ, PATH="/opt/ollama/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=":".join(lib_dirs))
    if gpu:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        checks.add("GPU visible", smi.returncode == 0, smi.stdout.strip())
    ollama = subprocess.Popen(["/opt/ollama/bin/ollama", "serve"], env=env,
                              stdout=(logs / "ollama.log").open("wb"), stderr=subprocess.STDOUT)
    _wait("http://127.0.0.1:11434/api/version", 60)
    _, ver = _http("GET", "http://127.0.0.1:11434/api/version")
    checks.add("ollama serves", True, ver)

    rd = _rerankd(["--engine", str(llama_server), "--idle", "10m", "--max-models", "2"] + (["--engine-logs"] if gpu else []),
                  env, logs / "rerankd.log")
    base = "http://127.0.0.1:11435"

    # Streaming passthrough: pull through rerankd with stream=true and time the lines.
    for key, m in [*MODELS.items(), ("community", {"ollama": COMMUNITY})]:
        req = urllib.request.Request(f"{base}/api/pull", data=json.dumps({"model": m["ollama"], "stream": True}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        stamps, last = [], {}
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                stamps.append(time.time())
                last = json.loads(line)
        checks.add(f"pull {m['ollama']} through rerankd (streamed)", last.get("status") == "success",
                   f"{len(stamps)} chunks over {stamps[-1] - stamps[0]:.1f}s")
    code, tags = _http("GET", f"{base}/api/tags")
    checks.add("proxied /api/tags", code == 200 and len(tags.get("models", [])) >= 2, f"{len(tags.get('models', []))} models")

    # rerankd check
    for key, m in MODELS.items():
        r = subprocess.run(["rerankd", "check", m["ollama"]], env=env, capture_output=True, text=True)
        checks.add(f"rerankd check {m['ollama']}", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr.strip())
    r = subprocess.run(["rerankd", "check", COMMUNITY], env=env, capture_output=True, text=True)
    community = (r.stdout + r.stderr).strip()
    checks.items.append({"check": f"rerankd check {COMMUNITY} (informational)", "ok": None, "detail": community[-600:]})
    print("INFO community upload:", community[-600:], flush=True)

    pairs = json.loads(Path("/root/eval_pairs.json").read_text())
    q, docs = pairs[0]["query"], pairs[0]["documents"]
    results: dict = {}
    for key, m in MODELS.items():
        t0 = time.time()
        code, body = _http("POST", f"{base}/api/rerank", {"model": m["ollama"], "query": q, "documents": docs})
        cold = time.time() - t0
        ok = code == 200 and len(body.get("results", [])) == len(docs)
        scores = [r["relevance_score"] for r in body.get("results", [])] if ok else []
        checks.add(f"{key}: /api/rerank (Dify shape)", ok and scores == sorted(scores, reverse=True)
                   and all(0 <= s <= 1 for s in scores), f"cold start {cold:.1f}s; top index {body['results'][0]['index'] if ok else body}")
        t0 = time.time()
        code, body = _http("POST", f"{base}/v1/rerank", {"model": m["ollama"], "query": q, "documents": docs,
                                                         "top_n": 2, "return_documents": True})
        warm = time.time() - t0
        ok = code == 200 and len(body["results"]) == 2 and "text" in body["results"][0]["document"]
        checks.add(f"{key}: /v1/rerank top_n + documents", ok, f"warm {warm * 1000:.0f} ms for {len(docs)} documents")
        t0 = time.time()
        results[key] = _rerank_all(base, m["ollama"], pairs)
        results[key + "_seconds"] = round(time.time() - t0, 2)
    code, body = _http("POST", f"{base}/v1/rerank", {"model": "nope:latest", "query": "x", "documents": ["a"]})
    checks.add("unknown model -> 404", code == 404, str(body)[:120])
    code, body = _http("POST", f"{base}/v1/rerank", {"model": "../../etc", "query": "x", "documents": ["a"]})
    checks.add("bad model name -> 400", code == 400)
    if gpu:
        # Definitive: is a llama-server process holding GPU memory right now?
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        # Inside a container nvidia-smi cannot name our processes (it shows the container's init), so
        # check the memory: Ollama has no model loaded here, so GPU memory in use is the rerankers'.
        used = sum(int(a.rsplit(",", 1)[-1].strip().split()[0]) for a in apps.splitlines() if a.strip())
        on_gpu = [f"{used} MiB of GPU memory in use with the rerankers loaded"] if used > 500 else []
        log = (logs / "rerankd.log").read_text(errors="replace")
        cuda_lines = [line.strip() for line in log.splitlines() if "CUDA" in line or "offload" in line][:6]
        checks.add("llama-server runs on the GPU", bool(on_gpu), "; ".join(on_gpu) or f"nvidia-smi: {apps or 'no processes'}")
        checks.items.append({"check": "engine log (CUDA lines)", "ok": None, "detail": cuda_lines})
    _, status = _http("GET", f"{base}/rerankd/status")
    checks.add("status lists loaded engines", len(status.get("engines", [])) >= 1, f"{len(status['engines'])} engine(s)")
    rd.terminate()
    rd.wait(20)

    # Ollama's bundled llama-server.
    try:
        rd = _rerankd(["--use-ollama-engine", "--allow-untested-ollama"], env, logs / "rerankd-ollama-engine.log")
        code, body = _http("POST", f"{base}/v1/rerank", {"model": MODELS["qwen3"]["ollama"], "query": q, "documents": docs})
        bundled = None
        if code == 200:
            bundled = [0.0] * len(docs)
            for r in body["results"]:
                bundled[r["index"]] = r["relevance_score"]
        checks.add("--use-ollama-engine serves qwen3", code == 200, str(body)[:200] if code != 200 else "")
        if bundled:
            ref = results["qwen3"][0]
            diff = max(abs(a - b) for a, b in zip(bundled, ref))
            same_order = sorted(range(len(ref)), key=lambda i: -ref[i]) == sorted(range(len(ref)), key=lambda i: -bundled[i])
            checks.add("bundled engine ranks like the llama.cpp release", same_order and diff < 0.05,
                       f"same order: {same_order}, max score diff {diff:.4f}")
        rd.terminate()
        rd.wait(20)
    except Exception as e:  # noqa: BLE001
        checks.add("--use-ollama-engine starts", False, str(e))

    ollama.terminate()
    tails = {p.name: p.read_text(errors="replace")[-1500:] for p in logs.iterdir()}
    return {"checks": checks.items, "scores": results, "ollama": ver, "llama_cpp": LLAMA[0], "logs": tails,
            "hardware": "L4 GPU" if gpu else "4 CPU"}


@app.function(image=reference_image, cpu=4, memory=8192, volumes={"/vol": volume}, timeout=45 * 60)
def reference() -> dict:
    """Scores from the original Hugging Face models, following their model cards."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_grad_enabled(False)
    pairs = json.loads(Path("/root/eval_pairs.json").read_text())
    out: dict = {}

    name, rev = MODELS["qwen3"]["hf"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=torch.float32).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
              "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instruct = "Given a web search query, retrieve relevant passages that answer the query"
    scores = []
    for p in pairs:
        texts = [f"{prefix}<Instruct>: {instruct}\n<Query>: {p['query']}\n<Document>: {d}{suffix}" for d in p["documents"]]
        batch = tok(texts, padding=True, return_tensors="pt")
        logits = model(**batch).logits[:, -1, :]
        probs = torch.softmax(torch.stack([logits[:, no], logits[:, yes]], dim=1), dim=1)[:, 1]
        scores.append([round(float(x), 6) for x in probs])
    out["qwen3"] = scores

    name, rev = MODELS["bge"]["hf"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev, torch_dtype=torch.float32).eval()
    scores = []
    for p in pairs:
        batch = tok([[p["query"], d] for d in p["documents"]], padding=True, truncation=True, max_length=512,
                    return_tensors="pt")
        logits = model(**batch).logits.view(-1)
        scores.append([round(float(torch.sigmoid(x)), 6) for x in logits])
    out["bge"] = scores
    return out


def kendall_tau(a: list[float], b: list[float]) -> float:
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += (1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1 if (a[i] - a[j]) * (b[i] - b[j]) < 0 else 0)
    return s / (n * (n - 1) / 2)


@app.local_entrypoint()
def main(gpu: bool = False) -> None:
    ref_call = reference.spawn()
    res = (integration_gpu if gpu else integration).remote()
    ref = ref_call.get()
    summary = {"ollama": res["ollama"], "llama_cpp": res["llama_cpp"], "hardware": res["hardware"], "agreement": {}}
    for key in MODELS:
        got, want = res["scores"].get(key), ref.get(key)
        if not got or not want:
            continue
        taus = [kendall_tau(g, w) for g, w in zip(got, want)]
        top1 = sum(1 for g, w in zip(got, want) if g.index(max(g)) == w.index(max(w)))
        maxdiff = max(abs(x - y) for g, w in zip(got, want) for x, y in zip(g, w))
        mean_tau = sum(taus) / len(taus)
        summary["agreement"][key] = {"mean_kendall_tau": round(mean_tau, 3), "min_tau": round(min(taus), 3),
                                     "top1": f"{top1}/{len(taus)}", "max_score_diff": round(maxdiff, 4),
                                     "seconds_for_8_queries": res["scores"].get(key + "_seconds")}
        ok = mean_tau >= 0.8 and top1 >= len(taus) - 1
        res["checks"].append({"check": f"{key}: ranking matches the Hugging Face model", "ok": ok,
                              "detail": summary["agreement"][key]})
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / ("latest-gpu.json" if gpu else "latest.json")).write_text(json.dumps({"summary": summary, **res, "reference": ref}, indent=2))
    failed = [c for c in res["checks"] if c["ok"] is False]
    print(json.dumps(summary, indent=2))
    for c in res["checks"]:
        mark = "PASS" if c["ok"] else ("INFO" if c["ok"] is None else "FAIL")
        print(f"{mark}  {c['check']}" + (f"  ({c['detail']})" if c["detail"] and c["ok"] is not True else ""))
    print(f"\n{len(failed)} failed. Full results in {out}")
    if failed:
        for name, tail in res["logs"].items():
            print(f"\n--- {name} (tail) ---\n{tail}")

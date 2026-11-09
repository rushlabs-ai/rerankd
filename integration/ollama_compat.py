"""Weekly check that rerankd's --use-ollama-engine still works with the newest Ollama release.

Standard library only. Used by .github/workflows/ollama-compat.yml:

    python3 integration/ollama_compat.py pick [TAG]   # prints OLLAMA_TAG=... and OLLAMA_SHA256=...
    python3 integration/ollama_compat.py check        # Ollama on PATH, ./rerankd built

`pick` only accepts a release that is at least 7 days old and not a pre-release, and only when the
checksum GitHub recorded for the upload matches the one in Ollama's own sha256sum.txt.
`check` serves Qwen3-Reranker-0.6B through Ollama's bundled llama-server and compares the
rankings with the Hugging Face reference in integration/reference-qwen3.json.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ASSET = "ollama-linux-amd64.tar.zst"
MIN_AGE = datetime.timedelta(days=7)
MODEL = "hf.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
OLLAMA = "http://127.0.0.1:11434"
RERANKD = "http://127.0.0.1:11435"


def github(path: str, accept: str = "application/vnd.github+json") -> bytes:
    req = urllib.request.Request(f"https://api.github.com{path}", headers={"Accept": accept})
    if os.environ.get("GH_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GH_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def pick(tag: str | None) -> None:
    releases = json.loads(github("/repos/ollama/ollama/releases?per_page=30"))
    now = datetime.datetime.now(datetime.timezone.utc)

    def age(r: dict) -> datetime.timedelta:
        return now - datetime.datetime.fromisoformat(r["published_at"].replace("Z", "+00:00"))

    if tag:
        rel = next((r for r in releases if r["tag_name"] == tag), None)
        if rel is None:
            sys.exit(f"{tag}: not among the 30 newest Ollama releases")
    else:
        rel = next((r for r in releases if not r["prerelease"] and not r["draft"] and age(r) >= MIN_AGE), None)
        if rel is None:
            sys.exit("no Ollama release is at least 7 days old")
    if rel["prerelease"] or age(rel) < MIN_AGE:
        sys.exit(f"{rel['tag_name']}: pre-release or less than 7 days old ({age(rel).days} days); not testing it yet")
    assets = {a["name"]: a for a in rel["assets"]}
    digest = assets[ASSET]["digest"].removeprefix("sha256:")
    sums = github(f"/repos/ollama/ollama/releases/assets/{assets['sha256sum.txt']['id']}", "application/octet-stream").decode()
    listed = next((line.split()[0] for line in sums.splitlines() if line.strip().endswith(ASSET)), None)
    if listed != digest:
        sys.exit(f"{rel['tag_name']}: checksum mismatch: GitHub {digest}, sha256sum.txt {listed}")
    print(f"OLLAMA_TAG={rel['tag_name']}")
    print(f"OLLAMA_SHA256={digest}")


def http(method: str, url: str, body: dict | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:500]}


def wait_for(url: str, seconds: int = 60) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    sys.exit(f"{url} did not come up")


def kendall_tau(a: list[float], b: list[float]) -> float:
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            x = (a[i] - a[j]) * (b[i] - b[j])
            s += 1 if x > 0 else -1 if x < 0 else 0
    return s / (n * (n - 1) / 2)


def check() -> None:
    pairs = json.loads((HERE / "eval_pairs.json").read_text())
    reference = json.loads((HERE / "reference-qwen3.json").read_text())["scores"]
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_MODELS=str(Path.home() / "ollama-models"))
    ollama = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rerankd = None
    try:
        wait_for(f"{OLLAMA}/api/version")
        _, ver = http("GET", f"{OLLAMA}/api/version")
        code, body = http("POST", f"{OLLAMA}/api/pull", {"model": MODEL, "stream": False}, timeout=1800)
        if code != 200:
            sys.exit(f"pull failed: {code} {body}")
        rerankd = subprocess.Popen(["./rerankd", "--use-ollama-engine", "--allow-untested-ollama", "--gpu-layers", "0"],
                                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for(f"{RERANKD}/rerankd/status")
        taus, top1, started = [], 0, time.time()
        for pair, want in zip(pairs, reference):
            code, body = http("POST", f"{RERANKD}/v1/rerank", {"model": MODEL, "query": pair["query"], "documents": pair["documents"]})
            if code != 200:
                sys.exit(f"--use-ollama-engine failed with Ollama {ver.get('version')}: {code} {body}")
            got = [0.0] * len(pair["documents"])
            for r in body["results"]:
                got[r["index"]] = r["relevance_score"]
            taus.append(kendall_tau(got, want))
            top1 += got.index(max(got)) == want.index(max(want))
        mean_tau = sum(taus) / len(taus)
        ok = mean_tau >= 0.8 and top1 >= len(taus) - 1
        report = (f"Ollama {ver.get('version')}: --use-ollama-engine {'works' if ok else 'FAILS'}. "
                  f"Rankings vs the Hugging Face model: mean Kendall tau {mean_tau:.3f}, top result {top1}/{len(taus)}, "
                  f"{time.time() - started:.1f}s for {len(taus)} queries.")
        print(report)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(report + ("\n\nTo accept this version, raise the upper bound of `testedOllama` in main.go "
                                  "after a full run of integration/modal_integration.py.\n" if ok else "\n"))
        if not ok:
            sys.exit(1)
    finally:
        for p in (rerankd, ollama):
            if p:
                p.terminate()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "pick":
        pick(sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None)
    elif len(sys.argv) == 2 and sys.argv[1] == "check":
        check()
    else:
        sys.exit(__doc__)

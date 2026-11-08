# rerankd

**Rerank for Ollama.** Ollama has no rerank endpoint ([ollama#3368](https://github.com/ollama/ollama/issues/3368)),
so apps that call `/api/rerank` or `/v1/rerank` on it get a 404. rerankd sits next to Ollama, answers
those endpoints with the reranker model you pulled with Ollama, and passes every other request
through to Ollama unchanged, streaming included.

```
your app ──► rerankd :11435 ──► Ollama :11434          (chat, embeddings, everything else)
                     └────────► llama-server --rerank  (started on demand, stopped when idle)
```

- One small static binary, Go standard library only.
- Uses the model files Ollama already downloaded: rerankd asks Ollama where they are (`/api/show`).
- Checks a GGUF before loading it, and explains how to fix the common broken Qwen3-Reranker conversions.
- Scores are 0..1 for every model; Jina/Cohere-style responses.

## Quick start

1. Pull a reranker into Ollama. These two GGUFs pass `rerankd check`:

   ```sh
   ollama pull hf.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF          # Apache-2.0, multilingual
   ollama pull hf.co/gpustack/bge-reranker-v2-m3-GGUF:Q8_0            # Apache-2.0, multilingual
   ```

2. Have a `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) on your PATH
   (`brew install llama.cpp`, `winget install llama.cpp`, or a release build). rerankd starts it only
   when a rerank request arrives.

3. Run rerankd and point your app at port **11435** instead of 11434:

   ```sh
   rerankd
   curl http://127.0.0.1:11435/v1/rerank -d '{
     "model": "hf.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
     "query": "What is the capital of France?",
     "documents": ["Paris is the capital of France.", "Berlin is in Germany."]
   }'
   ```

## Endpoints

| Path | Request | Response |
|---|---|---|
| `POST /api/rerank` | `model`, `query`, `documents` (strings or `{"text"}`), optional `top_n`, `return_documents` | `results: [{index, relevance_score, document?}]`, sorted best first. `document` is a plain string here (what Dify's Ollama plugin reads). |
| `POST /v1/rerank`, `/v2/rerank`, `/rerank` (also `/v1/reranking`, `/reranking`) | same | same, with `document: {"text": ...}` (Jina and Cohere style), plus `id`, `model`, `usage`, `meta`. |
| `GET /rerankd/status` | | loaded rerankers, their process ids and last use |
| everything else | | proxied to Ollama |

Documents are returned only when `return_documents` is true. `top_n` limits the results.

**Scores.** Qwen3 rerankers give probabilities. BERT-style cross-encoders (bge-reranker) give
logits, which rerankd maps to 0..1 with a sigmoid so score thresholds work the same for every model
(`--raw-scores` turns this off).

## Using it from apps

- **Dify** (Ollama model provider): add a model of type *Rerank* with Base URL `http://<host>:11435`.
  The plugin calls `/api/rerank`.
- **Cherry Studio**: set the Ollama provider's address to `http://<host>:11435` and choose the reranker
  model in the knowledge base. It calls `/v1/rerank`.
- Anything that speaks the Jina or Cohere rerank API: base URL `http://127.0.0.1:11435`.

These request and response formats are covered by the test suite. Please open an issue if an app
behaves differently.

## Engines

| Mode | How | When |
|---|---|---|
| Your llama-server (default) | `rerankd` or `rerankd --engine /path/to/llama-server` | Recommended. Starts one llama-server per reranker on demand. |
| Your running server | `rerankd --backend http://127.0.0.1:8012` | You already run `llama-server --rerank -m model.gguf` yourself. Add `--sigmoid` for BERT-style models. |
| Ollama's bundled llama-server | `rerankd --use-ollama-engine` | **Experimental.** No separate llama.cpp install. Ollama's bundled build is an internal detail that can change between releases, so rerankd checks that it supports `--rerank` and accepts only Ollama versions it was tested with (0.40.13 so far; `--allow-untested-ollama` overrides). |

**GPU memory.** The reranker runs in its own process, which Ollama's scheduler does not see. A 0.6B
reranker needs well under 1 GB, but on a full GPU use `--gpu-layers 0` to keep it on the CPU, and
rerankd stops it after `--idle` (default 5 minutes) without requests.

## Check a model

```sh
rerankd check hf.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF   # an Ollama model name
rerankd check ./some-reranker.gguf                           # or a file
```

Many Qwen3-Reranker GGUFs shared online were converted as plain chat models: they lack the
classification head (`cls.output.weight`) and the rerank prompt, and give meaningless scores.
rerankd refuses to load them and says so; re-convert with a current llama.cpp, which detects
Qwen3-Reranker, or use one of the models above.

## Docker and remote Ollama

rerankd reads the model file itself, so it must see Ollama's `models` folder. If Ollama runs in a
container or on another machine, mount the folder and pass `--models-dir /path/to/models`.

## Flags

```
--listen 127.0.0.1:11435     where rerankd listens (RERANKD_LISTEN)
--ollama http://127.0.0.1:11434   Ollama (RERANKD_OLLAMA)
--engine PATH                llama-server to run
--backend URL                use a llama-server you run (--sigmoid for logit models)
--use-ollama-engine          use Ollama's bundled llama-server (experimental)
--allow-untested-ollama      ... even on an Ollama version not tested with rerankd
--models-dir DIR             Ollama's models folder, if not visible at the reported path
--ctx 4096                   tokens per query+document pair
--parallel 2                 pairs scored at once per model
--gpu-layers N               llama-server -ngl ("0" = CPU only)
--idle 5m                    stop a reranker after this long unused
--max-models 1               rerankers loaded at once
--max-docs 1000              documents per request
--raw-scores                 return logits unchanged
--engine-arg ARG             extra llama-server argument (repeatable)
--engine-logs                show llama-server output
```

## Security

- Listens on 127.0.0.1 by default. On loopback it rejects requests whose `Host` is not a local name,
  so a web page cannot reach it (or Ollama behind it) through DNS rebinding.
- rerankd adds no authentication. If you listen on another interface, anyone who can reach it can
  use your Ollama; put it behind something that authenticates.
- Model names are validated strictly; the model file must be an Ollama blob (`sha256-…`); llama-server
  is started without a shell, bound to 127.0.0.1 on a random port.
- GGUF headers are parsed with size limits, so a corrupt file cannot exhaust memory.

Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## Install

Download a release for your platform, check it against `SHA256SUMS`, and optionally verify its
build provenance:

```sh
gh attestation verify rerankd_*_linux_amd64.tar.gz --repo rushlabs-ai/rerankd
```

Or build from source (Go 1.24+): `go install github.com/rushlabs-ai/rerankd@latest`.

## Tested against the real thing

`integration/modal_integration.py` runs rerankd on a [Modal](https://modal.com) container with the
official Ollama 0.40.13 and llama.cpp b11980 release builds (both checksum-pinned), and compares
the rankings with the original Hugging Face models run in PyTorch. Latest runs, 8 queries × 6 documents:

| Model (GGUF via Ollama) | Top result matches the original | Mean Kendall τ (CPU / GPU) | Warm request, 6 documents (4 CPU / L4 GPU) |
|---|---|---|---|
| ggml-org/Qwen3-Reranker-0.6B-Q8_0 | 8/8 | 0.98 / 1.00 | ~570 ms / ~95 ms |
| gpustack/bge-reranker-v2-m3 Q8_0 | 8/8 | 0.98 / 0.99 | ~165 ms / ~55 ms |

The GPU run uses llama.cpp's official CUDA 12.8 build and confirms the rerankers sit in GPU memory.

The same run checks every endpoint, streaming passthrough, `--use-ollama-engine` (identical
scores), and `rerankd check`. It also confirmed that one widely shared Qwen3-Reranker upload on
the Ollama library (`dengcao/Qwen3-Reranker-0.6B:Q8_0`) has no reranking head: rerankd refuses it.
Run it yourself with `./integration/run.sh` (add `--gpu` for the L4 run).

## Development

```sh
go vet ./... && go test -race ./...
```

The tests use a fake Ollama and a fake llama-server (the test binary itself), and cover the three
response shapes, streaming passthrough, DNS-rebinding rejection, GGUF parsing of good, broken and
hostile files, and the engine lifecycle (start on demand, health wait, eviction, idle stop, restart).

## Licence

Apache-2.0. Built by [Rush Labs](https://rushlabs.ai), an applied AI lab.

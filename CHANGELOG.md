# Changelog

## 0.1.1

- `--use-ollama-engine` accepts Ollama 0.40.11 to 0.40.13 (was 0.40.13 only); each passed the full
  integration run.
- A weekly CI check tests the newest Ollama release once it is at least 7 days old, with its
  checksum confirmed against both GitHub and Ollama's sha256sum.txt.

## 0.1.0 (2026-11-08)

- `/api/rerank` (Dify), `/v1/rerank`, `/v2/rerank`, `/rerank` with Jina/Cohere-style requests and
  responses; everything else proxied to Ollama, streaming included.
- Finds the model file through Ollama's `/api/show`; `--models-dir` for containers.
- Starts llama-server per reranker on demand, stops it when idle; or forwards to your own server.
- `rerankd check`: tells whether a GGUF can be served as a reranker, and why not.
- 0..1 scores for every model (sigmoid for logit models); DNS-rebinding protection on loopback.

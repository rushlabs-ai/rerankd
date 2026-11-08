// rerankd adds a rerank endpoint to Ollama.
//
// It listens next to Ollama, serves /api/rerank, /v1/rerank and /rerank, and passes
// every other request through to Ollama unchanged (streaming included). Rerank
// requests are answered by llama-server running the reranker model you already
// pulled with Ollama.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var version = "0.1.0-dev"

// Ollama releases whose bundled llama-server passed integration/modal_integration.py (inclusive
// range). Other versions need --allow-untested-ollama.
var testedOllama = [2]string{"0.40.13", "0.40.13"}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, " ") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "rerankd:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) > 0 && args[0] == "check" {
		return runCheck(args[1:])
	}
	fs := flag.NewFlagSet("rerankd", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: rerankd [flags]\n       rerankd check <model name or .gguf path>\n\nFlags:")
		fs.PrintDefaults()
	}
	listen := fs.String("listen", envOr("RERANKD_LISTEN", "127.0.0.1:11435"), "address to listen on")
	ollamaURL := fs.String("ollama", envOr("RERANKD_OLLAMA", "http://127.0.0.1:11434"), "Ollama base URL")
	backendURL := fs.String("backend", "", "use a llama-server you run yourself (started with --rerank), e.g. http://127.0.0.1:8012")
	sigmoid := fs.Bool("sigmoid", false, "with --backend: apply a sigmoid to scores (for BERT-style rerankers that return logits)")
	engine := fs.String("engine", "", "llama-server binary to start per model (default: llama-server on PATH)")
	useOllama := fs.Bool("use-ollama-engine", false, "use the llama-server bundled with Ollama instead of your own")
	ollamaLib := fs.String("ollama-lib", "", "Ollama's lib/ollama folder (default: found next to the ollama binary)")
	allowUntested := fs.Bool("allow-untested-ollama", false, "with --use-ollama-engine: allow Ollama versions rerankd has not been tested with")
	modelsDir := fs.String("models-dir", os.Getenv("OLLAMA_MODELS"), "Ollama's models folder as seen from here (only needed for Docker or remote mounts)")
	ctxSize := fs.Int("ctx", 4096, "tokens per query+document pair")
	parallel := fs.Int("parallel", 2, "pairs scored in parallel per model")
	gpuLayers := fs.String("gpu-layers", "", `layers on the GPU (llama-server -ngl); "0" for CPU only; default: llama-server decides`)
	idle := fs.Duration("idle", 5*time.Minute, "stop a reranker after this long unused (0: never)")
	maxModels := fs.Int("max-models", 1, "reranker models loaded at once")
	maxDocs := fs.Int("max-docs", 1000, "documents per request")
	rawScores := fs.Bool("raw-scores", false, "return logits as-is instead of 0..1 scores")
	engineLogs := fs.Bool("engine-logs", false, "show llama-server's output")
	var engineArgs multiFlag
	fs.Var(&engineArgs, "engine-arg", "extra llama-server argument (repeatable), e.g. --engine-arg=--flash-attn=on")
	showVersion := fs.Bool("version", false, "print the version")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *showVersion {
		fmt.Println("rerankd", version)
		return nil
	}

	ollama, err := url.Parse(*ollamaURL)
	if err != nil || (ollama.Scheme != "http" && ollama.Scheme != "https") || ollama.Host == "" {
		return fmt.Errorf("--ollama must be an http(s) URL, got %q", *ollamaURL)
	}

	var backend Backend
	switch {
	case *backendURL != "":
		u, err := url.Parse(*backendURL)
		if err != nil || u.Host == "" {
			return fmt.Errorf("--backend must be a URL, got %q", *backendURL)
		}
		kind := ScoreProbability
		if *sigmoid {
			kind = ScoreLogit
		}
		backend = &StaticBackend{Base: strings.TrimRight(*backendURL, "/"), Kind: kind}
		slog.Info("using your llama-server", "url", *backendURL)
	default:
		binary, libDir, err := findEngine(*engine, *useOllama, *ollamaLib)
		if err != nil {
			return err
		}
		if *useOllama {
			if err := checkOllamaVersion(ollama, *allowUntested); err != nil {
				return err
			}
		}
		if err := ProbeRerank(binary, libDir); err != nil {
			return err
		}
		var logs io.Writer
		if *engineLogs {
			logs = os.Stderr
		}
		resolver := &Resolver{Ollama: ollama.String(), ModelsDir: *modelsDir, Client: &http.Client{}}
		backend = NewManagedBackend(EngineConfig{
			Binary: binary, LibDir: libDir, Ctx: *ctxSize, Parallel: *parallel, GPULayers: *gpuLayers,
			ExtraArgs: engineArgs, Idle: *idle, MaxEngines: *maxModels, Logs: logs,
		}, resolver)
		slog.Info("engine", "llama-server", binary)
	}
	defer backend.Close()

	srv := NewServer(backend, ollama)
	srv.RawScores = *rawScores
	srv.MaxDocs = *maxDocs
	host, _, err := net.SplitHostPort(*listen)
	if err != nil {
		return fmt.Errorf("--listen: %w", err)
	}
	srv.LocalOnly = loopbackHost(host)

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return err
	}
	httpSrv := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}
	slog.Info("rerankd listening", "address", ln.Addr().String(), "ollama", ollama.String(), "version", version)
	if !srv.LocalOnly {
		slog.Warn("listening beyond this machine: anyone who can reach this address can use Ollama through rerankd")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errc := make(chan error, 1)
	go func() { errc <- httpSrv.Serve(ln) }()
	select {
	case <-ctx.Done():
	case err := <-errc:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return httpSrv.Shutdown(shutdown)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// findEngine returns the llama-server binary to run and, for Ollama's bundled one, its lib dir.
func findEngine(explicit string, useOllama bool, ollamaLib string) (string, string, error) {
	exe := "llama-server"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	if !useOllama {
		name := explicit
		if name == "" {
			name = exe
		}
		path, err := exec.LookPath(name)
		if err != nil {
			return "", "", fmt.Errorf("llama-server not found (%v). Install llama.cpp, pass --engine /path/to/llama-server, "+
				"run your own and pass --backend URL, or try --use-ollama-engine", err)
		}
		return path, "", nil
	}
	dir := ollamaLib
	if dir == "" {
		ollamaBin, err := exec.LookPath("ollama")
		if err != nil {
			return "", "", errors.New("--use-ollama-engine: ollama not found on PATH; pass --ollama-lib")
		}
		if real, err := filepath.EvalSymlinks(ollamaBin); err == nil {
			ollamaBin = real
		}
		base := filepath.Dir(ollamaBin)
		// Ollama's own search order for its helpers (llm/llama_binary.go).
		candidates := []string{filepath.Join(base, "..", "lib", "ollama"), filepath.Join(base, "lib", "ollama"), base}
		for _, c := range candidates {
			if st, err := os.Stat(filepath.Join(c, exe)); err == nil && st.Mode().IsRegular() {
				dir = filepath.Clean(c)
				break
			}
		}
		if dir == "" {
			return "", "", fmt.Errorf("--use-ollama-engine: no bundled %s next to %s; pass --ollama-lib", exe, ollamaBin)
		}
	}
	return filepath.Join(dir, exe), dir, nil
}

func checkOllamaVersion(ollama *url.URL, allowUntested bool) error {
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Get(strings.TrimRight(ollama.String(), "/") + "/api/version")
	if err != nil {
		return fmt.Errorf("--use-ollama-engine needs Ollama running to check its version: %w", err)
	}
	defer resp.Body.Close()
	var v struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		return fmt.Errorf("reading Ollama's version: %w", err)
	}
	tested := "none yet"
	if testedOllama[0] != "" {
		if versionBetween(v.Version, testedOllama[0], testedOllama[1]) {
			return nil
		}
		tested = testedOllama[0] + " to " + testedOllama[1]
	}
	if allowUntested {
		slog.Warn("Ollama version not tested with rerankd", "version", v.Version, "tested", tested)
		return nil
	}
	return fmt.Errorf("Ollama %s has not been tested with --use-ollama-engine (tested: %s). "+
		"Use your own llama-server (--engine), or pass --allow-untested-ollama", v.Version, tested)
}

// versionBetween compares dotted numeric versions; pre-release suffixes are ignored.
func versionBetween(v, lo, hi string) bool {
	return compareVersions(v, lo) >= 0 && compareVersions(v, hi) <= 0
}

func compareVersions(a, b string) int {
	pa, pb := versionParts(a), versionParts(b)
	for i := 0; i < 3; i++ {
		if pa[i] != pb[i] {
			if pa[i] < pb[i] {
				return -1
			}
			return 1
		}
	}
	return 0
}

func versionParts(v string) [3]int {
	v = strings.TrimPrefix(v, "v")
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		v = v[:i]
	}
	var out [3]int
	for i, p := range strings.SplitN(v, ".", 3) {
		n, err := strconv.Atoi(p)
		if err != nil {
			return [3]int{-1, -1, -1}
		}
		out[i] = n
	}
	return out
}

// runCheck reports whether a model (an Ollama name or a GGUF path) can be served as a reranker.
func runCheck(args []string) error {
	fs := flag.NewFlagSet("rerankd check", flag.ContinueOnError)
	ollamaURL := fs.String("ollama", envOr("RERANKD_OLLAMA", "http://127.0.0.1:11434"), "Ollama base URL")
	modelsDir := fs.String("models-dir", os.Getenv("OLLAMA_MODELS"), "Ollama's models folder as seen from here")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return errors.New("usage: rerankd check <model name or .gguf path>")
	}
	target := fs.Arg(0)
	path := target
	if st, err := os.Stat(target); err != nil || !st.Mode().IsRegular() {
		r := &Resolver{Ollama: *ollamaURL, ModelsDir: *modelsDir, Client: &http.Client{}}
		if path, err = r.Resolve(context.Background(), target); err != nil {
			return err
		}
	}
	info, err := ReadGGUF(path)
	if err != nil {
		return err
	}
	fmt.Printf("file:          %s\narchitecture:  %s\npooling:       %d (4 = rank)\nrerank head:   %v\nrerank prompt: %v\n",
		path, info.Architecture, info.PoolingType, info.Tensors["cls.output.weight"] || info.Tensors["cls.weight"], info.HasRerankTpl)
	kind, err := CheckReranker(info)
	if err != nil {
		return err
	}
	scores := "probabilities (0..1)"
	if kind == ScoreLogit {
		scores = "logits, mapped to 0..1 with a sigmoid unless --raw-scores"
	}
	fmt.Printf("OK: llama-server can serve this as a reranker. Scores: %s.\n", scores)
	return nil
}

package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// The test binary doubles as a fake llama-server when RERANKD_FAKE_ENGINE is set.
func TestMain(m *testing.M) {
	if os.Getenv("RERANKD_FAKE_ENGINE") == "1" {
		fakeLlamaServer(os.Args[1:])
		return
	}
	os.Exit(m.Run())
}

func fakeLlamaServer(args []string) {
	for _, a := range args {
		if a == "--help" {
			fmt.Println("usage: llama-server ... --rerank, --reranking  enable reranking endpoint")
			return
		}
	}
	fs := flag.NewFlagSet("fake", flag.ContinueOnError)
	port := fs.Int("port", 0, "")
	model := fs.String("m", "", "")
	fs.Bool("rerank", false, "")
	fs.String("host", "", "")
	fs.String("ctx-size", "", "")
	fs.String("parallel", "", "")
	fs.String("batch-size", "", "")
	fs.String("ubatch-size", "", "")
	fs.String("n-gpu-layers", "", "")
	if err := fs.Parse(args); err != nil || *model == "" {
		os.Exit(2)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(*port))
	if err != nil {
		os.Exit(3)
	}
	var healthCalls atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		if healthCalls.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable) // "loading model"
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.Handle("POST /v1/rerank", fakeEngineHandler(nil))
	_ = http.Serve(ln, mux)
}

func managed(t *testing.T, ollama *fakeOllama, idle time.Duration, maxEngines int) *ManagedBackend {
	t.Helper()
	t.Setenv("RERANKD_FAKE_ENGINE", "1")
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := ProbeRerank(exe, ""); err != nil {
		t.Fatal(err)
	}
	m := NewManagedBackend(EngineConfig{Binary: exe, Ctx: 1024, Parallel: 1, Idle: idle, MaxEngines: maxEngines,
		StartWait: 20 * time.Second}, &Resolver{Ollama: ollama.URL, Client: http.DefaultClient})
	t.Cleanup(m.Close)
	return m
}

func addModel(t *testing.T, ollama *fakeOllama, dir, name string, b *ggufBuilder) {
	t.Helper()
	blob := filepath.Join(dir, "sha256-"+fmt.Sprintf("%064x", len(ollama.models)+1))
	if err := os.WriteFile(blob, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	ollama.models[name] = blob
}

func TestManagedEngineLifecycle(t *testing.T) {
	ollama := newFakeOllama(t)
	dir := t.TempDir()
	addModel(t, ollama, dir, "rr", qwenReranker())
	addModel(t, ollama, dir, "rr2", qwenReranker())
	addModel(t, ollama, dir, "chat", (&ggufBuilder{}).String("general.architecture", "qwen3").Tensor("output.weight"))
	m := managed(t, ollama, 1500*time.Millisecond, 1)
	u, _ := url.Parse(ollama.URL)
	srv := NewServer(m, u)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, out := post(t, ts.URL+"/api/rerank", map[string]any{"model": "rr", "query": "capital of France", "documents": docs})
	if code != 200 || len(out["results"].([]any)) != 3 {
		t.Fatalf("rerank via managed engine: %d %v", code, out)
	}
	if st := m.Status(); len(st) != 1 || st[0].PID == 0 || st[0].Model != "rr" {
		t.Fatalf("status: %+v", st)
	}

	// A second model replaces the first (max one loaded).
	if code, out := post(t, ts.URL+"/v1/rerank", map[string]any{"model": "rr2", "query": "x", "documents": []string{"a"}}); code != 200 {
		t.Fatalf("rr2: %d %v", code, out)
	}
	if st := m.Status(); len(st) != 1 || st[0].Model != "rr2" {
		t.Fatalf("eviction: %+v", st)
	}

	// A chat model converted without a rerank head is refused before anything starts.
	code, out = post(t, ts.URL+"/v1/rerank", map[string]any{"model": "chat", "query": "x", "documents": []string{"a"}})
	if code != http.StatusUnprocessableEntity || !strings.Contains(fmt.Sprint(out), "cls.output.weight") {
		t.Fatalf("chat model: %d %v", code, out)
	}
	code, _ = post(t, ts.URL+"/v1/rerank", map[string]any{"model": "nope", "query": "x", "documents": []string{"a"}})
	if code != http.StatusNotFound {
		t.Fatalf("unknown model: %d", code)
	}
	code, _ = post(t, ts.URL+"/v1/rerank", map[string]any{"model": "../../x", "query": "x", "documents": []string{"a"}})
	if code != http.StatusBadRequest {
		t.Fatalf("bad name: %d", code)
	}

	// Idle engines stop.
	deadline := time.Now().Add(6 * time.Second)
	for len(m.Status()) > 0 && time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
	}
	if st := m.Status(); len(st) != 0 {
		t.Fatalf("idle engine still running: %+v", st)
	}
	// And start again on the next request.
	if code, _ := post(t, ts.URL+"/v1/rerank", map[string]any{"model": "rr", "query": "x", "documents": []string{"a"}}); code != 200 {
		t.Fatalf("restart: %d", code)
	}
}

func TestEngineThatDiesIsReported(t *testing.T) {
	ollama := newFakeOllama(t)
	dir := t.TempDir()
	addModel(t, ollama, dir, "rr", qwenReranker())
	t.Setenv("RERANKD_FAKE_ENGINE", "1")
	exe, _ := os.Executable()
	m := NewManagedBackend(EngineConfig{Binary: exe, ExtraArgs: []string{"--unknown-flag"}, StartWait: 10 * time.Second},
		&Resolver{Ollama: ollama.URL, Client: http.DefaultClient})
	defer m.Close()
	_, _, _, err := m.Acquire(context.Background(), "rr")
	if err == nil || !strings.Contains(err.Error(), "exited while loading") {
		t.Fatalf("want exit error, got %v", err)
	}
	if len(m.Status()) != 0 {
		t.Fatal("a failed engine must not stay registered")
	}
}

func TestProbeRejectsBinaryWithoutRerank(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses /bin/true")
	}
	if err := ProbeRerank("/bin/true", ""); err == nil {
		t.Fatal("expected an error for a binary without --rerank")
	}
}

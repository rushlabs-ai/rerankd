package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Backend answers rerank requests in llama-server's Jina-style format.
type Backend interface {
	// URL returns the base URL of a llama-server that serves model, starting one if needed.
	Acquire(ctx context.Context, model string) (base string, kind ScoreKind, release func(), err error)
	Close()
	Status() []EngineStatus
}

// EngineStatus is reported on /rerankd/status.
type EngineStatus struct {
	Model    string    `json:"model"`
	URL      string    `json:"url"`
	PID      int       `json:"pid,omitempty"`
	Started  time.Time `json:"started"`
	LastUsed time.Time `json:"last_used"`
	InFlight int       `json:"in_flight"`
}

// StaticBackend forwards every request to one llama-server the user runs themselves.
type StaticBackend struct {
	Base string
	Kind ScoreKind
}

func (s *StaticBackend) Acquire(context.Context, string) (string, ScoreKind, func(), error) {
	return s.Base, s.Kind, func() {}, nil
}
func (s *StaticBackend) Close()                 {}
func (s *StaticBackend) Status() []EngineStatus { return []EngineStatus{{URL: s.Base}} }

// EngineConfig controls llama-server processes started by rerankd.
type EngineConfig struct {
	Binary     string        // llama-server path
	LibDir     string        // extra library dir (Ollama's bundled build)
	Ctx        int           // context size per sequence
	Parallel   int           // sequences in parallel
	GPULayers  string        // -ngl value; "" lets llama-server decide
	ExtraArgs  []string      // appended verbatim (from --engine-arg)
	Idle       time.Duration // stop an engine after this long unused
	MaxEngines int           // models loaded at once
	StartWait  time.Duration
	Logs       io.Writer
}

type engine struct {
	model    string
	path     string
	kind     ScoreKind
	base     string
	cmd      *exec.Cmd
	done     chan struct{}
	started  time.Time
	lastUsed time.Time
	inFlight int
	ready    chan struct{} // closed when loaded (or failed)
	err      error
}

// ManagedBackend starts one llama-server per reranker model on demand and stops idle ones.
type ManagedBackend struct {
	cfg      EngineConfig
	resolver *Resolver
	check    func(path string) (ScoreKind, error)
	client   *http.Client

	mu      sync.Mutex
	engines map[string]*engine
	stop    chan struct{}
}

func NewManagedBackend(cfg EngineConfig, r *Resolver) *ManagedBackend {
	if cfg.MaxEngines < 1 {
		cfg.MaxEngines = 1
	}
	if cfg.StartWait == 0 {
		cfg.StartWait = 3 * time.Minute
	}
	m := &ManagedBackend{
		cfg:      cfg,
		resolver: r,
		check:    checkFile,
		client:   &http.Client{Timeout: 5 * time.Second},
		engines:  map[string]*engine{},
		stop:     make(chan struct{}),
	}
	go m.reaper()
	return m
}

func checkFile(path string) (ScoreKind, error) {
	info, err := ReadGGUF(path)
	if err != nil {
		return 0, err
	}
	return CheckReranker(info)
}

func (m *ManagedBackend) Acquire(ctx context.Context, model string) (string, ScoreKind, func(), error) {
	path, err := m.resolver.Resolve(ctx, model)
	if err != nil {
		return "", 0, nil, err
	}
	m.mu.Lock()
	_, loaded := m.engines[path]
	m.mu.Unlock()
	var kind ScoreKind
	if !loaded {
		// Read the GGUF header outside the lock; it can take a moment for a large vocabulary.
		if kind, err = m.check(path); err != nil {
			return "", 0, nil, &modelError{model: model, err: err}
		}
	}
	m.mu.Lock()
	e, ok := m.engines[path]
	if !ok && loaded {
		m.mu.Unlock() // it was stopped in between; start over (and check the file again)
		return m.Acquire(ctx, model)
	}
	if !ok {
		m.evictLocked()
		e = &engine{model: model, path: path, kind: kind, ready: make(chan struct{}), started: time.Now()}
		m.engines[path] = e
		go m.start(e)
	}
	e.inFlight++
	e.lastUsed = time.Now()
	m.mu.Unlock()

	release := func() {
		m.mu.Lock()
		e.inFlight--
		e.lastUsed = time.Now()
		m.mu.Unlock()
	}
	select {
	case <-e.ready:
	case <-ctx.Done():
		release()
		return "", 0, nil, ctx.Err()
	}
	if e.err != nil {
		release()
		m.mu.Lock()
		if m.engines[path] == e {
			delete(m.engines, path)
		}
		m.mu.Unlock()
		return "", 0, nil, e.err
	}
	return e.base, e.kind, release, nil
}

// modelError is a problem with the model itself (a client error, not a server fault).
type modelError struct {
	model string
	err   error
}

func (e *modelError) Error() string { return fmt.Sprintf("%s: %v", e.model, e.err) }
func (e *modelError) Unwrap() error { return e.err }

// evictLocked stops the least recently used idle engine when at capacity.
func (m *ManagedBackend) evictLocked() {
	for len(m.engines) >= m.cfg.MaxEngines {
		var victim *engine
		for _, e := range m.engines {
			if e.inFlight == 0 && (victim == nil || e.lastUsed.Before(victim.lastUsed)) {
				victim = e
			}
		}
		if victim == nil {
			return // all busy: allow one over the limit rather than fail the request
		}
		delete(m.engines, victim.path)
		go stopEngine(victim)
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func (m *ManagedBackend) args(path string, port int) []string {
	parallel := max(1, m.cfg.Parallel)
	ctx := max(512, m.cfg.Ctx)
	args := []string{
		"-m", path,
		"--rerank",
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--ctx-size", strconv.Itoa(ctx * parallel),
		"--parallel", strconv.Itoa(parallel),
		// A rerank pair must fit in one micro-batch.
		"--batch-size", strconv.Itoa(ctx),
		"--ubatch-size", strconv.Itoa(ctx),
	}
	if m.cfg.GPULayers != "" {
		args = append(args, "--n-gpu-layers", m.cfg.GPULayers)
	}
	return append(args, m.cfg.ExtraArgs...)
}

func (m *ManagedBackend) start(e *engine) {
	defer close(e.ready)
	port, err := freePort()
	if err != nil {
		e.err = err
		return
	}
	e.base = fmt.Sprintf("http://127.0.0.1:%d", port)
	cmd := exec.Command(m.cfg.Binary, m.args(e.path, port)...)
	cmd.Env = engineEnv(m.cfg.LibDir)
	if m.cfg.Logs != nil {
		cmd.Stdout, cmd.Stderr = m.cfg.Logs, m.cfg.Logs
	}
	setProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		e.err = fmt.Errorf("starting llama-server: %w", err)
		return
	}
	e.cmd = cmd
	e.done = make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(e.done)
	}()
	slog.Info("loading reranker", "model", e.model, "pid", cmd.Process.Pid, "port", port)
	deadline := time.Now().Add(m.cfg.StartWait)
	for time.Now().Before(deadline) {
		select {
		case <-e.done:
			e.err = fmt.Errorf("llama-server exited while loading %s; run rerankd with --engine-logs to see why", e.model)
			return
		case <-time.After(250 * time.Millisecond):
		}
		resp, err := m.client.Get(e.base + "/health")
		if err != nil {
			continue
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			slog.Info("reranker ready", "model", e.model, "seconds", time.Since(e.started).Round(time.Millisecond).Seconds())
			return
		}
	}
	stopEngine(e)
	e.err = fmt.Errorf("llama-server did not become ready for %s within %s", e.model, m.cfg.StartWait)
}

func engineEnv(libDir string) []string {
	env := os.Environ()
	if libDir == "" {
		return env
	}
	// Ollama's bundled llama-server loads its GPU backends from lib/ollama and its subfolders.
	dirs := []string{libDir}
	if subs, err := os.ReadDir(libDir); err == nil {
		for _, s := range subs {
			if s.IsDir() {
				dirs = append(dirs, filepath.Join(libDir, s.Name()))
			}
		}
	}
	key := "LD_LIBRARY_PATH"
	if old := os.Getenv(key); old != "" {
		dirs = append(dirs, old)
	}
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		if !strings.HasPrefix(kv, key+"=") {
			out = append(out, kv)
		}
	}
	return append(out, key+"="+strings.Join(dirs, string(os.PathListSeparator)))
}

func stopEngine(e *engine) {
	if e.cmd == nil || e.cmd.Process == nil {
		return
	}
	slog.Info("stopping reranker", "model", e.model, "pid", e.cmd.Process.Pid)
	terminate(e.cmd)
	select {
	case <-e.done:
	case <-time.After(10 * time.Second):
		_ = e.cmd.Process.Kill()
		<-e.done
	}
}

func (m *ManagedBackend) reaper() {
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-tick.C:
		}
		m.mu.Lock()
		for path, e := range m.engines {
			select {
			case <-e.ready:
			default:
				continue // still loading
			}
			exited := false
			if e.done != nil {
				select {
				case <-e.done:
					exited = true
				default:
				}
			}
			if exited || (e.inFlight == 0 && m.cfg.Idle > 0 && time.Since(e.lastUsed) > m.cfg.Idle) {
				delete(m.engines, path)
				go stopEngine(e)
			}
		}
		m.mu.Unlock()
	}
}

func (m *ManagedBackend) Close() {
	close(m.stop)
	m.mu.Lock()
	engines := m.engines
	m.engines = map[string]*engine{}
	m.mu.Unlock()
	var wg sync.WaitGroup
	for _, e := range engines {
		wg.Add(1)
		go func(e *engine) {
			defer wg.Done()
			<-e.ready
			stopEngine(e)
		}(e)
	}
	wg.Wait()
}

func (m *ManagedBackend) Status() []EngineStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := []EngineStatus{}
	for _, e := range m.engines {
		s := EngineStatus{Model: e.model, URL: e.base, Started: e.started, LastUsed: e.lastUsed, InFlight: e.inFlight}
		if e.cmd != nil && e.cmd.Process != nil {
			s.PID = e.cmd.Process.Pid
		}
		out = append(out, s)
	}
	return out
}

// ProbeRerank checks that a llama-server binary supports --rerank (via --help).
func ProbeRerank(binary, libDir string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "--help")
	cmd.Env = engineEnv(libDir)
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	_ = cmd.Run() // some builds exit non-zero after printing help
	if !strings.Contains(out.String(), "--rerank") {
		if ctx.Err() != nil {
			return errors.New("llama-server --help timed out")
		}
		return fmt.Errorf("%s does not list --rerank in its --help output", binary)
	}
	return nil
}

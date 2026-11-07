package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sort"
	"strings"
	"time"
)

// Server serves the rerank endpoints and proxies everything else to Ollama.
type Server struct {
	Backend     Backend
	Ollama      *url.URL
	RawScores   bool // never apply a sigmoid to logit scores
	MaxDocs     int
	MaxBody     int64
	LocalOnly   bool // reject non-loopback Host headers (DNS rebinding), like Ollama does
	StaticModel string
	client      *http.Client
	proxy       *httputil.ReverseProxy
}

func NewServer(backend Backend, ollama *url.URL) *Server {
	s := &Server{Backend: backend, Ollama: ollama, MaxDocs: 1000, MaxBody: 16 << 20,
		client: &http.Client{Timeout: 10 * time.Minute}}
	s.proxy = &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(ollama)
			r.Out.Host = ollama.Host
		},
		FlushInterval: -1, // stream chat and generate responses as they arrive
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Warn("ollama unreachable", "path", r.URL.Path, "error", err)
			writeError(w, http.StatusBadGateway, "upstream_error", "Ollama is not reachable at "+ollama.String())
		},
	}
	return s
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	for _, p := range []string{"/api/rerank", "/v1/rerank", "/v2/rerank", "/rerank", "/v1/reranking", "/reranking"} {
		dify := p == "/api/rerank"
		mux.HandleFunc("POST "+p, func(w http.ResponseWriter, r *http.Request) { s.rerank(w, r, dify) })
	}
	mux.HandleFunc("GET /rerankd/status", s.status)
	mux.Handle("/", s.proxy)
	return s.hostCheck(mux)
}

// hostCheck rejects requests whose Host is not a loopback name when rerankd listens on
// loopback, so a web page cannot reach it through DNS rebinding.
func (s *Server) hostCheck(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.LocalOnly && !loopbackHost(r.Host) {
			writeError(w, http.StatusForbidden, "forbidden", "host not allowed")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func loopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.Trim(strings.ToLower(host), "[]")
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

type rerankRequest struct {
	Model           string            `json:"model"`
	Query           json.RawMessage   `json:"query"`
	Documents       []json.RawMessage `json:"documents"`
	TopN            *int              `json:"top_n"`
	ReturnDocuments *bool             `json:"return_documents"`
}

type engineResult struct {
	Index int     `json:"index"`
	Score float64 `json:"relevance_score"`
}

type engineResponse struct {
	Results []engineResult `json:"results"`
	Usage   struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// text accepts "a string" or {"text": "a string"} (Jina and Cohere both allow either).
func text(raw json.RawMessage) (string, bool) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, true
	}
	var obj struct {
		Text *string `json:"text"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Text != nil {
		return *obj.Text, true
	}
	return "", false
}

func (s *Server) rerank(w http.ResponseWriter, r *http.Request, dify bool) {
	started := time.Now()
	r.Body = http.MaxBytesReader(w, r.Body, s.MaxBody)
	var req rerankRequest
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request body too large")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body")
		return
	}
	model := req.Model
	if model == "" {
		model = s.StaticModel
	}
	if model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", `"model" is required`)
		return
	}
	query, ok := text(req.Query)
	if !ok || query == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", `"query" must be a non-empty string`)
		return
	}
	if len(req.Documents) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", `"documents" must be a non-empty array`)
		return
	}
	if len(req.Documents) > s.MaxDocs {
		writeError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("at most %d documents per request", s.MaxDocs))
		return
	}
	docs := make([]string, len(req.Documents))
	for i, raw := range req.Documents {
		if docs[i], ok = text(raw); !ok {
			writeError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf(`documents[%d] must be a string or {"text": "..."}`, i))
			return
		}
	}
	if req.TopN != nil && *req.TopN < 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", `"top_n" must be >= 0`)
		return
	}

	base, kind, release, err := s.Backend.Acquire(r.Context(), model)
	if err != nil {
		s.backendError(w, model, err)
		return
	}
	defer release()

	payload, _ := json.Marshal(map[string]any{"model": model, "query": query, "documents": docs, "top_n": len(docs)})
	ereq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, base+"/v1/rerank", bytes.NewReader(payload))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "could not build the engine request")
		return
	}
	ereq.Header.Set("Content-Type", "application/json")
	eresp, err := s.client.Do(ereq)
	if err != nil {
		slog.Warn("engine request failed", "model", model, "error", err)
		writeError(w, http.StatusBadGateway, "upstream_error", "the reranker engine did not answer")
		return
	}
	defer eresp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(eresp.Body, 64<<20))
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", "reading the engine response failed")
		return
	}
	if eresp.StatusCode != http.StatusOK {
		msg := engineMessage(body)
		slog.Warn("engine error", "model", model, "status", eresp.StatusCode, "message", msg)
		status := http.StatusBadGateway
		if eresp.StatusCode == http.StatusBadRequest {
			status = http.StatusBadRequest // e.g. a document longer than the context
		}
		writeError(w, status, "upstream_error", "reranker engine: "+msg)
		return
	}
	var er engineResponse
	if err := json.Unmarshal(body, &er); err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", "the engine returned invalid JSON")
		return
	}

	results := make([]map[string]any, 0, len(er.Results))
	returnDocs := req.ReturnDocuments != nil && *req.ReturnDocuments
	sort.SliceStable(er.Results, func(i, j int) bool { return er.Results[i].Score > er.Results[j].Score })
	for _, res := range er.Results {
		if res.Index < 0 || res.Index >= len(docs) {
			continue
		}
		score := res.Score
		if kind == ScoreLogit && !s.RawScores {
			score = 1 / (1 + math.Exp(-score))
		}
		item := map[string]any{"index": res.Index, "relevance_score": score}
		if returnDocs {
			if dify {
				item["document"] = docs[res.Index] // Dify's Ollama plugin reads a plain string
			} else {
				item["document"] = map[string]string{"text": docs[res.Index]}
			}
		}
		results = append(results, item)
	}
	if req.TopN != nil && *req.TopN > 0 && *req.TopN < len(results) {
		results = results[:*req.TopN]
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      requestID(),
		"object":  "list",
		"model":   model,
		"results": results,
		"usage":   map[string]int{"prompt_tokens": er.Usage.PromptTokens, "total_tokens": er.Usage.TotalTokens},
		"meta":    map[string]any{"billed_units": map[string]int{"search_units": 1}},
	})
	slog.Info("rerank", "model", model, "documents", len(docs), "ms", time.Since(started).Milliseconds())
}

func (s *Server) backendError(w http.ResponseWriter, model string, err error) {
	var me *modelError
	switch {
	case errors.Is(err, ErrNotFound):
		writeError(w, http.StatusNotFound, "not_found_error", err.Error())
	case errors.As(err, &me):
		writeError(w, http.StatusUnprocessableEntity, "invalid_request_error", err.Error())
	case errors.Is(err, ErrInvalidName):
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
	case errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded):
		writeError(w, http.StatusServiceUnavailable, "server_error", "request cancelled while the model was loading")
	default:
		slog.Warn("backend error", "model", model, "error", err)
		writeError(w, http.StatusServiceUnavailable, "server_error", err.Error())
	}
}

func engineMessage(body []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	msg := strings.TrimSpace(string(body))
	if len(msg) > 300 {
		msg = msg[:300]
	}
	if msg == "" {
		msg = "no details"
	}
	return msg
}

func (s *Server) status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"version": version, "engines": s.Backend.Status()})
}

func requestID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return "rerank-" + hex.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"type": typ, "message": msg}})
}

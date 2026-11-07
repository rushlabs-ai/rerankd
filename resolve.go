package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Ollama model names: [host/][namespace/]model[:tag]. Strict on purpose: the name ends up
// in a request to Ollama and, indirectly, selects a file to load.
var modelNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,79}(/[A-Za-z0-9][A-Za-z0-9._-]{0,79}){0,2}(:[A-Za-z0-9][A-Za-z0-9._-]{0,79})?$`)

var blobPattern = regexp.MustCompile(`^sha256-[0-9a-f]{64}$`)

// ValidModelName reports whether name is a plausible Ollama model name.
func ValidModelName(name string) bool {
	return len(name) <= 256 && modelNamePattern.MatchString(name) && !strings.Contains(name, "..")
}

// Resolver finds the GGUF file behind an Ollama model name by asking Ollama itself
// (POST /api/show), so it follows Ollama's own store layout across versions.
type Resolver struct {
	Ollama    string // base URL, e.g. http://127.0.0.1:11434
	ModelsDir string // optional: Ollama's models dir as seen from here (Docker, remote mounts)
	Client    *http.Client
}

// ErrNotFound means Ollama does not know the model.
var ErrNotFound = errors.New("model not found")

// ErrInvalidName is returned for names that fail ValidModelName.
var ErrInvalidName = errors.New("invalid model name")

type showResponse struct {
	Modelfile string `json:"modelfile"`
	Details   struct {
		Format string `json:"format"`
	} `json:"details"`
}

// Resolve returns the local path of the model's GGUF blob.
func (r *Resolver) Resolve(ctx context.Context, name string) (string, error) {
	if !ValidModelName(name) {
		return "", fmt.Errorf("%w %q", ErrInvalidName, name)
	}
	body, _ := json.Marshal(map[string]any{"model": name})
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(r.Ollama, "/")+"/api/show", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("asking Ollama about %s: %w", name, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode == http.StatusNotFound {
		return "", fmt.Errorf("%w: %s (pull it with: ollama pull %s)", ErrNotFound, name, name)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama /api/show returned %d for %s", resp.StatusCode, name)
	}
	var show showResponse
	if err := json.Unmarshal(data, &show); err != nil {
		return "", fmt.Errorf("Ollama /api/show: %w", err)
	}
	if show.Details.Format != "" && show.Details.Format != "gguf" {
		return "", fmt.Errorf("%s is stored as %s, not GGUF; llama-server needs a GGUF model", name, show.Details.Format)
	}
	from := modelfileFrom(show.Modelfile)
	if from == "" {
		return "", fmt.Errorf("Ollama did not report a model file for %s (is it a cloud or remote model?)", name)
	}
	return r.localPath(from)
}

// modelfileFrom returns the first FROM path in a generated Modelfile (comments skipped).
func modelfileFrom(modelfile string) string {
	sc := bufio.NewScanner(strings.NewReader(modelfile))
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if len(line) > 5 && strings.EqualFold(line[:5], "FROM ") {
			return strings.TrimSpace(line[5:])
		}
	}
	return ""
}

// localPath checks that a path Ollama reported is a blob file we can read. When Ollama runs
// elsewhere (a container), the blob is looked up under ModelsDir/blobs instead.
func (r *Resolver) localPath(reported string) (string, error) {
	base := filepath.Base(filepath.FromSlash(strings.ReplaceAll(reported, `\`, "/")))
	if !blobPattern.MatchString(base) {
		return "", fmt.Errorf("Ollama reported %q, which is not a model blob", reported)
	}
	candidates := []string{}
	if filepath.IsAbs(reported) {
		candidates = append(candidates, filepath.Clean(reported))
	}
	if r.ModelsDir != "" {
		candidates = append(candidates, filepath.Join(r.ModelsDir, "blobs", base))
	}
	for _, c := range candidates {
		st, err := os.Stat(c)
		if err == nil && st.Mode().IsRegular() {
			return c, nil
		}
	}
	if r.ModelsDir == "" {
		return "", fmt.Errorf("model file %s is not readable from here; if Ollama runs in a container or on "+
			"another machine, pass --models-dir with Ollama's models folder", reported)
	}
	return "", fmt.Errorf("model file %s not found under %s", base, filepath.Join(r.ModelsDir, "blobs"))
}

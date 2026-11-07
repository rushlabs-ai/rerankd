package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// GGUF value types (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
const (
	ggufUint8 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

// Limits that keep a hostile or corrupt file from exhausting memory.
const (
	maxGGUFString = 16 << 20
	maxGGUFKVs    = 1 << 16
	maxGGUFTensor = 1 << 20
	maxGGUFArray  = 1 << 26
	maxGGUFDims   = 8
)

// llama.cpp's LLAMA_POOLING_TYPE_RANK: the classification head used by rerankers.
const poolingRank = 4

// GGUFInfo is the subset of GGUF metadata rerankd needs.
type GGUFInfo struct {
	Version      uint32
	Architecture string
	PoolingType  int64 // -1 when absent
	HasRerankTpl bool  // tokenizer.chat_template.rerank
	Labels       []string
	Tensors      map[string]bool
}

type ggufReader struct {
	r       *bufio.Reader
	version uint32
}

func (g *ggufReader) u32() (uint32, error) {
	var v uint32
	err := binary.Read(g.r, binary.LittleEndian, &v)
	return v, err
}

func (g *ggufReader) u64() (uint64, error) {
	var v uint64
	err := binary.Read(g.r, binary.LittleEndian, &v)
	return v, err
}

// count reads a length: 32-bit in GGUF v1, 64-bit from v2 on.
func (g *ggufReader) count() (uint64, error) {
	if g.version == 1 {
		v, err := g.u32()
		return uint64(v), err
	}
	return g.u64()
}

func (g *ggufReader) str() (string, error) {
	n, err := g.count()
	if err != nil {
		return "", err
	}
	if n > maxGGUFString {
		return "", fmt.Errorf("gguf: string of %d bytes is too long", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func (g *ggufReader) skip(n uint64) error {
	for n > 0 {
		step := n
		if step > 1<<30 {
			step = 1 << 30
		}
		if _, err := g.r.Discard(int(step)); err != nil {
			return err
		}
		n -= step
	}
	return nil
}

func scalarSize(t uint32) (uint64, bool) {
	switch t {
	case ggufUint8, ggufInt8, ggufBool:
		return 1, true
	case ggufUint16, ggufInt16:
		return 2, true
	case ggufUint32, ggufInt32, ggufFloat32:
		return 4, true
	case ggufUint64, ggufInt64, ggufFloat64:
		return 8, true
	}
	return 0, false
}

// value reads one value. Integers are returned as int64, strings as string,
// string arrays as []string; anything else is skipped and returned as nil.
func (g *ggufReader) value(t uint32, depth int) (any, error) {
	if depth > 2 {
		return nil, errors.New("gguf: arrays nested too deeply")
	}
	switch t {
	case ggufString:
		return g.str()
	case ggufArray:
		et, err := g.u32()
		if err != nil {
			return nil, err
		}
		n, err := g.count()
		if err != nil {
			return nil, err
		}
		if n > maxGGUFArray {
			return nil, fmt.Errorf("gguf: array of %d items is too long", n)
		}
		if size, ok := scalarSize(et); ok {
			return nil, g.skip(size * n)
		}
		var out []string
		keep := et == ggufString && n <= 64 // small string arrays only (labels)
		for i := uint64(0); i < n; i++ {
			if !keep && et == ggufString { // e.g. a 150k-token vocabulary: skip without allocating
				size, err := g.count()
				if err != nil {
					return nil, err
				}
				if size > maxGGUFString {
					return nil, fmt.Errorf("gguf: string of %d bytes is too long", size)
				}
				if err := g.skip(size); err != nil {
					return nil, err
				}
				continue
			}
			v, err := g.value(et, depth+1)
			if err != nil {
				return nil, err
			}
			if keep {
				s, _ := v.(string)
				out = append(out, s)
			}
		}
		if keep {
			return out, nil
		}
		return nil, nil
	}
	size, ok := scalarSize(t)
	if !ok {
		return nil, fmt.Errorf("gguf: unknown value type %d", t)
	}
	buf := make([]byte, size)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		return nil, err
	}
	switch t {
	case ggufUint8, ggufBool:
		return int64(buf[0]), nil
	case ggufInt8:
		return int64(int8(buf[0])), nil
	case ggufUint16:
		return int64(binary.LittleEndian.Uint16(buf)), nil
	case ggufInt16:
		return int64(int16(binary.LittleEndian.Uint16(buf))), nil
	case ggufUint32:
		return int64(binary.LittleEndian.Uint32(buf)), nil
	case ggufInt32:
		return int64(int32(binary.LittleEndian.Uint32(buf))), nil
	case ggufUint64, ggufInt64:
		return int64(binary.LittleEndian.Uint64(buf)), nil
	}
	return nil, nil // floats are not needed
}

// ReadGGUF reads the header, metadata and tensor names of a GGUF file.
func ReadGGUF(path string) (*GGUFInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return parseGGUF(f)
}

func parseGGUF(r io.Reader) (*GGUFInfo, error) {
	g := &ggufReader{r: bufio.NewReaderSize(r, 1<<20)}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(g.r, magic); err != nil {
		return nil, fmt.Errorf("gguf: %w", err)
	}
	if string(magic) != "GGUF" {
		return nil, errors.New("not a GGUF file")
	}
	v, err := g.u32()
	if err != nil {
		return nil, err
	}
	if v < 1 || v > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", v)
	}
	g.version = v
	nTensors, err := g.count()
	if err != nil {
		return nil, err
	}
	nKV, err := g.count()
	if err != nil {
		return nil, err
	}
	if nKV > maxGGUFKVs || nTensors > maxGGUFTensor {
		return nil, errors.New("gguf: header counts are implausible")
	}
	info := &GGUFInfo{Version: v, PoolingType: -1, Tensors: map[string]bool{}}
	kv := map[string]any{}
	for i := uint64(0); i < nKV; i++ {
		key, err := g.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: key %d: %w", i, err)
		}
		t, err := g.u32()
		if err != nil {
			return nil, err
		}
		val, err := g.value(t, 0)
		if err != nil {
			return nil, fmt.Errorf("gguf: %s: %w", key, err)
		}
		if val != nil {
			kv[key] = val
		}
	}
	for i := uint64(0); i < nTensors; i++ {
		name, err := g.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %d: %w", i, err)
		}
		dims, err := g.u32()
		if err != nil {
			return nil, err
		}
		if dims > maxGGUFDims {
			return nil, fmt.Errorf("gguf: tensor %s has %d dimensions", name, dims)
		}
		// dims × length, then type (u32) and offset (u64).
		if err := g.skip(uint64(dims)*8 + 4 + 8); err != nil {
			return nil, err
		}
		info.Tensors[name] = true
	}
	info.Architecture, _ = kv["general.architecture"].(string)
	if p, ok := kv[info.Architecture+".pooling_type"].(int64); ok {
		info.PoolingType = p
	}
	_, info.HasRerankTpl = kv["tokenizer.chat_template.rerank"].(string)
	info.Labels, _ = kv[info.Architecture+".classifier.output_labels"].([]string)
	return info, nil
}

// ScoreKind says how a reranker's scores come out of llama-server.
type ScoreKind int

const (
	// ScoreProbability: llama.cpp applies softmax over yes/no (Qwen3 rerankers).
	ScoreProbability ScoreKind = iota
	// ScoreLogit: a raw classifier logit (BERT-style cross-encoders such as bge-reranker).
	ScoreLogit
)

// CheckReranker reports whether llama-server can serve this GGUF as a reranker,
// with a fix-it message when it cannot.
func CheckReranker(info *GGUFInfo) (ScoreKind, error) {
	arch := info.Architecture
	if arch == "" {
		return 0, errors.New("the GGUF has no general.architecture")
	}
	if strings.HasPrefix(arch, "qwen3") {
		var missing []string
		if !info.Tensors["cls.output.weight"] {
			missing = append(missing, "the cls.output.weight tensor")
		}
		if !info.HasRerankTpl {
			missing = append(missing, "the rerank chat template")
		}
		if len(missing) > 0 {
			return 0, fmt.Errorf("this %s GGUF is not a usable reranker: it lacks %s. It was probably converted as a "+
				"plain chat model. Re-convert the Hugging Face model with a current llama.cpp convert_hf_to_gguf.py "+
				"(it detects Qwen3-Reranker), or pull a GGUF that was converted that way",
				arch, strings.Join(missing, ", "))
		}
		return ScoreProbability, nil
	}
	if info.PoolingType == poolingRank || info.Tensors["cls.output.weight"] || info.Tensors["cls.weight"] {
		return ScoreLogit, nil
	}
	return 0, fmt.Errorf("this %s GGUF has no reranking head (no rank pooling or cls tensors); it looks like a chat "+
		"or embedding model, not a reranker", arch)
}

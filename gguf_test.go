package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ggufBuilder writes small GGUF v3 files for tests.
type ggufBuilder struct {
	kvs     bytes.Buffer
	nKV     uint64
	tensors []string
}

func (b *ggufBuilder) str(w *bytes.Buffer, s string) {
	_ = binary.Write(w, binary.LittleEndian, uint64(len(s)))
	w.WriteString(s)
}

func (b *ggufBuilder) String(key, val string) *ggufBuilder {
	b.str(&b.kvs, key)
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint32(ggufString))
	b.str(&b.kvs, val)
	b.nKV++
	return b
}

func (b *ggufBuilder) U32(key string, val uint32) *ggufBuilder {
	b.str(&b.kvs, key)
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint32(ggufUint32))
	_ = binary.Write(&b.kvs, binary.LittleEndian, val)
	b.nKV++
	return b
}

func (b *ggufBuilder) Strings(key string, vals []string) *ggufBuilder {
	b.str(&b.kvs, key)
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint32(ggufArray))
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint32(ggufString))
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint64(len(vals)))
	for _, v := range vals {
		b.str(&b.kvs, v)
	}
	b.nKV++
	return b
}

func (b *ggufBuilder) F32s(key string, n int) *ggufBuilder {
	b.str(&b.kvs, key)
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint32(ggufArray))
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint32(ggufFloat32))
	_ = binary.Write(&b.kvs, binary.LittleEndian, uint64(n))
	b.kvs.Write(make([]byte, 4*n))
	b.nKV++
	return b
}

func (b *ggufBuilder) Tensor(name string) *ggufBuilder {
	b.tensors = append(b.tensors, name)
	return b
}

func (b *ggufBuilder) Bytes() []byte {
	var out bytes.Buffer
	out.WriteString("GGUF")
	_ = binary.Write(&out, binary.LittleEndian, uint32(3))
	_ = binary.Write(&out, binary.LittleEndian, uint64(len(b.tensors)))
	_ = binary.Write(&out, binary.LittleEndian, b.nKV)
	out.Write(b.kvs.Bytes())
	for _, t := range b.tensors {
		b.str(&out, t)
		_ = binary.Write(&out, binary.LittleEndian, uint32(2)) // dims
		_ = binary.Write(&out, binary.LittleEndian, uint64(8))
		_ = binary.Write(&out, binary.LittleEndian, uint64(2))
		_ = binary.Write(&out, binary.LittleEndian, uint32(0)) // type
		_ = binary.Write(&out, binary.LittleEndian, uint64(0)) // offset
	}
	return out.Bytes()
}

func (b *ggufBuilder) File(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func qwenReranker() *ggufBuilder {
	vocab := make([]string, 5000)
	for i := range vocab {
		vocab[i] = "tok"
	}
	return (&ggufBuilder{}).
		String("general.architecture", "qwen3").
		U32("qwen3.pooling_type", poolingRank).
		Strings("qwen3.classifier.output_labels", []string{"yes", "no"}).
		Strings("tokenizer.ggml.tokens", vocab).
		F32s("tokenizer.ggml.scores", 5000).
		String("tokenizer.chat_template.rerank", "<|im_start|>system ... {query} {document}").
		Tensor("token_embd.weight").Tensor("cls.output.weight")
}

func TestQwen3Reranker(t *testing.T) {
	info, err := parseGGUF(bytes.NewReader(qwenReranker().Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if info.Architecture != "qwen3" || info.PoolingType != poolingRank || !info.HasRerankTpl {
		t.Fatalf("unexpected info: %+v", info)
	}
	if strings.Join(info.Labels, ",") != "yes,no" {
		t.Fatalf("labels: %v", info.Labels)
	}
	kind, err := CheckReranker(info)
	if err != nil || kind != ScoreProbability {
		t.Fatalf("kind %v err %v", kind, err)
	}
}

func TestBrokenQwen3Reranker(t *testing.T) {
	// Converted as a plain chat model: no head, no rerank template, no rank pooling.
	b := (&ggufBuilder{}).String("general.architecture", "qwen3").Tensor("token_embd.weight").Tensor("output.weight")
	info, err := parseGGUF(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	_, err = CheckReranker(info)
	if err == nil || !strings.Contains(err.Error(), "cls.output.weight") || !strings.Contains(err.Error(), "Re-convert") {
		t.Fatalf("want a fix-it error, got %v", err)
	}
}

func TestBertReranker(t *testing.T) {
	b := (&ggufBuilder{}).String("general.architecture", "bert").U32("bert.pooling_type", poolingRank).
		Tensor("cls.weight").Tensor("cls.output.weight")
	info, _ := parseGGUF(bytes.NewReader(b.Bytes()))
	kind, err := CheckReranker(info)
	if err != nil || kind != ScoreLogit {
		t.Fatalf("kind %v err %v", kind, err)
	}
}

func TestChatModelIsNotAReranker(t *testing.T) {
	b := (&ggufBuilder{}).String("general.architecture", "llama").Tensor("output.weight")
	info, _ := parseGGUF(bytes.NewReader(b.Bytes()))
	if _, err := CheckReranker(info); err == nil || !strings.Contains(err.Error(), "not a reranker") {
		t.Fatalf("got %v", err)
	}
}

func TestBadFiles(t *testing.T) {
	good := qwenReranker().Bytes()
	cases := map[string][]byte{
		"not gguf":  []byte("PK\x03\x04 zip"),
		"truncated": good[:len(good)-20],
		"empty":     nil,
	}
	huge := append([]byte("GGUF"), 3, 0, 0, 0)
	huge = binary.LittleEndian.AppendUint64(huge, 1<<40) // tensors
	huge = binary.LittleEndian.AppendUint64(huge, 1)
	cases["implausible counts"] = huge
	longStr := (&ggufBuilder{})
	longStr.str(&longStr.kvs, "k")
	_ = binary.Write(&longStr.kvs, binary.LittleEndian, uint32(ggufString))
	_ = binary.Write(&longStr.kvs, binary.LittleEndian, uint64(1<<40))
	longStr.nKV = 1
	cases["huge string"] = longStr.Bytes()
	for name, data := range cases {
		if _, err := parseGGUF(bytes.NewReader(data)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestReadGGUFFromDisk(t *testing.T) {
	p := qwenReranker().File(t, t.TempDir(), "m.gguf")
	info, err := ReadGGUF(p)
	if err != nil || !info.Tensors["cls.output.weight"] {
		t.Fatalf("info %+v err %v", info, err)
	}
}

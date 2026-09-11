# Tutorial — HTTP server (Stage S4)

A working engine is not yet a product surface. Stage S4 adds a **thin** HTTP
layer so a local client can stream tokens — without turning the repo into a
web framework.

## Protocol vs engine

```text
HTTP request  →  http_server (JSON / SSE)  →  Session.generate
HTTP response ←  on_token / JSON body     ←  token ids / text
```

All inference stays in the existing Session path. The server only:

1. Parses a small JSON body
2. Builds prompt token ids (or uses `prompt_tokens` on mini)
3. Calls `generate` with optional `on_token`
4. Writes SSE chunks or a one-shot JSON completion

## Endpoints

| Method | Path | Role |
| --- | --- | --- |
| GET | `/health` | Liveness + model/backend label |
| GET | `/metrics` | Prometheus-ish counters |
| POST | `/v1/completions` | Prompt or `prompt_tokens` → text/stream |
| POST | `/v1/chat/completions` | Chat messages (tokenizer required) |

Streaming uses `Content-Type: text/event-stream`. Each event is
`data: {"text":"…"}` (or `{"id":N}` on mini), ending with `data: [DONE]`.

## Try it

```bash
zig build stageS4 -Dhip=off
./zig-out/bin/zynfer serve --mini --smoke
```

## Example queries

### Start the server

Use one terminal for the server and another for `curl`.

**Mini** (no tokenizer — completions take `prompt_tokens` only):

```bash
./zig-out/bin/zynfer serve --mini --port 8080
```

**Real artifact** (text prompts + chat; pick a free port if mini is still up):

```bash
./zig-out/bin/zynfer serve models/qwen3-0.6b-int8.zynfer \
  --tokenizer models/Qwen3-0.6B --backend apple --port 8081
```

Mini streams SSE events as `data: {"id":N}`. With a tokenizer, events are
`data: {"text":"…"}`. Both end with `data: [DONE]`. Use `curl -N` so SSE
chunks print as they arrive.

### Health and metrics

```bash
curl -s http://127.0.0.1:8080/health | jq
curl -s http://127.0.0.1:8080/metrics
```

### Completions (mini — `prompt_tokens`)

Stream tokens:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_tokens":[2,3,2,3],"max_tokens":8,"stream":true}'
```

One-shot JSON (no SSE):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_tokens":[1,2,3,4],"max_tokens":6,"stream":false}' | jq
```

Longer prompt / fixed seed:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_tokens":[4,2,3,4,2],"max_tokens":12,"stream":true,"temperature":0,"seed":42}'
```

### Expected errors on mini

Text `prompt` and chat need a tokenizer — on `--mini` they return an error JSON:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello"}'

curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

### Chat (real model)

```bash
curl -N -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in one sentence."}],"max_tokens":32,"stream":true}'
```

Text completions with the same server:

```bash
curl -N -X POST http://127.0.0.1:8081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a haiku about Zig.","max_tokens":64,"stream":true}'
```

## What this stage is not

- Not TLS, auth, or a full OpenAI clone
- Not the S1 FIFO/RR scheduler exposed over the wire
- Not multi-request Metal concurrency
- Not a reason to move kernels into request handlers

Keep the seam: when serving grows, change the protocol module first.

## See also

- [`docs/stages/S4-http-server.md`](../stages/S4-http-server.md)
- [`bench/results/stageS4-dev-laptop.md`](../../bench/results/stageS4-dev-laptop.md)
- Tutorials 23–25 (batching, prefix, speculative) — complementary serving pieces

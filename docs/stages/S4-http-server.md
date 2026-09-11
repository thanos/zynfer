# Stage S4 — HTTP server

**Status: done (protocol layer).** Small Zig HTTP front-end over
`Session.generate`. Protocol translation stays in `http_server.zig`; the
engine is unchanged.

Part of **Phase S** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 24.

## Goal

```bash
zig build stageS4 -Dhip=off
./zig-out/bin/zynfer stageS4
./zig-out/bin/zynfer serve --mini --smoke
./zig-out/bin/zynfer serve --mini --host 127.0.0.1 --port 8080
```

Client examples (health, stream, non-stream, chat, expected mini errors):
[`docs/tutorials/26-http-server.md`](../tutorials/26-http-server.md#example-queries).

## What landed

| Piece | Policy |
| --- | --- |
| Transport | `std.Io.net` listen/accept + `std.http.Server` |
| Endpoints | `GET /health`, `GET /metrics`, `POST /v1/completions`, `POST /v1/chat/completions` |
| Streaming | SSE (`text/event-stream`) via `on_token` |
| Mini | `prompt_tokens` without HF tokenizer |
| Chat | Last user message + chat template (tokenizer required) |
| Concurrency | One request at a time (blocking accept loop) |
| Smoke | `serve --smoke` / unit: health + streamed completions |

## Explicit non-goals

- TLS / auth / API keys
- Full OpenAI schema parity
- Multi-model routing / hot reload
- S1 scheduler over HTTP / concurrent Metal sessions
- S3b draft-model speculation

## Gate

1. Local client streams tokens over HTTP (`serve --smoke` + unit)
2. `/health` + `/metrics` + streamed `/v1/completions`
3. Tutorial teaches protocol vs engine boundary
4. Non-goals documented above

## Commands

```bash
zig build test -Dhip=off
zig build stageS4 -Dhip=off
zig build serve-smoke -Dhip=off
zig build integration -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/runtime/http_server.zig` | Engine wrapper, routes, SSE, smoke |
| `src/main.zig` | `stageS4`, `serve` |
| `docs/tutorials/26-http-server.md` | Walkthrough |
| `bench/results/stageS4-dev-laptop.md` | Ledger |

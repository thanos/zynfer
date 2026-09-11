# Stage S4 — HTTP server (dev laptop)

Protocol layer only: `std.Io.net` + `std.http.Server` over `Session.generate`.
Streaming via SSE. One request at a time.

```text
date:              2026-09-11
host:              MacBook-Pro (Apple Silicon)
OS:                Darwin 25.5.0 (arm64)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageS4 -Dhip=off
  zig build serve-smoke -Dhip=off
  ./zig-out/bin/zynfer serve --mini --smoke --backend cpu
```

## Correctness / gate

| Check | Result |
| --- | --- |
| Unit: `http smoke health + streamed completions on mini` | PASS |
| `serve --mini --smoke` | PASS |
| `stageS4` ledger text | PASS |

## Smoke metrics (mini CPU)

| Field | Value |
| --- | ---: |
| requests | 2 |
| completions | 1 |
| streams | 1 |
| tokens | 4 |

## Policy

1. Keep HTTP translation out of the forward path.
2. Default stream=true for completions; non-stream JSON still supported.
3. Mini uses `prompt_tokens`; chat requires a tokenizer.
4. Blocking single-flight accept loop — concurrency is future work (S1 over HTTP).

## Non-goals (this stage)

- TLS / auth / OpenAI parity / multi-model
- Scheduler multiplexing over HTTP
- Concurrent Metal sessions from HTTP

## See also

- `docs/stages/S4-http-server.md`
- `docs/tutorials/26-http-server.md`

# Tutorial — Prefix reuse (Stage S2)

Chat APIs often send the same system prompt (or tool preamble) on every turn.
Recomputing that shared prefix from scratch wastes prefill work. Stage S2
teaches **prompt caching** on a dense KV — truncate back to the shared length
and continue — before introducing paged blocks.

## In plain English

After you prefill tokens `P = [p0…pₖ₋₁]`, the KV cache holds layer-wise keys
and values for those positions. If the next request is `P ‖ S` (same prefix,
new suffix), you can:

1. Keep the Session’s KV,
2. Set live length back to `|P|` (`truncateTo`),
3. Prefill only `S` (`prefillContinue`).

You do **not** need to re-embed or re-attend over `P`.

That is different from Stage S1 (scheduling many independent Sessions) and
from Stage M3 (packing Metal ops inside one forward).

## Prefix identity and lookup

zynfer’s `PrefixCache` stores **exact** token-id sequences. Lookup returns
the **longest registered** prefix that matches the start of a prompt. Hits
bump an LRU clock; when the table is full, the least-recently-used entry is
evicted.

This is intentionally simple: identity = byte-identical token ids. No fuzzy
match, no hash-tree yet.

## Ownership and fragmentation

| Policy | What S2 does |
| --- | --- |
| Ownership | One Session owns the dense KV buffers |
| Truncate | Only moves the `used` cursor — no free-list |
| “Fragmentation” | Unused capacity past `used` stays allocated (predictable, not compacting) |

**Paged KV** (block allocator, copy-on-write forks, sharing across Sessions)
is the next mental step once this contiguous path is understood. S2 does
not ship it.

## Cold vs warm (what the bench measures)

```text
cold:  each trial  prefill(P ‖ S_i)           // reset every time
warm:  once        prefill(P)
       each trial  truncateTo(|P|); continue(S_i)
```

`savings_ratio = 1 - warm_trial_ns / cold_trial_ns` (one-time warm prefix
prefill is reported separately). Expect large savings when `|P| ≫ |S|`.

## Try it

```bash
zig build stageS2 -Dhip=off
./zig-out/bin/zynfer prefix-bench --mini --prefix-len 8 --suffix-len 2 --trials 4
```

Expect `logits_match: PASS` and `savings_ratio > 0`.

## What is not here

- Paged / block KV — later when contiguous reuse is boring
- Speculative decoding — Stage S3 (tutorial 25)
- HTTP — Stage S4
- Sharing one Metal weight residency across S1 scheduler slots — orthogonal

## See also

- [`docs/stages/S2-prefix-reuse.md`](../stages/S2-prefix-reuse.md)
- [`bench/results/stageS2-dev-laptop.md`](../../bench/results/stageS2-dev-laptop.md)
- Tutorial 23 (request scheduling) for the other half of “serving”

//! Qwen2/Qwen3-compatible byte-level BPE tokenizer.
//!
//! Loads Hugging Face `vocab.json` + `merges.txt` and recognizes Qwen added
//! tokens atomically. Pre-tokenization follows the Qwen2 Split regex using
//! Unicode letter/number checks.

const std = @import("std");

pub const Error = error{
    InvalidTokenizer,
    OutOfMemory,
    UnknownToken,
    InvalidUtf8,
} || std.mem.Allocator.Error;

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    pieces: [][]u8,
    vocab: std.StringHashMapUnmanaged(u32),
    merges: std.StringHashMapUnmanaged(u32),
    id_to_piece: [][]const u8,
    added: std.StringHashMapUnmanaged(u32),
    added_list: [][]u8,
    base_vocab: u32,
    byte_to_cp: [256]u21,
    cp_to_byte: std.AutoHashMapUnmanaged(u21, u8),
    eos_token_id: u32 = 151645,
    endoftext_id: u32 = 151643,
    im_start_id: u32 = 151644,
    im_end_id: u32 = 151645,

    pub fn deinit(self: *Tokenizer) void {
        for (self.pieces) |p| self.allocator.free(p);
        self.allocator.free(self.pieces);
        self.vocab.deinit(self.allocator);

        var mit = self.merges.keyIterator();
        while (mit.next()) |k| self.allocator.free(k.*);
        self.merges.deinit(self.allocator);

        var i: usize = self.base_vocab;
        while (i < self.id_to_piece.len) : (i += 1) {
            if (self.id_to_piece[i].len != 0) self.allocator.free(self.id_to_piece[i]);
        }
        self.allocator.free(self.id_to_piece);

        // added_list owns the same strings as added keys / id_to_piece[added]
        self.allocator.free(self.added_list);
        self.added.deinit(self.allocator);
        self.cp_to_byte.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn loadHfDir(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) Error!Tokenizer {
        var vbuf: [512]u8 = undefined;
        const vocab_path = std.fmt.bufPrint(&vbuf, "{s}/vocab.json", .{dir_path}) catch return error.InvalidTokenizer;
        const vocab_bytes = std.Io.Dir.cwd().readFileAlloc(io, vocab_path, allocator, .limited(64 * 1024 * 1024)) catch return error.InvalidTokenizer;
        defer allocator.free(vocab_bytes);

        var mbuf: [512]u8 = undefined;
        const merges_path = std.fmt.bufPrint(&mbuf, "{s}/merges.txt", .{dir_path}) catch return error.InvalidTokenizer;
        const merges_bytes = std.Io.Dir.cwd().readFileAlloc(io, merges_path, allocator, .limited(64 * 1024 * 1024)) catch return error.InvalidTokenizer;
        defer allocator.free(merges_bytes);

        return loadVocabMerges(allocator, vocab_bytes, merges_bytes);
    }

    pub fn loadVocabMerges(allocator: std.mem.Allocator, vocab_json: []const u8, merges_txt: []const u8) Error!Tokenizer {
        var tok: Tokenizer = .{
            .allocator = allocator,
            .pieces = &.{},
            .vocab = .{},
            .merges = .{},
            .id_to_piece = &.{},
            .added = .{},
            .added_list = &.{},
            .base_vocab = 0,
            .byte_to_cp = undefined,
            .cp_to_byte = .{},
        };
        errdefer tok.deinit();

        try initBytesToUnicode(&tok);

        var max_id: u32 = 0;
        var parsed: std.ArrayList(VocabEntry) = .empty;
        defer {
            for (parsed.items) |e| allocator.free(e.piece);
            parsed.deinit(allocator);
        }
        try parseVocabObject(allocator, vocab_json, &parsed, &max_id);

        tok.base_vocab = max_id + 1;
        tok.pieces = try allocator.alloc([]u8, tok.base_vocab);
        @memset(tok.pieces, &.{});

        for (parsed.items) |e| {
            if (e.id >= tok.base_vocab) return error.InvalidTokenizer;
            if (tok.pieces[e.id].len != 0) return error.InvalidTokenizer;
            tok.pieces[e.id] = e.piece;
        }
        // Prevent defer from freeing moved pieces.
        for (parsed.items) |*e| e.piece = &.{};
        parsed.clearRetainingCapacity();

        for (tok.pieces, 0..) |p, i| {
            if (p.len == 0) tok.pieces[i] = try allocator.dupe(u8, "");
            try tok.vocab.put(allocator, tok.pieces[i], @intCast(i));
        }

        var line_it = std.mem.splitScalar(u8, merges_txt, '\n');
        var rank: u32 = 0;
        while (line_it.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or std.mem.startsWith(u8, line, "#")) continue;
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidTokenizer;
            const key = try std.fmt.allocPrint(allocator, "{s}\x00{s}", .{ line[0..sp], line[sp + 1 ..] });
            errdefer allocator.free(key);
            try tok.merges.put(allocator, key, rank);
            rank += 1;
        }

        const added_defs = qwenAddedTokens();
        const total_ids = @max(tok.base_vocab, highestAddedId(added_defs) + 1);
        tok.id_to_piece = try allocator.alloc([]const u8, total_ids);
        @memset(tok.id_to_piece, "");
        for (tok.pieces, 0..) |p, i| tok.id_to_piece[i] = p;

        var owned_added: std.ArrayList([]u8) = .empty;
        errdefer {
            for (owned_added.items) |s| allocator.free(s);
            owned_added.deinit(allocator);
        }
        for (added_defs) |def| {
            const owned = try allocator.dupe(u8, def.text);
            try tok.added.put(allocator, owned, def.id);
            tok.id_to_piece[def.id] = owned;
            try owned_added.append(allocator, owned);
        }
        const list = try owned_added.toOwnedSlice(allocator);
        std.mem.sort([]u8, list, {}, struct {
            fn less(_: void, a: []u8, b: []u8) bool {
                if (a.len != b.len) return a.len > b.len;
                return std.mem.lessThan(u8, a, b);
            }
        }.less);
        tok.added_list = list;
        return tok;
    }

    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        var pos: usize = 0;
        while (pos < text.len) {
            var matched = false;
            for (self.added_list) |atok| {
                if (pos + atok.len <= text.len and std.mem.eql(u8, text[pos .. pos + atok.len], atok)) {
                    try out.append(allocator, self.added.get(atok).?);
                    pos += atok.len;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;

            var end = text.len;
            var s = pos + 1;
            outer: while (s < text.len) : (s += 1) {
                for (self.added_list) |atok| {
                    if (s + atok.len <= text.len and std.mem.eql(u8, text[s .. s + atok.len], atok)) {
                        end = s;
                        break :outer;
                    }
                }
            }
            try encodePlain(self, allocator, text[pos..end], &out);
            pos = end;
        }
        return try out.toOwnedSlice(allocator);
    }

    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) Error![]u8 {
        var bytes: std.ArrayList(u8) = .empty;
        errdefer bytes.deinit(allocator);
        for (ids) |id| {
            if (id >= self.id_to_piece.len) return error.UnknownToken;
            const piece = self.id_to_piece[id];
            if (id >= self.base_vocab) {
                try bytes.appendSlice(allocator, piece);
                continue;
            }
            var i: usize = 0;
            while (i < piece.len) {
                const n = std.unicode.utf8ByteSequenceLength(piece[i]) catch return error.InvalidUtf8;
                if (i + n > piece.len) return error.InvalidUtf8;
                const cp = std.unicode.utf8Decode(piece[i..][0..n]) catch return error.InvalidUtf8;
                const b = self.cp_to_byte.get(cp) orelse return error.InvalidUtf8;
                try bytes.append(allocator, b);
                i += n;
            }
        }
        return try bytes.toOwnedSlice(allocator);
    }

    pub fn applyChatTemplate(_: *const Tokenizer, allocator: std.mem.Allocator, user_text: []const u8) Error![]u8 {
        return std.fmt.allocPrint(
            allocator,
            "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            .{user_text},
        );
    }
};

fn encodePlain(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) Error!void {
    var pos: usize = 0;
    while (pos < text.len) {
        const next = matchPretoken(text, pos) orelse return error.InvalidUtf8;
        const piece_utf8 = text[pos..next];
        pos = next;

        // UTF-8 bytes → GPT-2 unicode string
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(allocator);
        for (piece_utf8) |byte| {
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(self.byte_to_cp[byte], &buf) catch return error.InvalidUtf8;
            try encoded.appendSlice(allocator, buf[0..n]);
        }
        try bpeAppend(self, allocator, encoded.items, out);
    }
}

fn bpeAppend(self: *const Tokenizer, allocator: std.mem.Allocator, token: []const u8, out: *std.ArrayList(u32)) Error!void {
    if (token.len == 0) return;
    if (self.vocab.get(token)) |id| {
        try out.append(allocator, id);
        return;
    }

    var word: std.ArrayList([]u8) = .empty;
    defer {
        for (word.items) |w| allocator.free(w);
        word.deinit(allocator);
    }
    var i: usize = 0;
    while (i < token.len) {
        const n = std.unicode.utf8ByteSequenceLength(token[i]) catch return error.InvalidUtf8;
        if (i + n > token.len) return error.InvalidUtf8;
        try word.append(allocator, try allocator.dupe(u8, token[i .. i + n]));
        i += n;
    }

    while (word.items.len > 1) {
        var best_rank: u32 = std.math.maxInt(u32);
        var best_i: ?usize = null;
        var c: usize = 0;
        while (c + 1 < word.items.len) : (c += 1) {
            var key_buf: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buf, "{s}\x00{s}", .{ word.items[c], word.items[c + 1] }) catch continue;
            if (self.merges.get(key)) |r| {
                if (r < best_rank) {
                    best_rank = r;
                    best_i = c;
                }
            }
        }
        const bi = best_i orelse break;
        const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{ word.items[bi], word.items[bi + 1] });
        allocator.free(word.items[bi]);
        allocator.free(word.items[bi + 1]);
        word.items[bi] = merged;
        _ = word.orderedRemove(bi + 1);
    }

    for (word.items) |w| {
        const id = self.vocab.get(w) orelse return error.UnknownToken;
        try out.append(allocator, id);
    }
}

/// Qwen2 pretokenizer: return end index of match starting at `pos`, or null.
fn matchPretoken(text: []const u8, pos: usize) ?usize {
    if (pos >= text.len) return null;

    // 1) contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (matchContractions(text, pos)) |e| return e;

    // 2) [^\r\n\p{L}\p{N}]?\p{L}+
    if (matchLetters(text, pos)) |e| return e;

    // 3) \p{N}
    if (matchNumber(text, pos)) |e| return e;

    // 4)  ?[^\s\p{L}\p{N}]+[\r\n]*
    if (matchOther(text, pos)) |e| return e;

    // 5) \s*[\r\n]+
    if (matchNewlineRun(text, pos)) |e| return e;

    // 6/7) whitespace
    if (matchWhitespace(text, pos)) |e| return e;

    // Fallback: single UTF-8 character
    const n = std.unicode.utf8ByteSequenceLength(text[pos]) catch return null;
    if (pos + n > text.len) return null;
    return pos + n;
}

fn matchContractions(text: []const u8, pos: usize) ?usize {
    if (text[pos] != '\'') return null;
    const rests = [_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" };
    for (rests) |r| {
        if (pos + 1 + r.len <= text.len) {
            const slice = text[pos + 1 .. pos + 1 + r.len];
            if (std.ascii.eqlIgnoreCase(slice, r)) {
                // ensure it's exactly that contraction (next char not letter for re/ve/ll/d ambiguity is ok — regex is alternation)
                return pos + 1 + r.len;
            }
        }
    }
    return null;
}

fn matchLetters(text: []const u8, pos: usize) ?usize {
    var i = pos;
    // optional [^\r\n\p{L}\p{N}]
    const cp0 = peekCp(text, i) orelse return null;
    if (!isLetter(cp0.cp) and !isNumber(cp0.cp) and cp0.cp != '\r' and cp0.cp != '\n') {
        // consume one optional non-letter/number
        // but only if followed by letters
        const next = peekCp(text, i + cp0.len) orelse return null;
        if (!isLetter(next.cp)) return null;
        i += cp0.len;
    }
    const first = peekCp(text, i) orelse return null;
    if (!isLetter(first.cp)) return null;
    i += first.len;
    while (i < text.len) {
        const c = peekCp(text, i) orelse break;
        if (!isLetter(c.cp)) break;
        i += c.len;
    }
    return i;
}

fn matchNumber(text: []const u8, pos: usize) ?usize {
    const c = peekCp(text, pos) orelse return null;
    if (!isNumber(c.cp)) return null;
    return pos + c.len;
}

fn matchOther(text: []const u8, pos: usize) ?usize {
    var i = pos;
    if (text[i] == ' ') i += 1;
    const first = peekCp(text, i) orelse return null;
    if (isLetter(first.cp) or isNumber(first.cp) or isSpace(first.cp)) return null;
    i += first.len;
    while (i < text.len) {
        const c = peekCp(text, i) orelse break;
        if (isLetter(c.cp) or isNumber(c.cp) or isSpace(c.cp)) break;
        i += c.len;
    }
    while (i < text.len and (text[i] == '\r' or text[i] == '\n')) : (i += 1) {}
    return i;
}

fn matchNewlineRun(text: []const u8, pos: usize) ?usize {
    var i = pos;
    while (i < text.len and isSpaceCp(text[i]) and text[i] != '\r' and text[i] != '\n') : (i += 1) {}
    if (i >= text.len or (text[i] != '\r' and text[i] != '\n')) return null;
    while (i < text.len and (text[i] == '\r' or text[i] == '\n')) : (i += 1) {}
    return i;
}

fn matchWhitespace(text: []const u8, pos: usize) ?usize {
    if (!isSpaceCp(text[pos])) return null;
    var i = pos + 1;
    while (i < text.len and isSpaceCp(text[i])) : (i += 1) {}
    return i;
}

fn isSpaceCp(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0c;
}

fn isSpace(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0c;
}

const Cp = struct { cp: u21, len: usize };

fn peekCp(text: []const u8, pos: usize) ?Cp {
    if (pos >= text.len) return null;
    const n = std.unicode.utf8ByteSequenceLength(text[pos]) catch return null;
    if (pos + n > text.len) return null;
    const cp = std.unicode.utf8Decode(text[pos..][0..n]) catch return null;
    return .{ .cp = cp, .len = n };
}

fn isLetter(cp: u21) bool {
    if (cp >= 'A' and cp <= 'Z') return true;
    if (cp >= 'a' and cp <= 'z') return true;
    // Latin-1 supplement letters and common extended ranges (practical subset).
    if (cp >= 0x00C0 and cp <= 0x02FF) return true;
    if (cp >= 0x0370 and cp <= 0x03FF) return true; // Greek
    if (cp >= 0x0400 and cp <= 0x052F) return true; // Cyrillic
    if (cp >= 0x0900 and cp <= 0x097F) return true; // Devanagari
    if (cp >= 0x4E00 and cp <= 0x9FFF) return true; // CJK
    if (cp >= 0x3400 and cp <= 0x4DBF) return true;
    if (cp >= 0xF900 and cp <= 0xFAFF) return true;
    if (cp >= 0x3040 and cp <= 0x30FF) return true; // Hiragana/Katakana
    if (cp >= 0xAC00 and cp <= 0xD7AF) return true; // Hangul
    if (cp >= 0xFF21 and cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 and cp <= 0xFF5A) return true;
    return false;
}

fn isNumber(cp: u21) bool {
    if (cp >= '0' and cp <= '9') return true;
    if (cp >= 0xFF10 and cp <= 0xFF19) return true; // fullwidth
    return false;
}

fn initBytesToUnicode(self: *Tokenizer) Error!void {
    var n: u21 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const ok = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255);
        if (ok) {
            self.byte_to_cp[b] = @intCast(b);
        } else {
            self.byte_to_cp[b] = 256 + n;
            n += 1;
        }
    }
    try self.cp_to_byte.ensureTotalCapacity(self.allocator, 256);
    b = 0;
    while (b < 256) : (b += 1) {
        try self.cp_to_byte.put(self.allocator, self.byte_to_cp[b], @intCast(b));
    }
}

const AddedDef = struct { id: u32, text: []const u8 };

fn qwenAddedTokens() []const AddedDef {
    return &[_]AddedDef{
        .{ .id = 151643, .text = "<|endoftext|>" },
        .{ .id = 151644, .text = "<|im_start|>" },
        .{ .id = 151645, .text = "<|im_end|>" },
        .{ .id = 151646, .text = "<|object_ref_start|>" },
        .{ .id = 151647, .text = "<|object_ref_end|>" },
        .{ .id = 151648, .text = "<|box_start|>" },
        .{ .id = 151649, .text = "<|box_end|>" },
        .{ .id = 151650, .text = "<|quad_start|>" },
        .{ .id = 151651, .text = "<|quad_end|>" },
        .{ .id = 151652, .text = "<|vision_start|>" },
        .{ .id = 151653, .text = "<|vision_end|>" },
        .{ .id = 151654, .text = "<|vision_pad|>" },
        .{ .id = 151655, .text = "<|image_pad|>" },
        .{ .id = 151656, .text = "<|video_pad|>" },
        .{ .id = 151657, .text = "<tool_call>" },
        .{ .id = 151658, .text = "</tool_call>" },
        .{ .id = 151659, .text = "<|fim_prefix|>" },
        .{ .id = 151660, .text = "<|fim_middle|>" },
        .{ .id = 151661, .text = "<|fim_suffix|>" },
        .{ .id = 151662, .text = "<|fim_pad|>" },
        .{ .id = 151663, .text = "<|repo_name|>" },
        .{ .id = 151664, .text = "<|file_sep|>" },
        .{ .id = 151665, .text = "<tool_response>" },
        .{ .id = 151666, .text = "</tool_response>" },
        .{ .id = 151667, .text = "<think>" },
        .{ .id = 151668, .text = "</think>" },
    };
}

fn highestAddedId(defs: []const AddedDef) u32 {
    var m: u32 = 0;
    for (defs) |d| m = @max(m, d.id);
    return m;
}

const VocabEntry = struct { id: u32, piece: []u8 };

fn parseVocabObject(
    allocator: std.mem.Allocator,
    src: []const u8,
    out: *std.ArrayList(VocabEntry),
    max_id: *u32,
) Error!void {
    var i: usize = 0;
    while (i < src.len and (src[i] == ' ' or src[i] == '\n' or src[i] == '\r' or src[i] == '\t')) : (i += 1) {}
    if (i >= src.len or src[i] != '{') return error.InvalidTokenizer;
    i += 1;
    while (i < src.len) {
        while (i < src.len and (src[i] == ' ' or src[i] == '\n' or src[i] == '\r' or src[i] == '\t' or src[i] == ',')) : (i += 1) {}
        if (i < src.len and src[i] == '}') return;
        if (i >= src.len or src[i] != '"') return error.InvalidTokenizer;
        i += 1;
        const key = try parseJsonString(allocator, src, &i);
        errdefer allocator.free(key);
        while (i < src.len and (src[i] == ' ' or src[i] == '\t')) : (i += 1) {}
        if (i >= src.len or src[i] != ':') return error.InvalidTokenizer;
        i += 1;
        while (i < src.len and (src[i] == ' ' or src[i] == '\t')) : (i += 1) {}
        const id_start = i;
        while (i < src.len and src[i] >= '0' and src[i] <= '9') : (i += 1) {}
        if (id_start == i) return error.InvalidTokenizer;
        const id = std.fmt.parseInt(u32, src[id_start..i], 10) catch return error.InvalidTokenizer;
        max_id.* = @max(max_id.*, id);
        try out.append(allocator, .{ .id = id, .piece = key });
    }
    return error.InvalidTokenizer;
}

fn parseJsonString(allocator: std.mem.Allocator, src: []const u8, i: *usize) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    while (i.* < src.len) {
        const c = src[i.*];
        i.* += 1;
        if (c == '"') return try out.toOwnedSlice(allocator);
        if (c == '\\') {
            if (i.* >= src.len) return error.InvalidTokenizer;
            const e = src[i.*];
            i.* += 1;
            switch (e) {
                '"', '\\', '/' => try out.append(allocator, e),
                'b' => try out.append(allocator, 0x08),
                'f' => try out.append(allocator, 0x0c),
                'n' => try out.append(allocator, '\n'),
                'r' => try out.append(allocator, '\r'),
                't' => try out.append(allocator, '\t'),
                'u' => {
                    if (i.* + 4 > src.len) return error.InvalidTokenizer;
                    const hex = src[i.* .. i.* + 4];
                    i.* += 4;
                    const cp = std.fmt.parseInt(u16, hex, 16) catch return error.InvalidTokenizer;
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &buf) catch return error.InvalidTokenizer;
                    try out.appendSlice(allocator, buf[0..n]);
                },
                else => return error.InvalidTokenizer,
            }
        } else {
            try out.append(allocator, c);
        }
    }
    return error.InvalidTokenizer;
}

test "mini bpe round-trip" {
    // Tiny vocab: bytes for 'a','b','ab' after GPT-2 unicode (same as ascii for these).
    const vocab =
        \\{"a":0,"b":1,"ab":2,".":3}
    ;
    const merges =
        \\#version: 0.2
        \\a b
    ;
    var tok = try Tokenizer.loadVocabMerges(std.testing.allocator, vocab, merges);
    defer tok.deinit();
    const ids = try tok.encode(std.testing.allocator, "ab");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 1), ids.len);
    try std.testing.expectEqual(@as(u32, 2), ids[0]);
    const text = try tok.decode(std.testing.allocator, ids);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("ab", text);
}

test "real qwen encode Explain gravity simply." {
    const path = "models/Qwen3-0.6B";
    std.Io.Dir.cwd().access(std.testing.io, path, .{}) catch return error.SkipZigTest;
    var tok = try Tokenizer.loadHfDir(std.testing.allocator, std.testing.io, path);
    defer tok.deinit();
    const ids = try tok.encode(std.testing.allocator, "Explain gravity simply.");
    defer std.testing.allocator.free(ids);
    const expected = [_]u32{ 840, 20772, 23249, 4936, 13 };
    try std.testing.expectEqualSlices(u32, &expected, ids);
    const text = try tok.decode(std.testing.allocator, ids);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("Explain gravity simply.", text);
}

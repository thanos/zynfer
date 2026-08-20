#!/usr/bin/env bash
# Build kcov from source. Ubuntu 24.04 (noble) dropped the apt package:
# https://answers.launchpad.net/ubuntu/+source/kcov/+question/818841
set -euo pipefail

KCOV_VERSION="${KCOV_VERSION:-43}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PREFIX="${KCOV_PREFIX:-$ROOT/.kcov-prefix}"
BIN="$PREFIX/bin/kcov"

if [[ -x "$BIN" ]] && "$BIN" --version >/dev/null 2>&1; then
    echo "kcov already at $BIN ($("$BIN" --version | head -n1))"
    if [[ -n "${GITHUB_PATH:-}" ]]; then
        echo "$PREFIX/bin" >> "$GITHUB_PATH"
    fi
    export PATH="$PREFIX/bin:$PATH"
    exit 0
fi

if command -v kcov >/dev/null 2>&1; then
    echo "using kcov from PATH: $(command -v kcov)"
    exit 0
fi

echo "building kcov v${KCOV_VERSION} into $PREFIX"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    binutils-dev \
    cmake \
    g++ \
    libcurl4-openssl-dev \
    libdw-dev \
    libelf-dev \
    libiberty-dev \
    ninja-build \
    pkg-config \
    python3 \
    zlib1g-dev

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl -sL "https://github.com/SimonKagstrom/kcov/archive/refs/tags/v${KCOV_VERSION}.tar.gz" \
    | tar xz -C "$work"
src="$work/kcov-${KCOV_VERSION}"
cmake -S "$src" -B "$src/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$src/build"
cmake --install "$src/build"

if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$PREFIX/bin" >> "$GITHUB_PATH"
fi
export PATH="$PREFIX/bin:$PATH"
"$BIN" --version

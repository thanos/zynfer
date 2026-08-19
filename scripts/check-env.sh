#!/usr/bin/env bash
# Print a concise zynfer development-environment report.
# Works on macOS (no GPU) and on the Linux ROCm host.
set -u

have() { command -v "$1" >/dev/null 2>&1; }

print_kv() {
    printf "  %-18s %s\n" "$1" "$2"
}

first_existing() {
    local path
    for path in "$@"; do
        if [[ -f "$path" ]]; then
            tr -d '\n' <"$path"
            echo
            return 0
        fi
    done
    echo "not found"
}

echo "zynfer environment report (shell)"
echo "================================="
echo
echo "host"
print_kv "hostname:" "$(hostname 2>/dev/null || echo unknown)"
print_kv "uname:" "$(uname -a 2>/dev/null || echo unknown)"
print_kv "kernel:" "$(uname -r 2>/dev/null || echo unknown)"
print_kv "machine:" "$(uname -m 2>/dev/null || echo unknown)"
echo
echo "toolchain"
if have zig; then
    print_kv "Zig version:" "$(zig version)"
    print_kv "Zig path:" "$(command -v zig)"
else
    print_kv "Zig version:" "not found"
    print_kv "Zig path:" "not found"
fi
if have clang; then
    print_kv "clang path:" "$(command -v clang)"
    print_kv "clang version:" "$(clang --version 2>/dev/null | head -n1)"
else
    print_kv "clang path:" "not found"
    print_kv "clang version:" "not found"
fi
echo
echo "ROCm / HIP"
print_kv "HIP_PATH:" "${HIP_PATH:-unset}"
print_kv "ROCM_PATH:" "${ROCM_PATH:-unset}"
print_kv "ROCm prefix:" "$(if [[ -d /opt/rocm ]]; then echo /opt/rocm; else echo not found; fi)"
print_kv "ROCm version:" "$(first_existing /opt/rocm/.info/version /opt/rocm/.info/version-dev)"
if have hipcc; then
    print_kv "hipcc path:" "$(command -v hipcc)"
    print_kv "hipcc version:" "$(hipcc --version 2>/dev/null | tr '\n' ' ')"
else
    print_kv "hipcc path:" "not found"
    print_kv "hipcc version:" "not found"
fi
if have rocminfo; then
    print_kv "rocminfo path:" "$(command -v rocminfo)"
else
    print_kv "rocminfo path:" "not found"
fi
echo
echo "driver"
if [[ -d /sys/module/amdgpu ]]; then
    print_kv "amdgpu module:" "loaded"
    print_kv "amdgpu version:" "$(first_existing /sys/module/amdgpu/version)"
    print_kv "init state:" "$(first_existing /sys/module/amdgpu/initstate)"
else
    print_kv "amdgpu module:" "not present"
    print_kv "driver state:" "expected only on the Linux GPU host"
fi
echo
echo "GPU"
if have rocminfo; then
    rocminfo 2>/dev/null | awk '
        /Marketing Name:/ { name=$0; sub(/^.*Marketing Name:[[:space:]]*/, "", name); if (!seen++) print "  marketing name:   " name }
        /Name:.*gfx/ && !arch { arch=$0; sub(/^.*Name:[[:space:]]*/, "", arch); print "  ISA / arch:       " arch }
        /Device Type:.*GPU/ { gpu=1 }
    '
else
    print_kv "rocminfo:" "not available"
fi
if have lspci; then
    echo
    echo "lspci (AMD/display)"
    lspci 2>/dev/null | grep -i -E 'vga|display|amd' | sed 's/^/  /' || true
fi
echo
echo "Apple / Metal"
if [[ "$(uname -s)" == "Darwin" ]]; then
    print_kv "Metal.framework:" "$(if [[ -d /System/Library/Frameworks/Metal.framework ]]; then echo present; else echo missing; fi)"
    if xcrun -sdk macosx metal -v >/dev/null 2>&1; then
        print_kv "metal compiler:" "$(xcrun -sdk macosx metal -v 2>&1 | head -n1)"
    else
        print_kv "metal compiler:" "not available (xcodebuild -downloadComponent MetalToolchain)"
    fi
else
    print_kv "Metal.framework:" "n/a (not macOS)"
fi
echo
echo "expected target: AMD Radeon AI PRO R9700 / gfx1201"
echo "Apple development path: Metal baseline ops on the local M-series GPU"
echo "this script does not require Zig; use 'zig build run' for the Zig report."

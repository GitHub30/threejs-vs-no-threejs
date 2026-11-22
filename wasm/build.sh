#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building engine_simd.wasm (Rust #![no_std], zero crates, wasm32 + simd128)..."
rustc \
  --edition=2021 \
  --target wasm32-unknown-unknown \
  --crate-type cdylib \
  -C opt-level=3 \
  -C lto=fat \
  -C codegen-units=1 \
  -C panic=abort \
  -C strip=symbols \
  -C target-feature=+simd128 \
  "$DIR/src/lib.rs" \
  -o "$DIR/engine_simd.wasm"

echo "Building engine_scalar.wasm (Rust #![no_std], zero crates, wasm32 scalar fallback)..."
rustc \
  --edition=2021 \
  --target wasm32-unknown-unknown \
  --crate-type cdylib \
  -C opt-level=3 \
  -C lto=fat \
  -C codegen-units=1 \
  -C panic=abort \
  -C strip=symbols \
  "$DIR/src/lib.rs" \
  -o "$DIR/engine_scalar.wasm"

ls -lh "$DIR/engine_simd.wasm" "$DIR/engine_scalar.wasm"

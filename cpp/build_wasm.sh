#!/bin/bash
# Rebuild the WASM engine.
# Run from any directory — it finds the project root automatically.

cd "$(dirname "$0")/.."
em++ cpp/engine_wasm.cpp -O3 -std=c++17 -fexceptions \
     -s MODULARIZE=1 -s EXPORT_NAME="HexEngine" \
     -s ALLOW_MEMORY_GROWTH=1 --bind \
     -Icpp -DNDEBUG -o cpp/hex_engine.js

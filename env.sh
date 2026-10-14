#!/bin/bash
export UV_CACHE_DIR="$(pwd)/.uv-cache"
export HF_HOME="$(pwd)/.hf-cache"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

#!/bin/bash
# DeepSeek-V4.1-Flash on 4×CMP 170HX (sm80, PCIe Gen2 x4, no P2P):
# PP4×TP1 + shadow KV + DSpark n=5 + EXL3 2bpw (Pollard-calibrated) + 512k ctx.
# Image = official vllm-openai:deepseekv41-flash-0909 + dsv41reap overlay (54 files)
# + patches/ (4 patch scripts) + vllm-exl3 plugin + exllamav3 v1.5.0 ext. See README.
set -u

NAME=${NAME:-dsv41-pp4}
PORT=${PORT:-8095}
MODEL=${MODEL:-/models/v41-pollard-calibrated}   # EXL3 2bpw pack (334GB)
GRAFT=${GRAFT:-$HOME/v41-graft}                   # dsv41reap clone + sitecustomize
IMAGE=${IMAGE:-pp4-exl3:v14}

docker rm -f $NAME 2>/dev/null; sleep 3

# guards: GPUs must be free; engram pin needs ~190GiB host RAM
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "$USED" -gt 2000 ] && { echo "GPU busy: $USED MiB"; exit 1; }
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
[ "$AVAIL" -lt 200 ] && { echo "RAM low: ${AVAIL}GB (need 200+)"; exit 1; }

mkdir -p $HOME/tilelang-cache   # persistent TileLang JIT cache: 9min → 7.5min startup

docker run -d --name $NAME \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/models:/models:ro \
  -v $GRAFT/dsv41reap/overlay:/overlay:ro \
  -v $GRAFT/v41_sitecustom.py:/usr/lib/python3.12/sitecustomize.py:ro \
  -v $HOME/tilelang-cache:/root/.tilelang \
  -p $PORT:$PORT \
  -e VLLM_PLUGINS=vllm_exl3 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_PP_LAYER_PARTITION=10,10,11,9 \
  -e PYTHONPATH=/overlay -e TORCH_CUDA_ARCH_LIST="8.0" \
  --entrypoint vllm \
  $IMAGE serve $MODEL \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 --port $PORT \
  --max-model-len 524288 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload": true}' \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,12,16,20,24,32]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41

# NOTES (hard-won, do not "fix" without measuring):
# - VLLM_USE_V2_MODEL_RUNNER=0 is REQUIRED: the dspark port + engram piecewise
#   static-meta live on the V1 runner. V2 raises "dspark with PP not supported".
# - use_local_argmax_reduction is NOT supported by the overlay's DSparkProposer.
# - ngram speculation on PP4 is a disaster (7.6 t/s). DSpark conf=0 (verify all).
# - DSV41_DSPARK_CONF=0.7 truncation = -30% on this all-GPU bandwidth-bound arch
#   (their CPU-expert setup benefits; we don't).
# - partition 10,10,11,9: rank3 = 9 layers + drafter, maximizes the KV pool min.
echo "started $NAME @ :$PORT"

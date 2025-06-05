#!/bin/bash

# clear up old run files if they already existed
rm -f ./perf/torch_cpu.prof
rm -r ./perf/torch_gpu.prof
rm -f ./perf/jax_cpu.prof
rm -r ./perf/jax_gpu.prof
rm -f ./perf/jax_cpu_mem.prof
rm -r ./perf/jax_gpu_mem.prof

# export CUDA_VISIBLE_DEVICES="0"
# python -m perf.run \
# --backend_name "torch" \
# --profiler_name "cProfile" \
# --out_path "./perf/torch_gpu.prof"

# python -m perf.run \
# --backend_name "jax" \
# --profiler_name "cProfile" \
# --out_path "./perf/jax_gpu.prof" \

nsys launch python -m perf.run \
--backend_name "jax" \
--profiler_name "nsys" \
--out_path "./perf/jax_gpu.prof" \

# python -m perf.run \
# --backend_name "torch" \
# --profiler_name "torch" \
# --out_path "./perf/torch_gpu.prof" \


export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
# python -m perf.run \
# --backend_name "torch" \
# --profiler_name "cProfile" \
# --out_path "./perf/torch_cpu.prof"

# python -m perf.run \
# --backend_name "jax" \
# --profiler_name "cProfile" \
# --out_path "./perf/jax_cpu.prof"

# python -m perf.run \
# --backend_name "jax" \
# --profiler_name "memray" \
# --out_path "./perf/jax_cpu_mem.prof"

# python -m perf.run \
# --backend_name "torch" \
# --profiler_name "memray" \
# --out_path "./perf/torch_cpu_mem.prof"
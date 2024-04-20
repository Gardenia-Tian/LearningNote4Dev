#!/bin/bash 


# Time
metrics="smsp__cycles_elapsed.avg,smsp__cycles_elapsed.avg.per_second,"

# DP
metrics="${metrics}smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics="${metrics}smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics="${metrics}smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics="${metrics}smsp__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics="${metrics}dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum"

export CUDA_VISIBLE_DEVICES=1
# nvcc profile.cu -o profile && \
ncu --metrics $metrics --csv --print-fp ./exec/share_tile 40960 20480 40960 > ./output/$1.csv
# python postprocess.py
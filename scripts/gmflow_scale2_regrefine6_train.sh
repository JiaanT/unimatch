#!/usr/bin/env bash

# GMFlow with hierarchical matching refinement (1/8 + 1/4 features)
# with additional 6 local regression refinements

# number of gpus for training, please set according to your hardware
# can be trained on 8x 32G V100 or 8x 40GB A100 gpus
NUM_GPUS=4
NUM_NODES=2
NODE_RANK=0
MASTER_ADDR="10.71.106.252"


# # chairs, resume from scale2 model
# CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale2-regrefine6 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume pretrained/gmflow-scale2-chairs-020cc9be.pth \
# --no_resume_optimizer \
# --stage chairs \
# --batch_size 16 \
# --val_dataset chairs sintel kitti \
# --lr 4e-4 \
# --image_size 384 512 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --with_speed_metric \
# --val_freq 10000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# # things
# CHECKPOINT_DIR=checkpoints_flow/things-gmflow-scale2-regrefine6 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints_flow/chairs-gmflow-scale2-regrefine6/step_100000.pth \
# --stage things \
# --batch_size 8 \
# --val_dataset things sintel kitti \
# --lr 2e-4 \
# --image_size 384 768 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --with_speed_metric \
# --val_freq 40000 \
# --save_ckpt_freq 50000 \
# --num_steps 800000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# scannet, resume from things model
CHECKPOINT_DIR=checkpoints_flow/scannet-gmflow-scale2-regrefine6 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=9989 main_flow.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmflow-scale2-things-36579974.pth \
--stage scannet \
--batch_size 4 \
--val_dataset sintel kitti \
--lr 2e-4 \
--image_size 480 640 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 \
--with_speed_metric \
--val_freq 5000 \
--save_ckpt_freq 5000 \
--num_steps 200000 \
--scannet_scenes "all" \
--debug \
--count_time \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

# # sintel, resume from things model
# CHECKPOINT_DIR=checkpoints_flow/sintel-gmflow-scale2-regrefine6 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints_flow/things-gmflow-scale2-regrefine6/step_800000.pth \
# --stage sintel \
# --batch_size 8 \
# --val_dataset sintel kitti \
# --lr 2e-4 \
# --image_size 320 896 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --with_speed_metric \
# --val_freq 20000 \
# --save_ckpt_freq 20000 \
# --num_steps 200000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# # sintel finetune, resume from sintel model, this is our final model for sintel benchmark submission
# CHECKPOINT_DIR=checkpoints_flow/sintel-gmflow-scale2-regrefine6-ft && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints_flow/sintel-gmflow-scale2-regrefine6/step_200000.pth \
# --stage sintel_ft \
# --batch_size 8 \
# --val_dataset sintel \
# --lr 1e-4 \
# --image_size 416 1024 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --with_speed_metric \
# --val_freq 1000 \
# --save_ckpt_freq 1000 \
# --num_steps 5000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# # vkitti2, resume from things model
# CHECKPOINT_DIR=checkpoints_flow/vkitti2-gmflow-scale2-regrefine6 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints_flow/things-gmflow-scale2-regrefine6/step_800000.pth \
# --stage vkitti2 \
# --batch_size 16 \
# --val_dataset kitti \
# --lr 2e-4 \
# --image_size 320 832 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --with_speed_metric \
# --val_freq 10000 \
# --save_ckpt_freq 10000 \
# --num_steps 40000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


# # kitti, resume from vkitti2 model, this is our final model for kitti benchmark submission
# CHECKPOINT_DIR=checkpoints_flow/kitti-gmflow-scale2-regrefine6 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --resume checkpoints_flow/vkitti2-gmflow-scale2-regrefine6/step_040000.pth \
# --stage kitti_mix \
# --batch_size 8 \
# --val_dataset kitti \
# --lr 2e-4 \
# --image_size 352 1216 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --with_speed_metric \
# --val_freq 5000 \
# --save_ckpt_freq 10000 \
# --num_steps 30000 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log



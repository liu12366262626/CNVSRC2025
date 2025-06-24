# 运行脚本（示例）
# 假设您有 2 个 GPU
CUDA_DEVICES="3,4,5"
IFS=',' read -ra ADDR <<< "$CUDA_DEVICES"
DEVICE_COUNT=${#ADDR[@]}

current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")
ROOT_PATH='/home/liuzehua/task/VTS/LipVoicer_revise'

# 根据DEBUG_MODE选择输出路径
DEBUG_MODE=false
if [ "$DEBUG_MODE" = false ]; then
    SAVE_PATH=${ROOT_PATH}/main_log/${current_date}/${current_time}/model-v2
else
    SAVE_PATH=${ROOT_PATH}/main_log/temp
fi

mkdir -p ${SAVE_PATH}
export DEBUG_MODE
export HYDRA_FULL_ERROR=0
export NCCL_P2P_DISABLE=1

cd ..

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=29505 --nproc_per_node=$DEVICE_COUNT --nnodes=1 --node_rank=0 -m model_v2.main \
    input.num_gpus=$DEVICE_COUNT save.save_path=${SAVE_PATH} \
    --config-path ${ROOT_PATH}/exp/model_v2/config \
    --config-name train 
    # 2>&1 | tee ${SAVE_PATH}/train_output.log
#!/usr/bin/env bash



CONFIG=$1
GPUS=$2
GPUS_PER_NODE=$3

CONFIG_NAME=$(basename $1 .py) # name of script
OLD_IFS="$IFS"
IFS="_"
arr=($CONFIG_NAME) 
IFS="$OLD_IFS"

# params
# export MASTER_PORT=`expr substr ${arr[0]} 2 3 + 17000`
export MASTER_PORT=$((12000 + $RANDOM % 20000))

OLD_IFS="$IFS"
IFS="/"
arr_2=($CONFIG)
IFS="$OLD_IFS"

JOB_NAME=${arr[0]}_train_$(date +"%Y%m%d_%H%M%S")
WORK_DIR=.work_dirs/${arr_2[${#arr_2[*]}-4]}/${arr_2[${#arr_2[*]}-3]}/${arr_2[${#arr_2[*]}-2]}/$CONFIG_NAME


if [ ! -d $WORK_DIR  ]
then
  mkdir -p $WORK_DIR
fi

# Default arguments
GPUS=${GPUS:-8} # GPUS数，也是总任务数ntask
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS} # 每个结点（机器）上的GPUS数量，多机训练时小于GPUS
CPUS_PER_TASK=${CPUS_PER_TASK:-8} # 太大的话比如16在开8卡的时候可能cpu数量不够

echo ============ START_TIME: $(date +"%Y-%m-%d_%H:%M:%S") 
echo ============ JOB_NAME: $JOB_NAME
echo ============ CONFIG: $CONFIG_NAME
echo ============ GPUS: $GPUS GPUS_PER_NODE: $GPUS_PER_NODE
echo ============ CPUS_PER_TASK: $CPUS_PER_TASK
# export NCCL_DEBUG=info

echo "Training model from scratch !!!"
srun -p video5 --job-name $JOB_NAME \
--ntasks=$GPUS --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --quotatype=auto --kill-on-bad-exit=1 \
python -u tools/train.py $CONFIG --seed 95  \
--launcher="slurm" --cfg-options dist_params.port=$MASTER_PORT \
--work-dir=$WORK_DIR   
sleep 5s


echo ======== END_TIME: $(date +"%Y-%m-%d_%H:%M:%S") 
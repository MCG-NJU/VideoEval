#!/usr/bin/env bash


CONFIG=$1
CONFIG_NAME=$(basename $1 .py) # name of script
OLD_IFS="$IFS"
IFS="_"
arr=($CONFIG_NAME) 
IFS="$OLD_IFS"



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


# export NCCL_DEBUG=info

echo "Training model from scratch !!!"
python -u tools/train.py $CONFIG --seed 95 --work-dir=$WORK_DIR   
sleep 5s


echo ======== END_TIME: $(date +"%Y-%m-%d_%H:%M:%S") 
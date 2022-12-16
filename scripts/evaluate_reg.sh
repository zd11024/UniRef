if [ -z $dataset ]
then 
    echo "Missing the dataset!!!"
    exit
fi

if [ -z $checkpoint ]
then 
    echo "Missing the checkpoint"
    exit
fi

NPROC_PER_NODE=4
MASTER_PORT=1234

python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE  --nnodes=1 --master_port ${MASTER_PORT} --use_env reg.py \
    --config configs/uniref_finetune.yaml \
    --evaluate --dataset $dataset --checkpoint $checkpoint \
    --output_file results/reg_$dataset.json
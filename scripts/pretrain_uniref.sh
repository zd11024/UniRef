if [ -z $output_dir ]
then 
    echo "Missing the output diretory!!!"
    exit
fi

NPROC_PER_NODE=4
MASTER_PORT=1234

python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE  --nnodes=1 --master_port ${MASTER_PORT} --use_env uniref_pretrain.py \
    --config configs/uniref_pretrain.yaml \
    --output_dir $output_dir

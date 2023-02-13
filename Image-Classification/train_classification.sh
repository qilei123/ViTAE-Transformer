data_dir=data_fd_v4
exp_folder=outputs
port=18989
python -m torch.distributed.launch --nproc_per_node=2 --master_port $port main.py $data_dir --model ViTAEv2_B --epochs 100 \
    -b 64 --lr 1e-4 --weight-decay .003 --amp --output $data_dir/$exp_folder/cls4 --pretrained --num-classes 4 --NBI \
    --vflip 0.5 #--mixup 0

python -m torch.distributed.launch --nproc_per_node=2 --master_port $port main.py $data_dir --model ViTAEv2_B --epochs 100 \
    -b 64 --lr 1e-4 --weight-decay .003 --amp --output $data_dir/$exp_folder/cls3 --pretrained --num-classes 3 --NBI \
    --vflip 0.5 #--mixup 0

python -m torch.distributed.launch --nproc_per_node=2 --master_port $port main.py $data_dir --model ViTAEv2_B --epochs 100 \
    -b 64 --lr 1e-4 --weight-decay .003 --amp --output $data_dir/$exp_folder/cls2 --pretrained --num-classes 2 --NBI \
    --vflip 0.5 #--mixup 0


exp_folder=outputs1

python -m torch.distributed.launch --nproc_per_node=2 --master_port $port main.py $data_dir --model ViTAEv2_B --epochs 100 \
    -b 64 --lr 1e-4 --weight-decay .003 --amp --output $data_dir/$exp_folder/cls4 --pretrained --num-classes 4 --NBI \
    --vflip 0.5 --mixup 0

python -m torch.distributed.launch --nproc_per_node=2 --master_port $port main.py $data_dir --model ViTAEv2_B --epochs 100 \
    -b 64 --lr 1e-4 --weight-decay .003 --amp --output $data_dir/$exp_folder/cls3 --pretrained --num-classes 3 --NBI \
    --vflip 0.5 --mixup 0

python -m torch.distributed.launch --nproc_per_node=2 --master_port $port main.py $data_dir --model ViTAEv2_B --epochs 100 \
    -b 64 --lr 1e-4 --weight-decay .003 --amp --output $data_dir/$exp_folder/cls2 --pretrained --num-classes 2 --NBI \
    --vflip 0.5 --mixup 0


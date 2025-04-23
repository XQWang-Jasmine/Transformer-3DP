python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model resnet18 --batch-size 256 --data-path /dataset/xqwang/code/printing/data/debug --output_dir outputs/debug --epochs 3

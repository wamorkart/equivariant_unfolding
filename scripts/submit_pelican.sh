#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 4
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH -t 10:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --module=gpu,nccl-2.18

export TF_CPP_MIN_LOG_LEVEL=2
module load conda
conda activate zjets

# echo torchrun --nnodes=1 --nproc_per_node=4  train_pelican_classifier.py --datadir=/global/cfs/cdirs/m3246/twamorka/equivariant_unfolding/scripts/test/train_val_test/ --target=is_signal --nobj=60  --nobj-avg=50 --num-epoch=35  --batch-size=30 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005 --save-every=10 --summarize-csv=all --prefix=zjets_split8020_25092024
# torchrun --nnodes=1 --nproc_per_node=4  train_pelican_classifier.py --datadir=/global/cfs/cdirs/m3246/twamorka/equivariant_unfolding/scripts/test/train_val_test/ --target=is_signal --nobj=60  --nobj-avg=50 --num-epoch=35  --batch-size=30 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005 --save-every=10 --summarize-csv=all --prefix=zjets_split8020_25092024


echo torchrun --nnodes=1 --nproc_per_node=4  train_pelican_classifier.py --datadir=/global/cfs/cdirs/m3246/twamorka/equivariant_unfolding/scripts/test/train_val_test/ --target=is_signal  --nobj-avg=50 --num-epoch=35  --batch-size=128 --nobj=80 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005 --save-every=10 --summarize-csv=all --prefix=zjets_split8020
torchrun --nnodes=1 --nproc_per_node=4  train_pelican_classifier.py --datadir=/global/cfs/cdirs/m3246/twamorka/equivariant_unfolding/scripts/test/train_val_test/ --target=is_signal  --nobj-avg=50 --num-epoch=35  --batch-size=128 --nobj=80 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005 --save-every=10 --summarize-csv=all --prefix=zjets_split8020

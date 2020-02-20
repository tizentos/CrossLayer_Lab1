#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J myjob                        # Job name
#SBATCH -o logs/myjob.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e logs/myjob.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                          # Queue (partition) name
#SBATCH -N 1                            # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 10:30:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=jacobhread@gmail.com
#SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A Cross-Layer-ML-HW     # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

#module load intel/17.0.4 python3/3.6.3
#module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.0/lib64
#source $WORK/HWARCH/Lab1B_virtualenv/bin/activate
#mkdir -p $WORK/HWARCH/Lab1B/output
#source activate ml_codesign

# Launch code...

export CUDA_VISIBLE_DEVICES=0
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .001 --batch-size 10  > $WORK/CXLAYER/Lab/data/res001_10

export CUDA_VISIBLE_DEVICES=1
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .01 --batch-size 10  > $WORK/CXLAYER/Lab/data/res01_10

export CUDA_VISIBLE_DEVICES=2
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .05 --batch-size 10  > $WORK/CXLAYER/Lab/data/res05_10

export CUDA_VISIBLE_DEVICES=3
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .1 --batch-size 10  > $WORK/CXLAYER/Lab/data/res1_10

wait

export CUDA_VISIBLE_DEVICES=0
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .001 --batch-size 100  > $WORK/CXLAYER/Lab/data/res001_100

export CUDA_VISIBLE_DEVICES=1
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .01 --batch-size 100  > $WORK/CXLAYER/Lab/data/res01_100

export CUDA_VISIBLE_DEVICES=2
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .05 --batch-size 100  > $WORK/CXLAYER/Lab/data/res05_100

export CUDA_VISIBLE_DEVICES=3
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .1 --batch-size 100  > $WORK/CXLAYER/Lab/data/res1_100

wait

export CUDA_VISIBLE_DEVICES=0
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .001 --batch-size 500  > $WORK/CXLAYER/Lab/data/res001_500

export CUDA_VISIBLE_DEVICES=1
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .01 --batch-size 500  > $WORK/CXLAYER/Lab/data/res01_500

export CUDA_VISIBLE_DEVICES=2
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .05 --batch-size 500  > $WORK/CXLAYER/Lab/data/res05_500

export CUDA_VISIBLE_DEVICES=3
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .1 --batch-size 500  > $WORK/CXLAYER/Lab/data/res1_500

wait

export CUDA_VISIBLE_DEVICES=0
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .001 --batch-size 1000  > $WORK/CXLAYER/Lab/data/res001_1000

export CUDA_VISIBLE_DEVICES=1
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .01 --batch-size 1000  > $WORK/CXLAYER/Lab/data/res01_1000

export CUDA_VISIBLE_DEVICES=2
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .05 --batch-size 1000  > $WORK/CXLAYER/Lab/data/res05_1000

export CUDA_VISIBLE_DEVICES=3
python3 $WORK/CXLAYER/Lab/cnn7c5fc_res.py --lr .1 --batch-size 1000  > $WORK/CXLAYER/Lab/data/res1_1000

wait


# ---------------------------------------------------


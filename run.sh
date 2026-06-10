#!/bin/bash
#SBATCH --job-name=MD
#SBATCH --output=result_1.out
#SBATCH --error=error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --account=bdai-dtai-gh 
#SBATCH --time=30:00:00
##SBATCH --mail-user=choi0652@illinois.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=0

#module purge
#module load nvidia
#module load cuda
module load python/miniforge3_pytorch
unset PYTHONPATH
unset PYTHONHOME
source /work/nvme/bdai/mchoi3/.venv_E3NN/bin/activate
# source /work/nvme/bdai/mchoi3/.venv2/bin/activate
# unset PYTHONPATH
# unset PYTHONHOME

python main.py 
python main_eval.py 
#python main_eval_cryo.py 

# salloc \
#   --job-name=TEST \
#   --nodes=1 \
#   --ntasks-per-node=1 \
#   --partition=ghx4-interactive \
#   --gpus-per-node=1 \
#   --gpus-per-task=1 \
#   --account=bcol-dtai-gh \
#   --time=1:00:00 \
#   --mem=0

# srun --pty /bin/bash
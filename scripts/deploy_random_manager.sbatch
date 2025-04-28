#!/bin/zsh

#SBATCH -J lis-rand # 任务名
#SBATCH -N 1 # 申请节点数，根据用户配额，设置为1即可
#SBATCH -c 8 # CPU 核心数
#SBATCH -G 0 # 申请的GPU数量
#SBATCH -o "logs/terminal_random.log"
#SBATCH -q debug

module purge
module add anaconda3 cuda-11.7
conda activate lis-sci311

python -m src.run_selection \
	--general_config "configs/general.yaml" \
	--manager_config "configs/random_manager.yaml"

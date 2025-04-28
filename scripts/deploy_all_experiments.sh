#!/bin/zsh

sbatch scripts/deploy_CUCB_manager.sbatch

sbatch scripts/deploy_epsilon_greedy_manager.sbatch

sbatch scripts/deploy_Thompson_sampling_manager.sbatch

sbatch scripts/deploy_random_manager.sbatch

sbatch scripts/deploy_oracle_manager.sbatch

sleep 5s

squeue
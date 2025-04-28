#!/bin/zsh

#SBATCH -J lis-kasz
#SBATCH -N 1
#SBATCH -c 30
#SBATCH -G 0
#SBATCH -o "logs/terminal_CUCB.log"
#SBATCH -q debug

module purge
module add anaconda3 cuda-11.7
conda activate lis-sci311

for kappa_file in configs/general_kappas_subzero/*.yaml; do
    (
        echo "Running with config file: $kappa_file"
        python -m src.run_selection \
            --general_config "$kappa_file" \
            --manager_config "configs/CUCB_manager.yaml"
    ) &
done

wait
echo "All subsero kappa experiments are done!"

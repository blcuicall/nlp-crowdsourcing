<h1 align="center" > NLP Crowdsourcing </h1>

<p align="center">
 📝<a href="https://arxiv.org/abs/2305.06683" target="_blank"> Paper </a> • 📊<a href="./nlp_crowdsourcing_poster.pdf" target="_blank"> Poster </a> • 👋 Visit our <a href="https://blcuicall.org/" target="_blank">official website</a>
</p>

This repository contains code and data for the paper **Cost-Optimized Crowdsourcing for NLP via Worker Selection and Data Augmentation** published on IEEE Transactions of Network Sciences and Engineering.

If you are interested in our research, please visit our official website: [ICALL Research Group at Beijing Language and Culture University](https://blcuicall.org/).


## What's New

**2025/04/01:** Our paper **Cost-Optimized Crowdsourcing for NLP via Worker Selection and Data Augmentation** has been accepted by IEEE Transactions of Network Sciences and Engineering! 🎉🎉🎉

**2024/07/27:** We presented our work with a poster at CCL 2024. The poster can be found [here](nlp_crowdsourcing_poster.pdf).

**2024/05/25:** Our paper **Cost-efficient Crowdsourcing for Span-based Sequence Labeling: Worker Selection and Data Augmentation** has been accepted by CCL 2024! 🎉🎉🎉

**2023/05/11:** We uploaded the [initial version](https://arxiv.org/abs/2305.06683v1) of out paper to arXiv.

## Installation

1. Clone this repository:
    ```shell
    git clone https://github.com/blcuicall/nlp-crowdsourcing.git
    cd ./nlp-crowdsourcing
    ```
2. Create a conda environment:
    ```shell
    conda create -n nlp-crowdsourcing python=3.11
    conda activate nlp-crowdsourcing
    ```
   This is necessary since the experiments are run in different screen sessions in which this environment will be activated automatically.

3. Install the requirements:
    ```shell
    pip install -r requirements.txt
    ```

## Usage

To reproduce the experiments mentioned in the paper, you can simply run the provided shell script like:

```shell
chmod +x ./script/deploy_all_experiments.sh 
./script/deploy_all_experiments.sh
```

This script will deploy the experiments on the slurm system. If you dont use slurm on your machine, you may check each independent script and run the shell command manually, like:

```shell
python -m src.run_selection \
	--general_config "configs/general.yaml" \
	--manager_config "configs/CUCB_manager.yaml"
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Reference

If you find our work helpful, please consider citing the following paper.

```bibtex
@article{yang-etal-2025-cost-optimized,
   title = {Cost-Optimized Crowdsourcing for NLP via Worker Selection and Data Augmentation},
   author = {Yang, Liner and Wang, Yujie and Fang, Zhixuan and Huang, Yaping and Yang, Erhong},
   journal = {IEEE Transactions on Network Science and Engineering},
   year = {2025},
   volume = {},
   number = {},
   pages = {1-18},
   doi = {10.1109/TNSE.2025.3559342} }

@inproceedings{wang-etal-2024-crowdsourcing-span,
   title = {Cost-efficient Crowdsourcing for Span-based Sequence Labeling: Worker Selection and Data Augmentation},
   author = {Yujie Wang and Chao Huang and Liner Yang and Zhixuan Fang and Yaping Huang and Yang Liu and Jingsi Yu and Erhong Yang},
   booktitle = {CCL},
   month = {July},
   year = {2024},
}
```
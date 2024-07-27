<h1 align="center" > NLP Crowdsourcing </h1>

<p align="center">
 📝<a href="https://arxiv.org/abs/2305.06683" target="_blank"> Paper </a> • 📊<a href="./nlp_crowdsourcing_poster.pdf" target="_blank"> Poster </a> • 👋 Visit our <a href="https://blcuicall.org/" target="_blank">official website</a>
</p>

This repository contains code and data for the paper **Cost-efficient Crowdsourcing for Span-based Sequence Labeling: Worker Selection and Data Augmentation** presented on CCL 2024. 

If you are interested in our research, please visit our official website: [ICALL Research Group at Beijing Language and Culture University](https://blcuicall.org/).


## What's New

**2024/07/27:** We presented our work with a poster at CCL 2024. The poster can be found [here](nlp_crowdsourcing_poster.pdf).

**2024/05/25:** Our paper has been accepted by CCL 2024! 🎉🎉🎉

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

To run the experiments mentioned in the paper, you can use the shell scripts provided in this repo like:

```shell
./run_table_tests.sh [oei|conll]
```

We provide five scripts for different experiments.

The script `./run_table_tests.sh` reproduces the numerical results in Table 3 and 4 in the paper. 
```shell
./run_table_tests.sh [oei|conll]
```

The script `run_regret_tests.sh` compares different CMAB algorithms on the regret metric. 
```shell
./run_regret_tests.sh [oei|conll]
```

The script `run_epsilon_tests.sh` tests for the best Epsilon value in the Epsilon-Greedy algorithm. 
```shell
./run_epsilon_tests.sh [oei|conll]
```

The script `run_ucb_scale_tests.sh` tests for the best UCB scale value in the CUCB algorithm. 
```shell
./run_ucb_scale_tests.sh [oei|conll]
```

The script `run_kappa_tests.sh` tests for the best kappa threshold in the combined feedback mechanism.
```shell
./run_kappa_tests.sh [oei|conll]
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Reference

If you find our work helpful, please consider citing the following paper.

```bibtex
@misc{wang2023costefficientcrowdsourcingspanbasedsequence,
      title={Cost-efficient Crowdsourcing for Span-based Sequence Labeling: Worker Selection and Data Augmentation}, 
      author={Yujie Wang and Chao Huang and Liner Yang and Zhixuan Fang and Yaping Huang and Yang Liu and Erhong Yang},
      year={2023},
      eprint={2305.06683},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.06683}, 
}
```
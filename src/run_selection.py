import argparse
import json
import os.path
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import numpy as np
import yaml
from tqdm import tqdm

from src.dataset import YACLCDataset, Dataset
from src.manager import Parameters, Manager, OracleManager, CUCBManager, RandomManager, EpsilonGreedyManager, ThompsonSamplingManager

parser = argparse.ArgumentParser()

parser.add_argument('-mc', '--manager_config', type=str, default='configs/Thompson_sampling_manager.yaml')
parser.add_argument('-gc', '--general_config', type=str, default='configs/general.yaml')

args = parser.parse_args()

with open(args.general_config, 'r') as reader:
    general_config = yaml.safe_load(reader)

with open(args.manager_config, 'r') as reader:
    manager_config = yaml.safe_load(reader)


def run_average(num_run=10) -> NoReturn:
    start_time = datetime.now()

    parameters = Parameters(num_workers_in_step=general_config['num_workers_in_step'],
                            num_anno_per_sent=general_config['num_anno_per_sent'],
                            kappa_threshold=general_config['kappa_threshold'],)
    if manager_config['manager_class'] == 'CUCBManager':
        parameters.ucb_scale = manager_config['ucb_scale']
        parameters.evaluation_type = manager_config['evaluation_type']
    if manager_config['manager_class'] == 'EpsilonGreedyManager':
        parameters.epsilon = manager_config['epsilon']
        parameters.evaluation_type = manager_config['evaluation_type']
    if manager_config['manager_class'] == 'ThompsonSamplingManager':
        parameters.sample_num = manager_config['sample_num']
        parameters.evaluation_type = manager_config['evaluation_type']

    dataset = YACLCDataset()
    dataset.read_data(use_cache=general_config['use_cache'], cache_path=general_config['cache_path'])

    manager_class = globals()[manager_config['manager_class']]

    avg_scores: list[float] = []
    regret_histories: list[list[float]] = []

    expert_usage_percents: list[float] = []

    mv_hits_on_experts: list[int] = []

    # with tqdm(total=num_run, desc='Total run') as pbar:
    for i in range(num_run):
        with open('logs/kappa_tests.log', 'a') as writer:
            print(f'Run #{i + 1}/{num_run} with {manager_class} using kappa={parameters.kappa_threshold}',
                  file=writer, flush=True)
        manager = manager_class(dataset=dataset, parameters=parameters)
        manager.run(-1)
        avg_score, regret_history = manager.get_results()
        avg_scores.append(avg_score)
        regret_histories.append(regret_history)
        if manager_config['manager_class'] in ['CUCBManager', 'EpsilonGreedyManager', 'ThompsonSamplingManager']:
            expert_usage_percents.append(manager.expert_evaluation_usage / len(manager.dataset.sents))
            if manager.expert_evaluation_usage == len(manager.dataset.sents):
                mv_hits_on_experts.append(0)
            else:
                mv_hits_on_experts.append(manager.mv_hits_on_expert / (len(manager.dataset.sents) - manager.expert_evaluation_usage))
            # pbar.update(1)

            # Temporary dump the worker2current_scores to a file to monitor the progress.
            # with open('out/worker2current_scores.json', 'w') as writer:
            #     json.dump({str(worker): scores for worker, scores in manager.worker2current_scores.items()}, writer, ensure_ascii=False, indent=2)


    results_path = Path(os.path.join(general_config['output_dir'], f'results.json'))
    if results_path.exists():
        with open(results_path, 'r') as reader:
            manager2results = json.load(reader)
    else:
        manager2results = {}

    if manager_config['manager_class'] not in manager2results.keys():
        manager2results[manager_config['manager_class']] = []

    new_result = {
        'start_time': str(start_time),
        'end_time': str(datetime.now()),
        'parameters': parameters.__dict__,
        'avg_f1': np.average(avg_scores),
        'f1_of_runs': avg_scores,
    }

    if expert_usage_percents:
        new_result['avg_expert_usage'] = np.average(expert_usage_percents)
        new_result['expert_usage_of_runs'] = expert_usage_percents
        new_result['mv_hits_on_experts'] = mv_hits_on_experts

    new_result['avg_regret_history'] = [np.average(step_regret) for step_regret in zip(*regret_histories)]

    updated_results = [new_result]
    for result in manager2results[manager_config['manager_class']]:
        if result['parameters'] != parameters.__dict__:
            updated_results.append(result)

    manager2results[manager_config['manager_class']] = updated_results

    with open(results_path, 'w') as writer:
        json.dump(manager2results, writer, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    run_average(general_config['num_run'])

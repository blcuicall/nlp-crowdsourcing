import argparse

from algorithm import run_average, run_once
from utils import run_cache_annotations

argparser = argparse.ArgumentParser()

argparser.add_argument('--average_over', type=int, default=1)
argparser.add_argument('--num_steps', type=int, default=10000)
argparser.add_argument('--annotation_num_per_sentence', type=int, default=1)
argparser.add_argument('--allow_exhaustion', action='store_true')
argparser.add_argument('--evaluation_interval', type=int, default=0)
argparser.add_argument('--manager_type', type=str, default='normal', choices=['normal', 'mv_expert', 'mv', 'kappa_agg',
                                                                              'best', 'worst', 'random',
                                                                              'epsilon_greedy', 'thompson_sampling'])
argparser.add_argument('--allow_fake_annotations', action='store_true')
argparser.add_argument('--only_fake_annotations', action='store_true')
argparser.add_argument('--metrics_type', type=str, default='span_exact', choices=['span_exact', 'span_proportional', 'token', 'pearson'])
argparser.add_argument('--split_spans', action='store_true')
argparser.add_argument('--ucb_scale', type=float, default=1.0)
argparser.add_argument('--acceptable_mv_char_error_nums', type=int, default=0)
argparser.add_argument('--acceptable_mv_percentage', type=float, default=0.67)
argparser.add_argument('--fleiss_kappa_threshold', type=float, default=0.5)
argparser.add_argument('--agg_method', type=str, default='mv', choices=['mv', 'bsc'])
argparser.add_argument('--use_fake_annotation_cache', action='store_true')
argparser.add_argument('--epsilon', type=float)
argparser.add_argument('--selected_worker_num', type=int, default=20)

argparser.add_argument('--cache_annotations', action='store_true')


if __name__ == '__main__':
    args = argparser.parse_args()

    if args.cache_annotations:
        run_cache_annotations(metrics_type=args.metrics_type)
    elif args.average_over == 1:
        run_args = vars(args)
        run_args.pop('average_over')
        run_args.pop('cache_annotations')
        run_once(**run_args)
    else:
        run_args = vars(args)
        run_args.pop('cache_annotations')
        run_average(**run_args)
        # run_average(**vars(args))
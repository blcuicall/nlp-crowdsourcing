import argparse

from algorithm import run_average, run_once

argparser = argparse.ArgumentParser()

argparser.add_argument('--average_over', type=int, default=1)
argparser.add_argument('--num_steps', type=int, default=10000)
argparser.add_argument('--annotation_num_per_sentence', type=int, default=1)
argparser.add_argument('--allow_exhaustion', action='store_true')
argparser.add_argument('--evaluation_interval', type=int, default=0)
argparser.add_argument('--manager_type', type=str, default='normal', choices=['normal', 'mv_expert', 'random', 'best'])
argparser.add_argument('--allow_fake_annotations', action='store_true')
argparser.add_argument('--metrics_type', type=str, default='span_exact', choices=['span_exact', 'span_proportional', 'token'])
argparser.add_argument('--ucb_scale', type=float, default=1.0)
argparser.add_argument('--mv_ratio', type=float, default=0.1)

if __name__ == '__main__':
    args = argparser.parse_args()

    if args.average_over == 1:
        run_once(num_steps=args.num_steps,
                 annotation_num_per_sentence=args.annotation_num_per_sentence,
                 allow_exhaustion=args.allow_exhaustion,
                 evaluation_interval=args.evaluation_interval,
                 manager_type=args.manager_type,
                 allow_fake_annotations=args.allow_fake_annotations,
                 metrics_type=args.metrics_type,
                 ucb_scale=args.ucb_scale,
                 mv_ratio=args.mv_ratio)
    else:
        run_average(**vars(args))


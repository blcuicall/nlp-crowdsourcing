#!/usr/bin/env bash

kappas=(-1 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 3)

if [ "$1" = "oei" ]; then
    test_path="src"
    use_gold="--use_gold_expert"
    ucb_scale=0.4
elif [ "$1" = "conll" ]; then
    test_path="conll_test"
    use_gold=""
    ucb_scale=0.2
fi

for kappa in "${kappas[@]}" ; do
  sc_name="$1-k=${kappa}-kappa"
  echo $sc_name
  screen -dmS $sc_name
  screen -x -S $sc_name -X stuff "conda activate nlp-crowdsourcing"
  screen -x -S $sc_name -X stuff $'\n'
  screen -x -S $sc_name -X stuff "python ${test_path}/main.py \
    --average_over 10 \
    --num_steps 10000 \
    --annotation_num_per_sentence 4 \
    --allow_exhaustion \
    --evaluation_interval 0 \
    --allow_fake_annotations \
    --manager_type kappa_agg \
    --metrics_type span_proportional \
    --ucb_scale ${ucb_scale} \
    --agg_method mv \
    --use_fake_annotation_cache ${use_gold} \
    --fleiss_kappa_threshold ${kappa}"
  screen -x -S $sc_name -X stuff $'\n'
  sleep 1s
done


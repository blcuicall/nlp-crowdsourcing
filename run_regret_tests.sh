#!/usr/bin/env bash

manager_types=(kappa_agg epsilon_greedy thompson_sampling random)

average_over=10

if [ "$1" = "oei" ]; then
    test_path="src"
    use_gold="--use_gold_expert"
    kappas=(-1 0.4 3)
    epsilon=0.025
    ucb_scale=0.4
    selected_worker_num=20
elif [ "$1" = "conll" ]; then
    test_path="conll_test"
    use_gold=""
    kappas=(-1 0.65 3)
    epsilon=0.025
    ucb_scale=0.2
    selected_worker_num=20
fi

for manager_type in "${manager_types[@]}"; do
  if [ "$manager_type" = "random" ]; then
    sc_name="${test_path}-${manager_type}-regret"
    echo $sc_name
    screen -dmS $sc_name
    screen -x -S $sc_name -X stuff "conda activate nlp-crowdsourcing"
    screen -x -S $sc_name -X stuff $'\n'
    screen -x -S $sc_name -X stuff "python ${test_path}/main.py \
      --average_over ${average_over} \
      --num_steps 10000 \
      --annotation_num_per_sentence 10 \
      --allow_exhaustion \
      --evaluation_interval 0 \
      --allow_fake_annotations \
      --manager_type ${manager_type} \
      --metrics_type span_proportional \
      --selected_worker_num ${selected_worker_num} \
      --use_fake_annotation_cache ${use_gold}"
    screen -x -S $sc_name -X stuff $'\n'
    sleep 1s
  else
    for kappa in "${kappas[@]}"; do
      sc_name="${test_path}-${manager_type}-${kappa}-regret"
      echo $sc_name
      screen -dmS $sc_name
      screen -x -S $sc_name -X stuff "conda activate nlp-crowdsourcing"
      screen -x -S $sc_name -X stuff $'\n'
      screen -x -S $sc_name -X stuff "python ${test_path}/main.py \
        --average_over ${average_over} \
        --num_steps 10000 \
        --annotation_num_per_sentence 10 \
        --allow_exhaustion \
        --evaluation_interval 0 \
        --allow_fake_annotations \
        --manager_type ${manager_type} \
        --metrics_type span_proportional \
        --ucb_scale ${ucb_scale} \
        --epsilon ${epsilon} \
        --agg_method mv \
        --use_fake_annotation_cache ${use_gold} \
        --selected_worker_num ${selected_worker_num} \
        --fleiss_kappa_threshold ${kappa}"
      screen -x -S $sc_name -X stuff $'\n'
      sleep 1s
    done
  fi
done

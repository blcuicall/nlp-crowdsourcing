#!/usr/bin/env bash

manager_types=(kappa_agg epsilon_greedy thompson_sampling random best)
metrics=(span_proportional span_exact token)

if [ "$1" = "oei" ]; then
    test_path="src"
    use_gold="--use_gold_expert"
    kappas=(-1 0.4 3)
    ucb_scale=0.4
    epsilon=0.025
elif [ "$1" = "conll" ]; then
    test_path="conll_test"
    use_gold=""
    kappas=(-1 0.65 3)
    ucb_scale=0.2
    epsilon=0.025
fi

for manager_type in "${manager_types[@]}"; do
  for metric in "${metrics[@]}" ; do
    if [ "$manager_type" = "random" ]; then
      sc_name="$1-${manager_type}-${metric}-table"
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
        --manager_type ${manager_type} \
        --metrics_type ${metric} \
        --use_fake_annotation_cache ${use_gold}"
      screen -x -S $sc_name -X stuff $'\n'
      sleep 1s
    elif [ "$manager_type" = "best" ]; then
      sc_name="$1-${manager_type}-${metric}-table"
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
        --manager_type ${manager_type} \
        --metrics_type ${metric} \
        --use_fake_annotation_cache ${use_gold}"
      screen -x -S $sc_name -X stuff $'\n'
      sleep 1s
    elif [ "$manager_type" = "kappa_agg" ]; then
      for kappa in "${kappas[@]}" ; do
        sc_name="$1-${manager_type}-${metric}-${kappa}-table"
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
          --manager_type ${manager_type} \
          --metrics_type ${metric} \
          --agg_method mv \
          --ucb_scale ${ucb_scale} \
          --use_fake_annotation_cache ${use_gold} \
          --fleiss_kappa_threshold ${kappa}"
        screen -x -S $sc_name -X stuff $'\n'
        sleep 1s
      done
    elif [ "$manager_type" = "epsilon_greedy" ]; then
      for kappa in "${kappas[@]}" ; do
        sc_name="$1-${manager_type}-${metric}-${kappa}-table"
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
          --manager_type ${manager_type} \
          --metrics_type ${metric} \
          --agg_method mv \
          --ucb_scale ${ucb_scale} \
          --epsilon ${epsilon} \
          --use_fake_annotation_cache ${use_gold} \
          --fleiss_kappa_threshold ${kappa}"
        screen -x -S $sc_name -X stuff $'\n'
        sleep 1s
      done
    elif [ "$manager_type" = "thompson_sampling" ]; then
      for kappa in "${kappas[@]}" ; do
        sc_name="$1-${manager_type}-${metric}-${kappa}-table"
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
          --manager_type ${manager_type} \
          --metrics_type ${metric} \
          --agg_method mv \
          --ucb_scale ${ucb_scale} \
          --use_fake_annotation_cache ${use_gold} \
          --fleiss_kappa_threshold ${kappa}"
        screen -x -S $sc_name -X stuff $'\n'
        sleep 1s
      done
    fi
  done
done

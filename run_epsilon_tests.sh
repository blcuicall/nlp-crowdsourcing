#!/usr/bin/env bash

epsilons=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)

if [ "$1" = "oei" ]; then
    test_path="src"
    use_gold="--use_gold_expert"
    kappas=(-1 0.4 3)
elif [ "$1" = "conll" ]; then
    test_path="conll_test"
    use_gold=""
    kappas=(-1 0.65 3)
fi

for epsilon in "${epsilons[@]}" ; do
  for kappa in "${kappas[@]}" ; do
    sc_name="$1-k=${kappa}-e=${epsilon}"
    echo $sc_name
    screen -dmS $sc_name
    screen -x -S $sc_name -X stuff "conda activate nlp-crowdsourcing"
    screen -x -S $sc_name -X stuff $'\n'
    screen -x -S $sc_name -X stuff "python ${test_path}/main.py \
      --average_over 5 \
      --num_steps 10000 \
      --annotation_num_per_sentence 4 \
      --allow_exhaustion \
      --evaluation_interval 0 \
      --allow_fake_annotations \
      --manager_type epsilon_greedy \
      --metrics_type span_proportional \
      --agg_method mv \
      --use_fake_annotation_cache ${use_gold} \
      --fleiss_kappa_threshold ${kappa} \
      --epsilon ${epsilon}"
    screen -x -S $sc_name -X stuff $'\n'
    sleep 1s
  done
done


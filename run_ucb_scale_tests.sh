#!/usr/bin/env bash

ucb_scales=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
kappas=(0.65)

if [ "$1" = "oei" ]; then
    test_path="src"
    use_gold="--use_gold_expert"
elif [ "$1" = "conll" ]; then
    test_path="conll_test"
    use_gold=""
fi

for ucb_scale in "${ucb_scales[@]}" ; do
  for kappa in "${kappas[@]}" ; do
    sc_name="${test_path}-k=${kappa}-u=${ucb_scale}"
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
      --manager_type kappa_agg \
      --metrics_type span_proportional \
      --agg_method mv \
      --use_fake_annotation_cache ${use_gold} \
      --fleiss_kappa_threshold ${kappa} \
      --ucb_scale ${ucb_scale}"
    screen -x -S $sc_name -X stuff $'\n'
    sleep 1s
  done
done


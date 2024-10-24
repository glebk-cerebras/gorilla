#!/bin/bash
set -x

#models=("mistral-large-2407-FC")

datasets=('simple' 'irrelevance' 'parallel' 'multiple' 'parallel_multiple' 'java' 'javascript' 'rest' 'live_simple' 'live_multiple' 'live_parallel' 'live_parallel_multiple' 'live_irrelevance' 'live_relevance' 'multi_turn_base' 'multi_turn_long_context' 'multi_turn_miss_func' 'multi_turn_miss_param')

#datasets=('simple' 'irrelevance')
#datasets=('all')

models=(${1})
export MISTRAL_URL=${2}

for model in "${models[@]}"; do
    for data in "${datasets[@]}"; do
        bfcl generate --model $model --test-category $data       
    done
done


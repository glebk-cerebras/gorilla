#!/bin/bash
set -x

#datasets=('simple' 'irrelevance' 'parallel' 'multiple' 'parallel_multiple' 'java' 'javascript' 'rest' 'live_simple' 'live_multiple' 'live_parallel' 'live_parallel_multiple' 'live_irrelevance' 'live_relevance' 'multi_turn_base' 'multi_turn_long_context' 'multi_turn_miss_func' 'multi_turn_miss_param')

datasets=('simple')
#datasets=('all')

models=(${1})
export CEREBRAS_API_KEY=${2}

for model in "${models[@]}"; do
    for data in "${datasets[@]}"; do
        bfcl generate --model $model --test-category $data
    done
done

bfcl evaluate --model $model


#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

model=medium
modelexpdir=/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_${model}_merged

for lang in "ara" "cmn" "kor" "rus" "spa"; do
    for target_dir in ${modelexpdir}/${lang}/decode/{train-all_sp,train-cts_sp}/*/asr/{none,lora}/merged; do
        # If the target directory does not exist, skip
        if [ ! -d "${target_dir}" ]; then
            continue
        fi
        prefix=${target_dir%/*}
        echo "Renaming ${prefix}/merged to ${prefix}/merged_merged"
        mv ${prefix}/merged ${prefix}/merged_merged
    done
done
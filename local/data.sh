#!/usr/bin/env bash

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

stage=1
stop_stage=2
python=python
conda_env=scale
src_lang=ara
tgt_lang=eng
seed=1234
mean=15
std=7
t_min=5
t_max=25
num_workers=$(nproc)
sampling_rate=16000
audio_format=flac
output_root=
train_sr_stms=
train_st_stms=
dev_sr_stms=
dev_st_stms=
test_sr_stms=
test_st_stms=
contrastive_ratios="0.5 0.25 0.25"

. "${repo_root}/utils/parse_options.sh"

if [ -z "${output_root}" ]; then
    output_root="${WHISPER_UT_DATA_ROOT:-${repo_root}/data}/${src_lang}"
fi

mkdir -p "${output_root}"
manifest_dir="${output_root}/manifests"
audio_dir="${output_root}/audio"
stm_dir="${output_root}/stm"
hf_dir="${output_root}/hf"

read_list_var() {
    local value=$1
    local target_name=$2
    local -n target_ref="${target_name}"
    target_ref=()
    if [ -n "${value}" ]; then
        # shellcheck disable=SC2206
        target_ref=(${value})
    fi
}

read_list_var "${train_sr_stms}" train_sr_array
read_list_var "${train_st_stms}" train_st_array
read_list_var "${dev_sr_stms}" dev_sr_array
read_list_var "${dev_st_stms}" dev_st_array
read_list_var "${test_sr_stms}" test_sr_array
read_list_var "${test_st_stms}" test_st_array

if [ ${#train_sr_array[@]} -eq 0 ] || [ ${#train_sr_array[@]} -ne ${#train_st_array[@]} ]; then
    log "Error: training SR/ST STM lists must both be non-empty and have the same length."
    exit 2
fi

if [ ${#dev_sr_array[@]} -ne ${#dev_st_array[@]} ]; then
    log "Error: dev SR/ST STM lists must have the same length."
    exit 2
fi

if [ ${#test_sr_array[@]} -ne ${#test_st_array[@]} ]; then
    log "Error: test SR/ST STM lists must have the same length."
    exit 2
fi

cd "${repo_root}"

if command -v conda >/dev/null 2>&1; then
    # Match the environment requested for the STM-first data prep flow.
    # shellcheck disable=SC1091
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_env}"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Build STM-first manifests for ${src_lang}-${tgt_lang} data prep."
    "${python}" pyscripts/utils/stm_recipe_merge_map.py \
        --train-sr-stms "${train_sr_array[@]}" \
        --train-st-stms "${train_st_array[@]}" \
        --dev-sr-stms "${dev_sr_array[@]}" \
        --dev-st-stms "${dev_st_array[@]}" \
        --test-sr-stms "${test_sr_array[@]}" \
        --test-st-stms "${test_st_array[@]}" \
        --output-dir "${manifest_dir}" \
        --src-lang "${src_lang}" \
        --tgt-lang "${tgt_lang}" \
        --seed "${seed}" \
        --mean "${mean}" \
        --std "${std}" \
        --t-min "${t_min}" \
        --t-max "${t_max}" \
        --ratios ${contrastive_ratios}
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Materialize audio, STM, and Hugging Face datasets."
    "${python}" pyscripts/utils/stm_recipe_materialize.py \
        --manifest-dir "${manifest_dir}" \
        --audio-dir "${audio_dir}" \
        --stm-dir "${stm_dir}" \
        --hf-dir "${hf_dir}" \
        --src-lang "${src_lang}" \
        --tgt-lang "${tgt_lang}" \
        --num-workers "${num_workers}" \
        --sampling-rate "${sampling_rate}" \
        --audio-format "${audio_format}"
fi

log "Finished STM-first data preparation for ${src_lang}-${tgt_lang}."

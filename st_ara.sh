#!/usr/bin/env bash

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}" && pwd)
workspace_root=$(cd "${repo_root}/.." && pwd)

conda_env=scale
python=python
python_hf=python3

src_lang=ara
tgt_lang=eng

hf_datadir="/exp/ahussein/whisper-ut/data_prep/hf"

train_set=train_3way_merged
valid_set=dev1_org
test_sets="dev2_org iwslt22_test_org"

on_the_fly_feat=true
peft_method=lora
st_config=conf/tuning/mtl_large-v2_ara_lora_train_all_merged.yaml
model_name=large-v2
ds_config=conf/tuning/ds2.json
normalize_text=false
speed_perturb_factors="0.9 1.0 1.1"
preprocessing_num_proc=32
master_port=29501
ngpu=8
inference_batch_size=16
use_gpu_inference=true
num_beams=1
prepare_keyfiles=true
force_rebuild_keyfiles=false
inference_checkpoint=
stage=0
stop_stage=3
st_tag=iwslt22
output_dir=
decode_dir_base=
score_dir_base=

help_message=$(
    cat <<EOF
Usage: $0 [wrapper-options] [hf_whisper_ft-options]

Wrapper defaults:
  - uses ${src_lang}.${train_set} for ST training
  - uses ${src_lang}.${valid_set} for validation
  - uses "${test_sets}" for ST inference/evaluation
  - runs direct Whisper ST with --mode st

Wrapper options:
  --hf_datadir PATH
  --src_lang LANG
  --tgt_lang LANG
  --train_set NAME
  --valid_set NAME
  --test_sets "NAME [NAME ...]"
  --on_the_fly_feat true|false
  --peft_method METHOD
  --st_config PATH
  --model_name NAME
  --ds_config PATH
  --python_hf PYTHON
  --normalize_text true|false
  --speed_perturb_factors "FACTORS"
  --preprocessing_num_proc INT
  --master_port INT
  --ngpu INT
  --inference_batch_size INT
  --use_gpu_inference true|false
  --num_beams INT
  --prepare_keyfiles true|false
  --force_rebuild_keyfiles true|false
  --inference_checkpoint NAME
  --conda_env NAME
  --python PYTHON
  --stage INT
  --stop_stage INT
  --st_tag TAG
  --output_dir PATH
  --decode_dir_base PATH
  --score_dir_base PATH

Stages:
  0: optional feature extraction via hf_whisper_ft.py --feat-extraction
  1: ST finetuning via hf_whisper_ft.py --mode st
  2: ST inference via hf_whisper_inference.py
  3: ST evaluation via score_hf_predictions.py

Layout defaults:
  - training model/checkpoints: <output_dir>
  - decoded hypotheses:        <output_dir>/decode/<dataset>/text
  - eval outputs:              <output_dir>/scores/<dataset>/

All other --name value pairs are forwarded to hf_whisper_ft.py for stages 0-1.
EOF
)

bool_is_true() {
    [ "$1" = "true" ]
}

require_arg() {
    if [ $# -lt 2 ] || [[ -z "${2:-}" ]] || [[ "${2}" == --* ]]; then
        log "Error: missing value for $1"
        exit 2
    fi
}

forward_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            printf '%s\n' "${help_message}"
            exit 0
            ;;
        --hf_datadir|--src_lang|--tgt_lang|--train_set|--valid_set|--test_sets|--on_the_fly_feat|--peft_method|--st_config|--model_name|--ds_config|--python_hf|--normalize_text|--speed_perturb_factors|--preprocessing_num_proc|--master_port|--ngpu|--inference_batch_size|--use_gpu_inference|--num_beams|--prepare_keyfiles|--force_rebuild_keyfiles|--inference_checkpoint|--conda_env|--python|--stage|--stop_stage|--st_tag|--output_dir|--decode_dir_base|--score_dir_base)
            require_arg "$1" "${2:-}"
            name=${1#--}
            name=${name//-/_}
            printf -v "${name}" '%s' "$2"
            shift 2
            ;;
        --*)
            require_arg "$1" "${2:-}"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        *)
            log "Error: no positional arguments are supported: $1"
            exit 2
            ;;
    esac
done

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_env}"
fi

validate_dataset_dir() {
    local dataset_name=$1
    local dataset_dir="${hf_datadir}/${src_lang}.${dataset_name}"
    if [ ! -d "${dataset_dir}" ]; then
        log "Error: required dataset not found: ${dataset_dir}"
        exit 2
    fi
}

ensure_keyfile_for_set() {
    local dataset_name=$1
    local dataset_dir="${hf_datadir}/${src_lang}.${dataset_name}"
    local keyfile="${hf_datadir}/${dataset_name}.wav.scp"
    local force_keyfile_args=()
    validate_dataset_dir "${dataset_name}"
    if bool_is_true "${force_rebuild_keyfiles}"; then
        force_keyfile_args+=(--force)
    fi
    if [ ! -f "${keyfile}" ] || bool_is_true "${force_rebuild_keyfiles}"; then
        if ! bool_is_true "${prepare_keyfiles}"; then
            log "Error: keyfile missing and --prepare_keyfiles is false: ${keyfile}"
            exit 2
        fi
        log "Ensuring keyfile ${keyfile}"
        "${python}" "${repo_root}/pyscripts/utils/concat_hf_datasets.py" \
            --inputs "${dataset_dir}" \
            --wav-scp "${keyfile}" \
            "${force_keyfile_args[@]}"
    fi
    echo "${keyfile}"
}

validate_dataset_dir "${train_set}"
validate_dataset_dir "${valid_set}"

feat_type=feats
if bool_is_true "${on_the_fly_feat}"; then
    feat_type=raw
fi
save_feature_dir="${hf_datadir}/features/${feat_type}"
mkdir -p "${hf_datadir}" "${save_feature_dir}"

if [ -z "${output_dir}" ]; then
    output_dir="${repo_root}/exp/hf_${st_tag}/${src_lang}/${train_set}/st/${peft_method}"
fi
if [ -z "${decode_dir_base}" ]; then
    decode_dir_base="${output_dir}/decode"
fi
if [ -z "${score_dir_base}" ]; then
    score_dir_base="${output_dir}/scores"
fi
mkdir -p "${output_dir}" "${decode_dir_base}" "${score_dir_base}"

base_args=(
    "${repo_root}/pyscripts/utils/hf_whisper_ft.py"
    --mode st
    --train-set "${train_set}"
    --src-lang "${src_lang}"
    --tgt-lang "${tgt_lang}"
    --hf_datadir "${hf_datadir}"
    --dev-name "${valid_set}"
    --model_name "${model_name}"
    --preprocessing_num_proc "${preprocessing_num_proc}"
    --save_feature_dir "${save_feature_dir}"
)

if bool_is_true "${normalize_text}"; then
    base_args+=(--normalize_text)
fi

if bool_is_true "${on_the_fly_feat}"; then
    base_args+=(--on-the-fly-feat-extraction)
fi

if [ -n "${st_config}" ]; then
    base_args+=(--config "${st_config}")
fi

if [ -n "${ds_config}" ]; then
    base_args+=(--deepspeed "${ds_config}")
fi

if [ "${peft_method}" != none ]; then
    base_args+=(--peft_method "${peft_method}")
fi

if [ -n "${speed_perturb_factors}" ]; then
    read -r -a speed_factors <<<"${speed_perturb_factors}"
    if [ "${#speed_factors[@]}" -gt 0 ]; then
        base_args+=(--speed-perturb-factors "${speed_factors[@]}")
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: ST feature extraction for train_set=${train_set}, valid_set=${valid_set}"
    "${python_hf}" "${base_args[@]}" --feat-extraction "${forward_args[@]}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: ST finetuning with output_dir=${output_dir}"
    if [ "${ngpu}" -gt 1 ]; then
        "${python_hf}" -m torch.distributed.launch --nproc_per_node "${ngpu}" --master_port "${master_port}" \
            "${base_args[@]}" \
            --output_dir "${output_dir}" \
            "${forward_args[@]}"
    else
        "${python_hf}" "${base_args[@]}" \
            --output_dir "${output_dir}" \
            "${forward_args[@]}"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: ST inference on test sets: ${test_sets}"
    modeldir="${output_dir}"
    if [ -n "${inference_checkpoint}" ]; then
        modeldir="${output_dir}/${inference_checkpoint}"
    fi
    if [ ! -d "${modeldir}" ]; then
        log "Error: inference model directory not found: ${modeldir}"
        exit 2
    fi

    for dset in ${test_sets}; do
        validate_dataset_dir "${dset}"
        key_file=$(ensure_keyfile_for_set "${dset}")
        decode_dir="${decode_dir_base}/${dset}"
        mkdir -p "${decode_dir}"

        infer_args=(
            "${repo_root}/pyscripts/utils/hf_whisper_inference.py"
            --keyfile "${key_file}"
            --dset "${hf_datadir}/${src_lang}.${dset}"
            --src-lang "${src_lang}"
            --tgt-lang "${tgt_lang}"
            --output_dir "${decode_dir}"
            --task translate
            --model_name "${model_name}"
            --batch-size "${inference_batch_size}"
            --num-beams "${num_beams}"
        )
        if [ "${peft_method}" != none ]; then
            infer_args+=(--peft-model "${modeldir}")
        else
            infer_args+=(--pretrained-model "${modeldir}")
        fi

        log "Running ST inference for ${dset} -> ${decode_dir}/text"
        if bool_is_true "${use_gpu_inference}"; then
            "${python_hf}" "${infer_args[@]}"
        else
            CUDA_VISIBLE_DEVICES="" "${python_hf}" "${infer_args[@]}"
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: ST evaluation on decoded outputs"
    for dset in ${test_sets}; do
        validate_dataset_dir "${dset}"
        hyp_file="${decode_dir_base}/${dset}/text"
        if [ ! -f "${hyp_file}" ]; then
            log "Error: missing decoded hypothesis file: ${hyp_file}"
            log "Run with --stage 2 first (or include stage 2 in this run)."
            exit 2
        fi
        score_dir="${score_dir_base}/${dset}"
        score_args=(
            "${repo_root}/pyscripts/utils/score_hf_predictions.py"
            --dataset "${hf_datadir}/${src_lang}.${dset}"
            --hyp-file "${hyp_file}"
            --task st
            --score-dir "${score_dir}"
        )
        if bool_is_true "${normalize_text}"; then
            score_args+=(--normalize-text)
        fi
        log "Scoring ST output for ${dset} -> ${score_dir}"
        "${python_hf}" "${score_args[@]}"
    done
fi

log "Done. model=${output_dir}, decode=${decode_dir_base}, scores=${score_dir_base}"

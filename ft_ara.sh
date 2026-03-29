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

src_lang=ara
tgt_lang=eng

default_hf_datadir="${workspace_root}/ut_data/${src_lang}/hf"
if [ -d "${default_hf_datadir}" ]; then
    hf_datadir="${default_hf_datadir}"
else
    hf_datadir="${WHISPER_UT_DATA_ROOT:-${workspace_root}/ut_data}/${src_lang}/hf"
fi

train_set=train_3way_merged
mt_train_set=train_3way_merged
valid_set=dev1_org
test_sets="dev2_org iwslt22_test_org"

prepare_keyfiles=true
force_rebuild_keyfiles=false
on_the_fly_feat=true
peft_method=lora
mtl_config=conf/tuning/mtl_large-v2_ara_lora_train_all_merged.yaml
model_name=large-v2
ds_config=conf/tuning/ds2.json
python_hf=python3
normalize_text=false
speed_perturb_factors="0.9 1.0 1.1"
preprocessing_num_proc=32
inference_nj=4
use_gpu_inference=true
inference_batch_size=16
use_asr_prompt=true
min_promptless_prob=0.1
max_promptless_prob=0.1
batch_mask_prob=0.8
token_mask_prob=0.4
min_alpha=0.4
max_alpha=0.5
dynamic_loss_start_step=1
dynamic_loss_k=0.25
use_asr_prompt_dev=false
use_asr_prompt_decode=true
promptless_decode=false
disable_asr_inference=false
no_glm=true
num_beams=1
master_port=29501

stage=0
stop_stage=7
st_tag=iwslt22
ngpu=8

help_message=$(
    cat <<EOF
Usage: $0 [wrapper-options] [finetune-options]

Wrapper defaults:
  - uses ${src_lang}.${train_set} for the speech-backed MML branch
  - uses ${src_lang}.${mt_train_set} for the MT branch
  - uses ${valid_set} for validation
  - uses "${test_sets}" for decoding/evaluation

Wrapper options:
  --hf_datadir PATH
  --src_lang LANG
  --tgt_lang LANG
  --train_set NAME
  --mt_train_set NAME
  --valid_set NAME
  --test_sets "NAME [NAME ...]"
  --prepare_keyfiles true|false
  --force_rebuild_keyfiles true|false
  --on_the_fly_feat true|false
  --peft_method METHOD
  --mtl_config PATH
  --model_name NAME
  --ds_config PATH
  --python_hf PYTHON
  --normalize_text true|false
  --speed_perturb_factors "FACTORS"
  --preprocessing_num_proc INT
  --inference_nj INT
  --use_gpu_inference true|false
  --inference_batch_size INT
  --use_asr_prompt true|false
  --min_promptless_prob FLOAT
  --max_promptless_prob FLOAT
  --batch_mask_prob FLOAT
  --token_mask_prob FLOAT
  --min_alpha FLOAT
  --max_alpha FLOAT
  --dynamic_loss_start_step INT
  --dynamic_loss_k FLOAT
  --use_asr_prompt_dev true|false
  --use_asr_prompt_decode true|false
  --promptless_decode true|false
  --disable_asr_inference true|false
  --no_glm true|false
  --num_beams INT
  --master_port INT
  --ngpu INT
  --conda_env NAME
  --python PYTHON
  --stage INT
  --stop_stage INT
  --st_tag TAG

All other --name value pairs are forwarded to finetune.sh.
Typical example:
  $0 --ngpu 4 --model_name large-v2 --mtl_config conf/mml_kl.yaml
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
        --hf_datadir|--src_lang|--tgt_lang|--train_set|--mt_train_set|--valid_set|--test_sets|--prepare_keyfiles|--force_rebuild_keyfiles|--on_the_fly_feat|--peft_method|--mtl_config|--model_name|--ds_config|--python_hf|--normalize_text|--speed_perturb_factors|--preprocessing_num_proc|--inference_nj|--use_gpu_inference|--inference_batch_size|--use_asr_prompt|--min_promptless_prob|--max_promptless_prob|--batch_mask_prob|--token_mask_prob|--min_alpha|--max_alpha|--dynamic_loss_start_step|--dynamic_loss_k|--use_asr_prompt_dev|--use_asr_prompt_decode|--promptless_decode|--disable_asr_inference|--no_glm|--num_beams|--master_port|--ngpu|--conda_env|--python|--stage|--stop_stage|--st_tag)
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
    local required=$2
    if [ -z "${dataset_name}" ]; then
        if [ "${required}" = "required" ]; then
            log "Error: dataset name is empty for a required input"
            exit 2
        fi
        return 0
    fi
    local dataset_dir="${hf_datadir}/${src_lang}.${dataset_name}"
    if [ ! -d "${dataset_dir}" ]; then
        if [ "${required}" = "required" ]; then
            log "Error: required dataset not found: ${dataset_dir}"
            exit 2
        fi
        log "Skipping missing optional dataset ${dataset_dir}"
        return 1
    fi
    return 0
}

mkdir -p "${hf_datadir}"
force_keyfile_args=()
if bool_is_true "${force_rebuild_keyfiles}"; then
    force_keyfile_args+=(--force)
fi

validate_dataset_dir "${train_set}" required >/dev/null
validate_dataset_dir "${mt_train_set}" required >/dev/null
validate_dataset_dir "${valid_set}" required >/dev/null

ensure_keyfile_for_set() {
    local dataset_name=$1
    local dataset_dir="${hf_datadir}/${src_lang}.${dataset_name}"
    local keyfile="${hf_datadir}/${dataset_name}.wav.scp"
    validate_dataset_dir "${dataset_name}" required >/dev/null
    log "Ensuring keyfile ${keyfile}"
    "${python}" "${repo_root}/pyscripts/utils/concat_hf_datasets.py" \
        --inputs "${dataset_dir}" \
        --wav-scp "${keyfile}" \
        "${force_keyfile_args[@]}"
}

if bool_is_true "${prepare_keyfiles}"; then
    ensure_keyfile_for_set "${valid_set}"
    for dataset_name in ${test_sets}; do
        ensure_keyfile_for_set "${dataset_name}"
    done
fi

log "Launching finetune.sh with train_set=${train_set}, mt_train_set=${mt_train_set}, valid_set=${valid_set}, test_sets=${test_sets}"
bash "${repo_root}/finetune.sh" \
    --ngpu "${ngpu}" \
    --src_lang "${src_lang}" \
    --tgt_lang "${tgt_lang}" \
    --hf_datadir "${hf_datadir}" \
    --on_the_fly_feat "${on_the_fly_feat}" \
    --peft_method "${peft_method}" \
    --mtl_config "${mtl_config}" \
    --model_name "${model_name}" \
    --ds_config "${ds_config}" \
    --python_hf "${python_hf}" \
    --normalize_text "${normalize_text}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --preprocessing_num_proc "${preprocessing_num_proc}" \
    --inference_nj "${inference_nj}" \
    --use_gpu_inference "${use_gpu_inference}" \
    --inference_batch_size "${inference_batch_size}" \
    --use_asr_prompt "${use_asr_prompt}" \
    --min_promptless_prob "${min_promptless_prob}" \
    --max_promptless_prob "${max_promptless_prob}" \
    --batch_mask_prob "${batch_mask_prob}" \
    --token_mask_prob "${token_mask_prob}" \
    --min_alpha "${min_alpha}" \
    --max_alpha "${max_alpha}" \
    --dynamic_loss_start_step "${dynamic_loss_start_step}" \
    --dynamic_loss_k "${dynamic_loss_k}" \
    --use_asr_prompt_dev "${use_asr_prompt_dev}" \
    --use_asr_prompt_decode "${use_asr_prompt_decode}" \
    --promptless_decode "${promptless_decode}" \
    --disable_asr_inference "${disable_asr_inference}" \
    --no_glm "${no_glm}" \
    --num_beams "${num_beams}" \
    --master_port "${master_port}" \
    --train_set "${train_set}" \
    --mt_train_set "${mt_train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --st_tag "${st_tag}" \
    "${forward_args[@]}"

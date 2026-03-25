#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}
SECONDS=0

# Evaluation related
sclite_path=sclite

# General configuration
stage=1                      # Processes starts from the specified stage.
stop_stage=10000             # Processes is stopped at the specified stage.
ngpu=1                       # The number of gpus ("0" uses cpu, otherwise use gpu).
inference_nj=32              # The number of parallel jobs in decoding.
expdir=exp                   # Directory to save experiments.
model_name=base              # Model name, e.g. "base", "large", etc.
framework=huggingface        # huggingface, openai
hf_datadir=                  # Directory to the hugging face dataset.
preprocessing_num_proc=4     # Number of parallel jobs in preprocessing
resume_from_checkpoint=      # Resume from checkpoint path
load_model_from_path=        # Load model from path
peft_method=none             # none, lora, qlora
on_the_fly_feat=true         # Whether to generate features on the fly
debug=false                  # Whether to use debug mode
ds_config=                   # Path to the deepspeed config file
st_save_eval_preds=          # Path to store the st evaluation predictions for analysis
master_port=29500            # Port for distributed training
normalize_text=false         # Whether to normalize text before training and during validation
python_hf=python3            # Specify python to execute hugging face commands.
fe_only=false                # Whether to do feature extraction only
mtl_config=                  # Config for multi-task model training.
eval_cer=false               # Whether to evaluate CER
inference_batch_size=32      # Batch size for inference
use_asr_prompt=false         # Whether to mask the ASR hypothesis at PMTL training time
min_promptless_prob=0.4      # The minimum probability for performing promptless ST finetuning
max_promptless_prob=0.4      # The maximum probability for performing promptless ST finetuning
batch_mask_prob=0            # The minimum probability for applying masks to the prompt
token_mask_prob=0            # The probability for masking tokens in the prompt
min_alpha=0.5                # The minimum alpha for the multi-task losses, i.e. the weight for the ST loss
max_alpha=0.8                # The maximum alpha for the multi-task losses, i.e. the weight for the ST loss (0.0 means disable ST loss)
dynamic_loss_start_step=1000 # The step to start the dynamic loss weight
dynamic_loss_k=0.25          # The k for the dynamic loss weight (the log base)
use_asr_prompt_decode=false  # Whether to mask the ASR hypothesis at PMTL inference time
promptless_decode=false      # Whether to perform promptless ST inference
use_asr_prompt_dev=false     # Whether to mask the ASR hypothesis at PMTL dev time
disable_asr_inference=false  # Whether to disable ASR inference at inference time, note this only works when use_asr_prompt_decode is false
use_gpu_inference=true       # Whether to use GPU for inference
num_beams=2                  # Number of beams for decoding
inference_checkpoint=        # Checkpoint to use for inference
no_glm=false                 # Whether to skip the GLM evaluation
score_dir_base=scores        # Base directory for storing the evaluation scores
score_backend=hf_dataset     # hf_dataset, legacy_covost2
auto_make_keyfiles=true      # Auto-create wav.scp keyfiles from HF datasets when missing

# Legacy compatibility options kept so older wrappers do not break.
num_nodes=1
nj=32
gpu_inference=false
dumpdir=dump
python=python3

# Speed perturbation related
speed_perturb_factors= # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# ST model related
st_tag= # Suffix to the result dir for st model training.
st_exp= # Specify the directory path for ST experiment.
# If this option is specified, st_tag is ignored.
# Note that it will overwrite args in st config.
src_lang=es # source language abbrev. id (e.g., es)
tgt_lang=en # target language abbrev. id (e.g., en)

# [Task dependent] Set the datadir name created by local/data.sh
train_set=    # Name of training set.
mt_train_set= # Name of MT training set.
valid_set=    # Name of validation set used for monitoring/tuning network training.
test_sets=    # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --expdir         # Directory to save experiments (default="${expdir}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # ST model related
    --st_tag           # Suffix to the result dir for st model training (default="${st_tag}").
    --st_exp           # Specify the directory path for ST experiment.
                       # If this option is specified, st_tag is ignored (default="${st_exp}").
    --src_lang=        # source language abbrev. id (e.g., es). (default="${src_lang}")
    --tgt_lang=        # target language abbrev. id (e.g., en). (default="${tgt_lang}")
    
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

append_opt() {
    local -n target_ref=$1
    local flag=$2
    local value=${3:-}
    if [ -n "${value}" ]; then
        target_ref+=" ${flag} ${value} "
    fi
}

append_flag() {
    local -n target_ref=$1
    local flag=$2
    local enabled=$3
    if "${enabled}"; then
        target_ref+=" ${flag} "
    fi
}

job_log_path() {
    local log_path=$1
    echo "${PWD}/${log_path}"
}

ensure_decode_keyfile() {
    local dset=$1
    local key_file="${hf_datadir}/${dset}.wav.scp"
    local hf_dataset="${hf_datadir}/${src_lang}.${dset}"

    if [ -f "${key_file}" ]; then
        echo "${key_file}"
        return 0
    fi

    if ! "${auto_make_keyfiles}"; then
        log "Error: Missing keyfile ${key_file} and --auto_make_keyfiles is false"
        exit 2
    fi

    if [ ! -d "${hf_dataset}" ]; then
        log "Error: Missing HF dataset ${hf_dataset}, cannot create ${key_file}"
        exit 2
    fi

    log "Creating missing keyfile ${key_file} from ${hf_dataset}"
    ${python_hf} pyscripts/utils/concat_hf_datasets.py \
        --inputs "${hf_dataset}" \
        --wav-scp "${key_file}"
    echo "${key_file}"
}

run_generic_hf_scoring() {
    local task=$1
    local dset=$2
    local hyp_file=$3
    local score_dir=$4
    local opts=

    local hf_dataset="${hf_datadir}/${src_lang}.${dset}"
    if [ ! -d "${hf_dataset}" ]; then
        log "Error: Missing HF dataset ${hf_dataset} for scoring"
        exit 2
    fi

    if "${normalize_text}"; then
        opts+=" --normalize-text "
    fi

    ${python_hf} pyscripts/utils/score_hf_predictions.py \
        --dataset "${hf_dataset}" \
        --hyp-file "${hyp_file}" \
        --task "${task}" \
        --score-dir "${score_dir}" ${opts}
}

# Check required arguments
[ -z "${train_set}" ] && {
    log "${help_message}"
    log "Error: --train_set is required"
    exit 2
}
[ -z "${valid_set}" ] && {
    log "${help_message}"
    log "Error: --valid_set is required"
    exit 2
}
[ -z "${test_sets}" ] && {
    log "${help_message}"
    log "Error: --test_sets is required"
    exit 2
}

# Experiment identifiers
if [ -z "${st_exp}" ]; then
    if [ "${framework}" = "huggingface" ]; then
        st_exp="${expdir}/hf_${st_tag}"
    else
        st_exp="${expdir}/${st_tag}"
    fi
fi

_feat_type=feats
if "${on_the_fly_feat}"; then
    _feat_type=raw
fi
feature_root="${hf_datadir}/features/${_feat_type}"
train_tag="${peft_method}_${batch_mask_prob}_${token_mask_prob}"

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Create the MT data and ASR/ST data separately and concatenate them."

    _dir="${st_exp}/${src_lang}/${train_set}/mml/${train_tag}"
    _logdir="${_dir}/logdir"
    mkdir -p "${_logdir}"
    train_tool="pyscripts/utils/hf_whisper_ft.py"
    mt_train_feat_dir="${feature_root}/${src_lang}.${mt_train_set}.mt"
    mt_valid_feat_dir="${feature_root}/${src_lang}.${valid_set}.mt"
    pmtl_train_feat_dir="${feature_root}/${src_lang}.${train_set}.pmtl"
    pmtl_valid_feat_dir="${feature_root}/${src_lang}.${valid_set}.pmtl"
    combined_mml_dir="${feature_root}/${src_lang}.${train_set}.${mt_train_set}.mml"
    train_mml_dir="${feature_root}/${src_lang}.${train_set}.mml"
    valid_mml_dir="${feature_root}/${src_lang}.${valid_set}.mml"
    # Step 1: Create the MT dataset from the original data
    # If the feature is already extracted in previous runs, skip this step
    if [ ! -d "${mt_train_feat_dir}" ] || [ ! -d "${mt_valid_feat_dir}" ]; then

        opts=" --mode mt "
        append_opt opts --hf_datadir "${hf_datadir}"
        if "${debug}"; then
            append_opt opts --preprocessing_num_proc 1
        else
            append_opt opts --preprocessing_num_proc "${preprocessing_num_proc}"
        fi
        append_opt opts --dev-name "${valid_set}"
        append_opt opts --save_feature_dir "${feature_root}"
        if "${debug}"; then
            ${python_hf} ${train_tool} \
                --feat-extraction \
                --train-set ${mt_train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        else
            # Submit the feature extraction jobs
            JOBID=$(date +'%Y%m%d%H%M%S')
            log "Submitting MT feature extraction... log: '$(job_log_path "${_logdir}/fe_${JOBID}.log")'"
            ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06' --mem 64G --gpu 1 "${_logdir}"/fe_${JOBID}.log \
                ${python_hf} ${train_tool} \
                --feat-extraction \
                --train-set ${mt_train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        fi
    else
        log "Skipping MT feature extraction: found ${mt_train_feat_dir} and ${mt_valid_feat_dir}"
    fi
    # Step 2: Create the ASR/ST dataset
    if [ ! -d "${pmtl_train_feat_dir}" ] || [ ! -d "${pmtl_valid_feat_dir}" ]; then

        opts=" --mode pmtl "
        append_opt opts --hf_datadir "${hf_datadir}"
        if "${debug}"; then
            append_opt opts --preprocessing_num_proc 1
        else
            append_opt opts --preprocessing_num_proc "${preprocessing_num_proc}"
        fi
        append_flag opts --on-the-fly-feat-extraction "${on_the_fly_feat}"
        append_opt opts --dev-name "${valid_set}"
        append_opt opts --save_feature_dir "${feature_root}"
        append_opt opts --speed-perturb-factors "${speed_perturb_factors}"
        if "${debug}"; then
            ${python_hf} ${train_tool} \
                --feat-extraction \
                --train-set ${train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        else
            # Submit the feature extraction jobs
            JOBID=$(date +'%Y%m%d%H%M%S')
            log "Submitting PMTL feature extraction... log: '$(job_log_path "${_logdir}/fe_${JOBID}.log")'"
            ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06' --mem 64G --gpu 1 "${_logdir}"/fe_${JOBID}.log \
                ${python_hf} ${train_tool} \
                --feat-extraction \
                --train-set ${train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        fi
    else
        log "Skipping PMTL feature extraction: found ${pmtl_train_feat_dir} and ${pmtl_valid_feat_dir}"
    fi

    # Step 3: Concatenate the MT and ASR/ST data
    if [ -d "${combined_mml_dir}" ]; then
        log "Skipping MML concatenation: found ${combined_mml_dir}"
    else
        log "Creating concatenated MML features at ${combined_mml_dir}"
        # ${cuda_cmd} JOB=1:1 "${_logdir}"/concatenate_features.log \
        ${python_hf} pyscripts/utils/concatenate_features.py \
            --dset1 "${pmtl_train_feat_dir}" \
            --dset2 "${mt_train_feat_dir}" \
            --output "${combined_mml_dir}"
    fi

    ln -sfnv "${pmtl_valid_feat_dir}" "${valid_mml_dir}"
    ln -sfnv "${combined_mml_dir}" "${train_mml_dir}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Run the multi-modal finetuning on the training data"
    _dir="${st_exp}/${src_lang}/${train_set}/mml/${train_tag}"
    _logdir="${_dir}/logdir"
    mkdir -p "${_logdir}"

    opts=" --mode mml "
    if [ "${framework}" == "huggingface" ]; then
        append_opt opts --hf_datadir "${hf_datadir}"
        if "${debug}"; then
            append_opt opts --preprocessing_num_proc 1
        else
            append_opt opts --preprocessing_num_proc "${preprocessing_num_proc}"
        fi
        append_opt opts --dev-name "${valid_set}"
        append_opt opts --config "${mtl_config}"
        if [ "${peft_method}" != none ]; then
            append_opt opts --peft_method "${peft_method}"
        fi
        append_flag opts --on-the-fly-feat-extraction "${on_the_fly_feat}"
        append_flag opts --normalize_text "${normalize_text}"
        append_opt opts --save_feature_dir "${feature_root}"

        train_tool="pyscripts/utils/hf_whisper_ft.py"
    else
        log "Error: not supported --framework ${framework}"
        exit 2
    fi
    append_opt opts --resume_from_checkpoint "${resume_from_checkpoint}"
    append_opt opts --load_model_from_path "${load_model_from_path}"
    append_opt opts --deepspeed "${ds_config}"
    append_opt opts --save-eval-preds "${st_save_eval_preds}"
    append_flag opts --use-asr-prompt "${use_asr_prompt}"
    append_flag opts --use-asr-prompt-dev "${use_asr_prompt_dev}"
    append_opt opts --min-promptless-prob "${min_promptless_prob}"
    append_opt opts --max-promptless-prob "${max_promptless_prob}"
    append_opt opts --batch-mask-prob "${batch_mask_prob}"
    append_opt opts --token-mask-prob "${token_mask_prob}"
    append_opt opts --min-alpha "${min_alpha}"
    append_opt opts --max-alpha "${max_alpha}"
    append_opt opts --loss-warmup "${dynamic_loss_start_step}"
    append_opt opts --loss-base "${dynamic_loss_k}"

    if "${fe_only}"; then
        log "Skip training as --fe_only is set to true"
    else
        # Submit the training jobs
        JOBID=$(date +'%Y%m%d%H%M%S')
        log "Submitting training... log: '$(job_log_path "${_logdir}/finetune_${JOBID}.log")'"

        if "${debug}"; then
            ${python_hf} ${train_tool} \
                --train-set ${train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        else
            # For some reason the node r9n01 is much faster than the other nodes
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            # ${cuda_cmd} --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
            # ${cuda_cmd} --hostname 'r9n03' --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
            ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r7n01' --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
                ${python_hf} -m torch.distributed.launch --nproc_per_node ${ngpu} --master_port ${master_port} \
                ${train_tool} \
                --train-set ${train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Run (distributed) inference on the dev/test data."
    decode_suf="_org"
    train_suf="/org"
    for dset in ${test_sets}; do
        _logdir="${st_exp}/logdir/inference_mml/${src_lang}/${train_set}/${dset}/${peft_method}${train_suf}${decode_suf}"
        mkdir -p "${_logdir}"

        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/${train_tag}${train_suf}${decode_suf}"
        _modeldir="${st_exp}/${src_lang}/${train_set}/mml/${train_tag}"
        if [ -n "${inference_checkpoint}" ]; then
            _modeldir="${_modeldir}/${inference_checkpoint}"
            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
        fi
        if "${promptless_decode}"; then
            _dir="${_dir}_promptless"
        elif "${use_asr_prompt_decode}"; then
            _dir="${_dir}_asr_prompt"
        fi

        key_file=$(ensure_decode_keyfile "${dset}")
        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Submitting MML inference for ${dset}... log: '$(job_log_path "${_logdir}/decode.*.log")'"

        opts=
        _hf_dset="${hf_datadir}/${src_lang}.${dset}"
        opts+=" --dset ${_hf_dset} "
        opts+=" --num-beams ${num_beams} "

        if [ "${peft_method}" != none ]; then
            opts+=" --peft-model ${_modeldir} "
        fi

        if ! "${promptless_decode}" && "${use_asr_prompt_decode}"; then
            opts+=" --use-asr-hyp "
        fi

        if ! "${promptless_decode}" || ! "${use_asr_prompt_decode}" || "${disable_asr_inference}"; then
            opts+=" --disable-asr "
        fi

        if "${promptless_decode}"; then
            inference_tool="pyscripts/utils/hf_whisper_inference.py"
            opts+=" --task translate "
        else
            inference_tool="pyscripts/utils/hf_whisper_inference_pmtl.py"
        fi

        if "${use_gpu_inference}"; then
            _cmd=${cuda_cmd}
            _gpu=1
        else
            log "Using CPU for inference..."
            _cmd=${decode_cmd}
            _gpu=0
        fi

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.1 \
                --pretrained-model ${_modeldir} \
                --batch-size ${inference_batch_size} \
                --model_name ${model_name} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r8n06\&!r9n02\&!r7n01' --mem 16G --gpu ${_gpu} JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.JOB \
                --pretrained-model ${_modeldir} \
                --batch-size ${inference_batch_size} \
                --model_name ${model_name} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        if ! "${promptless_decode}"; then
            if "${use_asr_prompt_decode}" && ! "${disable_asr_inference}"; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/asr"
                done | LC_ALL=C sort -k1 >"${_dir}/asr"
            fi
        fi
        if "${promptless_decode}"; then
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/text"
            done | LC_ALL=C sort -k1 >"${_dir}/st"
        else
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/st"
            done | LC_ALL=C sort -k1 >"${_dir}/st"
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Run evaluation on the MTL decoded data."

    decode_suf="_org"
    train_suf="/org"

    if ! "${promptless_decode}"; then
        if "${use_asr_prompt_decode}" && ! "${disable_asr_inference}"; then
            for dset in ${test_sets}; do
                # for dset in ${valid_set}; do
                # for dset in ${valid_set} ${test_sets}; do
                log "Running ASR evaluation on ${dset}"
                _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/${train_tag}${train_suf}${decode_suf}"
                if [ -n "${inference_checkpoint}" ]; then
                    _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
                fi
                if "${use_asr_prompt_decode}"; then
                    _dir="${_dir}_asr_prompt"
                fi
                _asr_hyp="${PWD}/${_dir}/asr"

                score_dir=${score_dir_base}/mml/asr/hf_whisper_${model_name}/${src_lang}/${train_tag}/${train_set}${train_suf}${decode_suf}/${dset}
                if "${promptless_decode}"; then
                    score_dir="${score_dir}_promptless"
                elif "${use_asr_prompt_decode}"; then
                    score_dir="${score_dir}_asr_prompt"
                fi

                if [ "${score_backend}" = "legacy_covost2" ]; then
                    eval_script=run-asr-eval-covost2.sh
                    _dset=$(echo "${dset}" | sed 's/_test$//')

                    opts=
                    if [ "${src_lang}" == "ara" ]; then
                        opts+=" --arabic true "
                    fi
                    opts+=" --cer ${eval_cer} "

                    cd evaluation
                    ${eval_script} \
                        --src_lang ${src_lang} \
                        --hyp_asr "${_asr_hyp}" \
                        --sclite ${sclite_path} \
                        --dset "${_dset}" \
                        --score_dir "${score_dir}" \
                        --data_base_dir "${hf_datadir}" \
                        --no_glm "${no_glm}" ${opts}
                    cd -
                else
                    run_generic_hf_scoring asr "${dset}" "${_asr_hyp}" "${score_dir}"
                fi
            done
        fi
    fi

    # Note that we assume the evaluation code is available in the path
    for dset in ${test_sets}; do
        log "Running ST evaluation on ${dset}"
        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/${train_tag}${train_suf}${decode_suf}"
        if [ -n "${inference_checkpoint}" ]; then
            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
        fi
        if "${promptless_decode}"; then
            _dir="${_dir}_promptless"
        elif "${use_asr_prompt_decode}"; then
            _dir="${_dir}_asr_prompt"
        fi
        _st_hyp="${PWD}/${_dir}/st"

        score_dir=${score_dir_base}/mml/st/hf_whisper_${model_name}/${src_lang}/${train_tag}/${train_set}${train_suf}${decode_suf}/${dset}
        if "${promptless_decode}"; then
            score_dir="${score_dir}_promptless"
        elif "${use_asr_prompt_decode}"; then
            score_dir="${score_dir}_asr_prompt"
        fi

        if [ "${score_backend}" = "legacy_covost2" ]; then
            eval_script=run-testset-eval-covost2.sh
            _dset=$(echo "${dset}" | sed 's/_test$//')

            opts=
            if [ "${src_lang}" == "ara" ]; then
                opts+=" --arabic true "
            fi

            cd evaluation
            ${eval_script} \
                --src_lang ${src_lang} \
                --hyp_mt "${_st_hyp}" \
                --dset "${_dset}" \
                --score_dir "${score_dir}" \
                --data_base_dir "${hf_datadir}" \
                --no_glm "${no_glm}" ${opts}
            cd -
        else
            run_generic_hf_scoring st "${dset}" "${_st_hyp}" "${score_dir}"
        fi
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Run (distributed) MT inference on the dev/test data."
    decode_suf="_org"
    train_suf="/org"

    for dset in ${test_sets}; do
        _logdir="${st_exp}/logdir/inference_mml/mt/${src_lang}/${train_set}/${dset}/${peft_method}${train_suf}${decode_suf}"
        mkdir -p "${_logdir}"

        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/mt/${train_tag}${train_suf}${decode_suf}"
        _modeldir="${st_exp}/${src_lang}/${train_set}/mml/${train_tag}"

        if [ -n "${inference_checkpoint}" ]; then
            _modeldir="${_modeldir}/${inference_checkpoint}"
            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/mt/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
        fi

        key_file=$(ensure_decode_keyfile "${dset}")
        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Submitting MT inference for ${dset}... log: '$(job_log_path "${_logdir}/decode.*.log")'"

        opts=
        _hf_dset="${hf_datadir}/${src_lang}.${dset}"
        opts+=" --dset ${_hf_dset} "
        opts+=" --num-beams ${num_beams} "

        if [ "${peft_method}" != none ]; then
            opts+=" --peft-model ${_modeldir} "
        fi

        opts+=" --disable-asr "
        opts+=" --mt "

        inference_tool="pyscripts/utils/hf_whisper_inference_pmtl.py"

        if "${use_gpu_inference}"; then
            _cmd=${cuda_cmd}
            _gpu=1
        else
            log "Using CPU for inference..."
            _cmd=${decode_cmd}
            _gpu=0
        fi

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.1 \
                --pretrained-model ${_modeldir} \
                --batch-size ${inference_batch_size} \
                --model_name ${model_name} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r8n06\&!r9n02\&!r7n01' --mem 16G --gpu ${_gpu} JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.JOB \
                --pretrained-model ${_modeldir} \
                --batch-size ${inference_batch_size} \
                --model_name ${model_name} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/st"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Run evaluation on the MT decoded data."

    decode_suf="_org"
    train_suf="/org"

    # Note that we assume the evaluation code is available in the path
    for dset in ${test_sets}; do
        log "Running MT evaluation on ${dset}"
        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/mt/${train_tag}${train_suf}${decode_suf}"
        if [ -n "${inference_checkpoint}" ]; then
            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/mt/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
        fi
        _st_hyp="${PWD}/${_dir}/text"

        score_dir=${score_dir_base}/mml/mt/hf_whisper_${model_name}/${src_lang}/${train_tag}/${train_set}${train_suf}${decode_suf}/${dset}

        if [ "${score_backend}" = "legacy_covost2" ]; then
            eval_script=run-testset-eval-covost2.sh
            _dset=$(echo "${dset}" | sed 's/_test$//')

            opts=
            if [ "${src_lang}" == "ara" ]; then
                opts+=" --arabic true "
            fi
            if "${no_glm}"; then
                opts+=" --no-glm true "
            fi

            cd evaluation
            ${eval_script} \
                --src_lang ${src_lang} \
                --hyp_mt "${_st_hyp}" \
                --dset "${_dset}" \
                --data_base_dir "${hf_datadir}" \
                --score_dir "${score_dir}" ${opts}
            cd -
        else
            run_generic_hf_scoring mt "${dset}" "${_st_hyp}" "${score_dir}"
        fi
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Run (distributed) ST inference on the dev/test data."
    decode_suf="_org"
    train_suf="/org"

    for dset in ${test_sets}; do
        _logdir="${st_exp}/logdir/inference_mml/st/${src_lang}/${train_set}/${dset}/${peft_method}${train_suf}${decode_suf}"
        mkdir -p "${_logdir}"

        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/st/${train_tag}${train_suf}${decode_suf}"
        _modeldir="${st_exp}/${src_lang}/${train_set}/mml/${train_tag}"

        if [ -n "${inference_checkpoint}" ]; then
            _modeldir="${_modeldir}/${inference_checkpoint}"
            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/st/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
        fi

        key_file=$(ensure_decode_keyfile "${dset}")
        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Submitting ST inference for ${dset}... log: '$(job_log_path "${_logdir}/decode.*.log")'"

        opts=
        if [ "${framework}" == "huggingface" ]; then
            _hf_dset="${hf_datadir}/${src_lang}.${dset}"
            opts+=" --dset ${_hf_dset} "
            opts+=" --num-beams ${num_beams} "

            if [ "${peft_method}" != none ]; then
                opts+=" --peft-model ${_modeldir} "
            fi

            inference_tool="pyscripts/utils/hf_whisper_inference.py"
        else
            inference_tool="pyscripts/utils/whisper_inference.py"
        fi

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.1 \
                --pretrained-model ${_modeldir} \
                --batch-size ${inference_batch_size} \
                --task "translate" \
                --model_name ${model_name} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r7n01' --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.JOB \
                --pretrained-model ${_modeldir} \
                --batch-size ${inference_batch_size} \
                --task "translate" \
                --model_name ${model_name} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Run evaluation on the ST decoded data."

    decode_suf="_org"
    train_suf="/org"

    # Note that we assume the evaluation code is available in the path
    for dset in ${test_sets}; do
        log "Running ST evaluation on ${dset}"
        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/st/${train_tag}${train_suf}${decode_suf}"
        if [ -n "${inference_checkpoint}" ]; then
            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mml/st/${train_tag}_${inference_checkpoint}${train_suf}${decode_suf}"
        fi
        _st_hyp="${PWD}/${_dir}/text"

        score_dir=${score_dir_base}/mml/e2e_st/hf_whisper_${model_name}/${src_lang}/${train_tag}/${train_set}${train_suf}${decode_suf}/${dset}
        if "${promptless_decode}"; then
            score_dir="${score_dir}_promptless"
        elif "${use_asr_prompt_decode}"; then
            score_dir="${score_dir}_asr_prompt"
        fi

        if [ "${score_backend}" = "legacy_covost2" ]; then
            eval_script=run-testset-eval-covost2.sh
            _dset=$(echo "${dset}" | sed 's/_test$//')

            opts=
            if [ "${src_lang}" == "ara" ]; then
                opts+=" --arabic true "
            fi
            if "${no_glm}"; then
                opts+=" --no-glm true "
            fi

            cd evaluation
            ${eval_script} \
                --src_lang ${src_lang} \
                --hyp_mt "${_st_hyp}" \
                --dset "${_dset}" \
                --data_base_dir "${hf_datadir}" \
                --score_dir "${score_dir}" ${opts}
            cd -
        else
            run_generic_hf_scoring st "${dset}" "${_st_hyp}" "${score_dir}"
        fi
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

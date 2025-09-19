#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# This script is used to run the final mult-task, i.e. ST + ASR + MT + Prompted-ST, experiments

# Change the following according to your experiments
src_lang=de
tgt_lang=eng

train_set=train
mt_train_set=train
train_dev=validation
extra_dev=validation2

debug=false
# debug=true

python_hf=python3
ds_config=conf/tuning/ds2.json # The deepspeed configuration file
peft_method=lora               # none, lora, qlora
normalize_text=false           # Whether or not to normalize the text at training time
master_port=29501              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=4                 # Number of jobs for decoding, note that each job will use a GPU
use_gpu_inference=true         # Whether to use GPU for inference
skip_data_prep=true            # Whether to skip data preparation
skip_training=false            # Whether to skip training
use_asr_prompt=true            # Whether to use the ASR prompt at training time
min_promptless_prob=0.1        # The minimum probability for performing promptless ST finetuning
max_promptless_prob=0.1        # The maximum probability for perforFming promptless ST finetuning
batch_mask_prob=0.8            # The probability for applying masks to the prompt
token_mask_prob=0.4            # The probability for masking tokens in the prompt
min_alpha=0.4                  # The minimum alpha for the multi-task losses, i.e. the weight for the ST loss
max_alpha=0.5                  # The maximum alpha for the multi-task losses, i.e. the weight for the ST loss (0.0 means disable ST loss)
dynamic_loss_start_step=1      # The step to start the dynamic loss weight
dynamic_loss_k=0.25            # The k for the dynamic loss weight (the log base)
use_asr_prompt_dev=false       # Whether to use ASR prompt at dev time

# Evaluation related
use_asr_prompt_decode=true  # Whether to use the ASR hypothesis at inference time
promptless_decode=false     # Whether to perform promptless decoding at inference time
disable_asr_inference=false # Whether to disable ASR inference at inference time, note this only works when use_asr_prompt_decode is false
no_glm=true                 # Whether to use the GLM for evaluation

# Initialization related
load_model_from_path=   # The path to load the model from
resume_from_checkpoint= # The path to resume from a checkpoint

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

opts=
if "${debug}"; then
    model=large-v2 # base, large, large-v2 etc.
    mtl_config=conf/tuning/whisper-debug.yaml
    resume_from_checkpoint=
else
    model=large-v2 # base, large, large-v2 etc.
    mtl_config=conf/tuning/mtl_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    # mtl_config=conf/tuning/whisper-debug.yaml
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
fi

if [ ${model} == "large-v2" ]; then
    inference_batch_size=16
elif [ ${model} == "medium" ]; then
    inference_batch_size=48
elif [ ${model} == "tiny" ]; then
    inference_batch_size=128
fi

# Where to save the output at evaluation time
_lang=${src_lang}

if [ -n "${load_model_from_path}" ]; then
    opts+=" --load_model_from_path ${load_model_from_path} "
fi
if [ -n "${resume_from_checkpoint}" ]; then
    opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
fi
if [ -n "${no_glm}" ]; then
    opts+=" --no_glm ${no_glm} "
fi
opts+=" --debug ${debug} "
opts+=" --use_asr_prompt ${use_asr_prompt}"
opts+=" --min_promptless_prob ${min_promptless_prob} "
opts+=" --max_promptless_prob ${max_promptless_prob} "
opts+=" --batch_mask_prob ${batch_mask_prob} "
opts+=" --token_mask_prob ${token_mask_prob} "
opts+=" --min_alpha ${min_alpha} "
opts+=" --max_alpha ${max_alpha} "
opts+=" --dynamic_loss_start_step ${dynamic_loss_start_step} "
opts+=" --dynamic_loss_k ${dynamic_loss_k} "

declare -A testset_dict

testset_dict+=(
    ["fr"]="test"
    ["de"]="test")

test_set=${testset_dict[${src_lang}]} # This option is to run eval

framework=huggingface # huggingface, openai
preprocessing_num_proc=32
on_the_fly_feat=true

hf_datadir=${HF_DATA_DIR}/${src_lang}
dumpdir=dump/covost2/${src_lang}

if ! "${skip_data_prep}"; then
    local/prep_covost2.py \
        --data-dir ${COVOST2_RAW_DATA_DIR}/${src_lang}/raw \
        --save-dir ${hf_datadir} \
        --lang ${src_lang}

    # If the extra_dev is set, it will be created using the ASR-prompted data
    if [ -n "${extra_dev}" ]; then
        # Run Whisper inference on the validation data to create the ASR-prompted validation2 data
        logdir="${dumpdir}/log/${train_dev}"
        mkdir -p "${logdir}"
        output_dir="${dumpdir}/decode/${train_dev}"
        mkdir -p "${output_dir}"

        key_file=${hf_datadir}/${train_dev}.wav.scp

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        opts=" --dset ${hf_datadir}/${src_lang}.${train_dev} "
        inference_tool="pyscripts/utils/hf_whisper_inference.py"

        . ./path.sh
        . ./cmd.sh

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${logdir}/decode.1.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${src_lang} \
                --output_dir ${logdir}/output.1 \
                --batch-size ${inference_batch_size} \
                --model_name large-v2 \
                --num-beams 1 ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r7n01' --mem 16G --gpu 1 JOB=1:"${_nj}" "${logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${logdir}/decode.JOB.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${src_lang} \
                --output_dir ${logdir}/output.JOB \
                --batch-size ${inference_batch_size} \
                --model_name large-v2 \
                --num-beams 1 ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${output_dir}"
        for i in $(seq "${_nj}"); do
            cat "${logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${output_dir}/text"

        # Create the extra_dev data
        pyscripts/utils/create_synth_data.py \
            --src-dset ${hf_datadir}/${src_lang}.${train_dev} \
            --tgt-dset ${hf_datadir}/${src_lang}.${extra_dev} \
            --asr-hyp ${output_dir}/text
    fi
fi

if [ -n "${extra_dev}" ]; then
    valid_set=${extra_dev}
else
    valid_set=${train_dev}
fi

if ! "${skip_training}"; then
    ./finetune.sh \
        --ngpu 2 \
        --expdir exp/covost2 \
        --nj 80 \
        --mtl_config ${mtl_config} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --speed_perturb_factors "0.9 1.0 1.1" \
        --train_set "${train_set}" \
        --mt_train_set "${mt_train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_set}" \
        --stage 2 \
        --stop_stage 3 \
        --dumpdir "${dumpdir}" \
        --st_tag whisper_${model} \
        --model_name ${model} \
        --use_gpu_inference ${use_gpu_inference} \
        --inference_nj ${inference_nj} \
        --framework ${framework} \
        --hf_datadir ${hf_datadir} \
        --peft_method ${peft_method} \
        --preprocessing_num_proc ${preprocessing_num_proc} \
        --on_the_fly_feat ${on_the_fly_feat} \
        --normalize_text ${normalize_text} \
        --master_port ${master_port} \
        --python_hf ${python_hf} \
        --inference_batch_size ${inference_batch_size} \
        --inference_checkpoint checkpoint-4 \
        --use_asr_prompt_decode ${use_asr_prompt_decode} \
        --promptless_decode ${promptless_decode} \
        --disable_asr_inference ${disable_asr_inference} \
        --use_asr_prompt_dev ${use_asr_prompt_dev} \
        --num_beams 1 \
        --score_dir_base scores/covost2 ${opts}
fi

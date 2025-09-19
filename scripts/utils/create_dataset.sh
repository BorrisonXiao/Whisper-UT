#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

logdir=logs/hf_datasets
python=python3
cmd=utils/run.pl
raw_data_location=/exp/cxiao/scale23/merged_data_base
output_path=/exp/cxiao/scale23/test_hf_data
src_lang=ara
stm=true

. ./path.sh
. utils/parse_options.sh

mkdir -p "${logdir}"

script=pyscripts/utils/create_dataset.py
if "${stm}"; then
    script=pyscripts/utils/create_dataset_stm.py
    logdir+=_stm
fi
${cmd} "JOB=1:1" "${logdir}/hf_datasets.JOB.log" \
    ${python} ${script} \
    --raw-data-location "${raw_data_location}" \
    --output-path "${output_path}" \
    --src-lang "${src_lang}"

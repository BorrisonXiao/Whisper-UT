#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_lang=ara
tgt_lang=eng
scale23_data_root=${SCALE23_DATA_ROOT:-/exp/scale23/data/3-way}
output_root="${WHISPER_UT_DATA_ROOT:-${repo_root}/data}/${src_lang}"
train_sr_stms="${scale23_data_root}/${src_lang}/sr.${src_lang}-${src_lang}.train-cts.stm"
train_st_stms="${scale23_data_root}/${src_lang}/st.${src_lang}-${tgt_lang}.train-cts.stm"
dev_sr_stms="${scale23_data_root}/${src_lang}/sr.${src_lang}-${src_lang}.dev1.stm ${scale23_data_root}/${src_lang}/sr.${src_lang}-${src_lang}.dev2.stm"
dev_st_stms="${scale23_data_root}/${src_lang}/st.${src_lang}-${tgt_lang}.dev1.stm ${scale23_data_root}/${src_lang}/st.${src_lang}-${tgt_lang}.dev2.stm"
test_sr_stms="${scale23_data_root}/${src_lang}/testsets/cts/sr.${src_lang}-${src_lang}.iwslt22.test.stm"
test_st_stms="${scale23_data_root}/${src_lang}/testsets/cts/st.${src_lang}-${tgt_lang}.iwslt22.test.stm"
stage=1
stop_stage=2

exec bash "${script_dir}/data.sh" \
    --src_lang "${src_lang}" \
    --tgt_lang "${tgt_lang}" \
    --output_root "${output_root}" \
    --train_sr_stms "${train_sr_stms}" \
    --train_st_stms "${train_st_stms}" \
    --dev_sr_stms "${dev_sr_stms}" \
    --dev_st_stms "${dev_st_stms}" \
    --test_sr_stms "${test_sr_stms}" \
    --test_st_stms "${test_st_stms}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    "$@"

MAIN_ROOT=$ESPNET_ROOT
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
export HF_DATA_DIR=$WHISPER_UT_DATA_ROOT/hf_data
export COVOST2_RAW_DATA_DIR=$WHISPER_UT_DATA_ROOT/covost2

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

. utils/activate_hf.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PYTHONPATH=$(conda info --base)/envs/hf/bin:$PYTHONPATH
export PATH=$(conda info --base)/envs/hf/bin:$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64

# NOTE(kamo): Source at the last to overwrite the setting
# NOTE(Cihan): No need to install moses for the whisper finetuning task
# . local/path.sh

# slurm-related setup
# ml load cuda12.0/toolkit/12.0.1-1
# ml load gcc/11.3.0
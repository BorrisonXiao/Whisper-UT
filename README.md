# Whisper-UT
## Abstract
We present **​​Whisper-UT**​​, a unified speech-text translation framework that leverages **​​lightweight adapters**​​ and ​​multi-task learning​​ to seamlessly adapt encoder-decoder models like Whisper for **diverse modalities**—speech-only, text-only, and multimodal inputs. By incorporating ​​cross-modal conditioning​​ and a novel ​**​2-stage decoding strategy​**​, our approach enhances translation quality even with imperfect transcripts. Trained efficiently on ​​3-way parallel and text-only data​​, Whisper-UT achieves state-of-the-art performance on ​**​CoVoST2​​, ​​Fisher-Spanish​​, and ​​BBN-Mandarin**​​ benchmarks, with gains of up to **​​+10 BLEU​**​ in multimodal settings and ​**​+8.7 BLEU​**​ in low-resource domains over strong baselines.
## Installation
The steps below will install all the dependencies needed to run the project, though package versions may differ across computing platforms.
1. Create a conda environment with `conda create --name hf python=3.9.16`.
2. Install torch that's compatible with your local CUDA driver, e.g., `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`.
3. Install core dependencies with `pip install transformers==4.31.0 accelerate==0.23.0 peft==0.6.0 deepspeed==0.9.5 "torch==2.1.0+cu121"`.
4. Install `datasets`, e.g., `pip install datasets==2.13.1`.
5. Install other dependencies with `pip install evaluate chinese-converter librosa soundfile jiwer sacrebleu tensorboard==2.11.2`.
6. On the internal SCALE setup, the scripts in this repo are expected to run from the `scale` conda environment.
## Path Configuration
For proper functionality, it is recommended to set the following environment variables in your `.bashrc` file:
- `ESPNET_ROOT`: The installation path of ESPNet. This is required as several utility scripts depend on it, e.g., `export ESPNET_ROOT=/path/to/ESPNet/`.
- `WHISPER_UT_DATA_ROOT`: The directory where you wish to store the downloaded and processed data, e.g., `export WHISPER_UT_DATA_ROOT=/path/to/data/`.
- `SCALE23_DATA_ROOT`: Optional root for the STM-based SCALE23 recipes, e.g., `export SCALE23_DATA_ROOT=/exp/scale23/data/3-way`.
## Dataset Preparation
### CoVoST2
1. Download the Common Voice Corpus 4 https://commonvoice.mozilla.org/en/datasets and unpack it to the `COVOST2_RAW_DATA_DIR/$src_lang/raw` directory, e.g., `COVOST2_RAW_DATA_DIR/de/raw` using the command `tar xvzf de.tar`.
2. Run `covost2_de.sh` with `skip_data_prep=false` to create the Hugging Face datasets under `${HF_DATA_DIR}/de`.

### STM-First Recipe
The repo also provides an ESPNet-free STM-first data prep path for SCALE-style corpora.

- Generic entry point: `local/data.sh`
- Arabic preset wrapper: `local/ara.sh`

The STM-first recipe works in two stages:

1. Stage 1 builds merge manifests from paired `sr` and `st` STM files.
2. Stage 2 materializes merged or original audio, STM files, and Hugging Face datasets.

Training STM files are merged by recording path and timestamps. Dev and test sets are kept with original segmentation. The training output can be split into:

- `train_3way_merged`
- `train_asr_only_merged`
- `train_st_only_merged`

Unmerged splits are tagged with `_org`, for example:

- `dev1_org`
- `dev2_org`
- `iwslt22_test_org`

Example Arabic data prep:

```bash
conda run -n scale bash local/ara.sh --stage 1 --stop_stage 2
```

The Arabic wrapper defaults to:

- `sr.ara-ara.train-cts.stm`
- `st.ara-eng.train-cts.stm`
- `sr.ara-ara.dev1.stm`, `sr.ara-ara.dev2.stm`
- `st.ara-eng.dev1.stm`, `st.ara-eng.dev2.stm`
- `testsets/cts/sr.ara-ara.iwslt22.test.stm`
- `testsets/cts/st.ara-eng.iwslt22.test.stm`

The materialized Hugging Face datasets are saved under `${WHISPER_UT_DATA_ROOT}/ara/hf` by default and are named like `ara.train_3way_merged` or `ara.dev1_org`.
## Fine-Tuning Example (CoVoST2-de)
1. Run data pre-processing with `skip_data_prep=false` and `skip_training=true` in `covost2_de.sh`.
2. Launch the training with `skip_data_prep=true` and `skip_training=false` in `covost2_de.sh`.
3. The CoVoST2 wrapper still uses the legacy CoVoST2 scoring path by passing `--score_backend legacy_covost2` into `finetune.sh`.

## Fine-Tuning Example (Arabic STM Recipe)
Use `ft_ara.sh` to fine-tune on the STM-first Arabic data. The wrapper:

- combines `train_3way_merged`, `train_asr_only_merged`, and `train_st_only_merged` into `train_all_merged`
- combines `train_3way_merged` and `train_st_only_merged` into `train_mt_sup_merged`
- uses `dev1_org` for validation
- uses `dev2_org` and `iwslt22_test_org` for decoding and evaluation
- defaults to LoRA and on-the-fly feature extraction

Example:

```bash
conda run -n scale bash ft_ara.sh --ngpu 4 --stage 0 --stop_stage 7
```

Feature caches created by `finetune.sh` are stored under:

```text
${HF_DATA_DIR_OR_HF_DATADIR}/features/raw
```

for on-the-fly mode, with subdirectories such as:

- `${src_lang}.${mt_train_set}.mt`
- `${src_lang}.${train_set}.pmtl`
- `${src_lang}.${train_set}.${mt_train_set}.mml`

If those feature directories already exist, stage 0 prints explicit skip messages and reuses them.
## Evaluation Example (CoVoST2-de)
1. Stages 2-7 evaluate the fine-tuned model's performance across various tasks: Automatic Speech Recognition (ASR), Machine Translation (MT), end-to-end Speech Translation (ST), and a 2-stage ST approach.
2. The 2-stage ST mode is activated by setting `use_asr_prompt_decode=true`; when set to `false`, the model performs multimodal translation without the ASR prompt.
3. Generic STM-first datasets can be scored directly against the saved Hugging Face datasets through the default `hf_dataset` scoring backend in `finetune.sh`.
## Contact
If you have any questions, please feel free to contact us via `cxiao7@jhu.edu`.

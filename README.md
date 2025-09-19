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
## Path Configuration
For proper functionality, it is recommended to set the following environment variables in your `.bashrc` file:
- `ESPNET_ROOT`: The installation path of ESPNet. This is required as several utility scripts depend on it, e.g., `export ESPNET_ROOT=/path/to/ESPNet/`.
- `WHISPER_UT_DATA_ROOT`: The directory where you wish to store the downloaded and processed data, e.g., `export WHISPER_UT_DATA_ROOT=/path/to/data/`.
## Dataset Preparation
### CoVoST2
1. Download the Common Voice Corpus 4 https://commonvoice.mozilla.org/en/datasets and unpack it to the `COVOST2_RAW_DATA_DIR/$src_lang/raw` directory, e.g., `COVOST2_RAW_DATA_DIR/de/raw` using the command `tar xvzf de.tar`.
## Fine-Tuning Example (CoVoST2-de)
1. Run data pre-processing with `skip_data_prep=false` and `skip_training=true` in `covost2_de.sh`.
2. Launch the training with `skip_data_prep=true` and `skip_training=false` in `covost2_de.sh` and `stage=1`.
## Evaluation Example (CoVoST2-de)
1. Stages 2-7 evaluate the fine-tuned model's performance across various tasks: Automatic Speech Recognition (ASR), Machine Translation (MT), end-to-end Speech Translation (ST), and a 2-Stage ST approach. The 2-Stage ST mode is activated by setting `use_asr_prompt_decode=true`; when set to `false`, the model performs Multimodal Machine Translation (MMT).
## Contact
If you have any questions, please feel free to contact us via `cxiao7@jhu.edu`.
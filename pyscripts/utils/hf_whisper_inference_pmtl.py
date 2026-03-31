#!/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
# Note that the hard-coded path above is specific to the HLT cluster due to ESPNet environment setup.
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

# For debugging ESPNet env issue (not loading the correct python interpreter)
# for dist in __import__('pkg_resources').working_set:
#     print(dist.project_name.replace('Python', ''))
# import sys; print(sys.executable)

from whisper_st.modeling_whisper import WhisperForConditionalGeneration
from whisper_st.feature_extraction_whisper import WhisperFeatureExtractor
from whisper_st.tokenization_whisper import WhisperTokenizer
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_from_disk
from peft import PeftModel, PeftConfig
from functools import partial
import re
from typing import Dict, Union, Any, Mapping
from transformers.feature_extraction_utils import BatchFeature
from whisper_st.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
import copy

LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
    "tus": "tunisian",
    "eng": "english",
    "fr": "french",
    "de": "german",
}


class LocalWhisperProcessor:
    """Lightweight decode-time processor wrapper using local whisper_st classes."""

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, audio, sampling_rate=None, **kwargs):
        return self.feature_extractor(audio, sampling_rate=sampling_rate, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(
            task=task, language=language, no_timestamps=no_timestamps
        )


def _load_local_whisper_processor(model_path):
    """Force local whisper_st tokenizer/feature extractor classes at decode time."""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, use_fast=False)
    return LocalWhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _prepare_input(data: Union[torch.Tensor, Any], device="cpu") -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    return data


def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device="cpu") -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    inputs = _prepare_input(inputs, device=device)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
            f"training dataset contains keys expected by the model."
        )

    return inputs


def _create_prompted_inputs(
    input_features,
    tokenizer,
    prompts=None,
    device="cpu",
    startofprev_token="<|startofprev|>",
) -> Dict[str, torch.Tensor]:
    inputs_st = {}
    inputs_st['input_features'] = input_features

    tokenizer = copy.deepcopy(tokenizer.tokenizer)
    # Enforce left padding for generation
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 128  # Max is 448, left some room for the prompt

    # Add asr reference prompt to the prefix
    # i.e. <startofprev> [asr_ref] [st_ref]
    # [asr_ref] is the reference transcript for ASR without special tokens
    # [st_ref] is the reference transcript for ST with special tokens
    # Normally form the dataset, note that the <|startoftranscript|> token should be added
    # and the decoder_input_ids must be specified explicitly so that it's not shifted to the right
    # with a starting <|startoftranscript|> token
    # e.g. labels: [asr_ref] <|startoftranscript|> [st_ref] <|endoftext|>
    # e.g. decoder_input_ids: <|startofprev|> [asr_ref] <|startoftranscript|> [st_ref]
    # Note that the first <|endoftext|> token should not be ignored, otherwise the model will not
    # learn to terminate the generation.
    print(startofprev_token)
    prompt_texts = [f"{startofprev_token}{prompt}" for prompt in prompts]
    _labels = tokenizer.batch_encode_plus(
        prompt_texts,
        add_special_tokens=False,
        truncation=True,
    ).input_ids
    # Otherwise, the decoder_input_ids are _labels
    decoder_input_ids = [{"input_ids": _label} for _label in _labels]
    decoder_input_ids = tokenizer.pad(decoder_input_ids, return_tensors="pt")
    assert decoder_input_ids["input_ids"].shape[
        1] <= 128, f"decoder_input_ids['input_ids'].shape[1]: {decoder_input_ids['input_ids'].shape[1]}"
    # Note that no right padding is applied to the labels since they are set to -100 already in the collator
    decoder_input_ids = decoder_input_ids["input_ids"]
    inputs_st['decoder_input_ids'] = decoder_input_ids

    # Add decoder_attention_mask to mask out the left padding
    decoder_attention_mask = torch.zeros_like(decoder_input_ids)
    # Replace all non-padding tokens with 1
    decoder_attention_mask = decoder_attention_mask.masked_fill(
        decoder_input_ids.ne(tokenizer.pad_token_id), 1)
    inputs_st['decoder_attention_mask'] = decoder_attention_mask

    inputs_st = BatchFeature(inputs_st)
    inputs_st = _prepare_inputs(inputs_st, device=device)

    return inputs_st


def inference(
    keyfile,
    dset,
    src_lang,
    tgt_lang,
    output_dir,
    model_name,
    pretrained_model=None,
    peft_model=None,
    batch_size=1,
    use_asr_hyp=False,
    disable_asr=False,
    num_beams=2,
    mt=False,
    asr_hyp=None,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    if peft_model is not None:
        print(f"Loading PEFT model from {peft_model}")
        peft_config = PeftConfig.from_pretrained(peft_model)
        print(f"Loading base model from {peft_config.base_model_name_or_path}")
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path).to(device)
        processor = _load_local_whisper_processor(peft_model)
        model = PeftModel.from_pretrained(model, peft_model)
    else:
        if pretrained_model is not None:
            print(f"Loading pretrained model from {pretrained_model}")
            processor = _load_local_whisper_processor(pretrained_model)
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model).to(device)
        else:
            print(
                f"Loading model from huggingface openai/whisper-{model_name}")
            processor = _load_local_whisper_processor(
                f"openai/whisper-{model_name}")
            model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/whisper-{model_name}").to(device)

    asr_forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=src_lang, task="transcribe")
    print(f"model.device: {model.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"batch_size: {batch_size}")
    print(f"num_beams: {num_beams}")

    if use_asr_hyp:
        print("Using ASR hypothesis as the prompt to the ST inference")

    # Load the HF dataset
    ds = load_from_disk(str(dset))
    last_uttid = ds[-1]["uttid"]

    # Load the keyfile
    if keyfile is not None:
        with open(keyfile, "r") as f:
            lines = f.readlines()
        _keys = [line.strip().split(maxsplit=1)[0] for line in lines]
        last_uttid = _keys[-1]
        keys = set(_keys)

    # Load the ASR hypothesis file
    asr_hyps = None
    if asr_hyp is not None:
        print(
            f"Loading ASR hypothesis from {asr_hyp}, instead of the model's ASR outputs")
        with open(asr_hyp, "r") as f:
            lines = f.readlines()
        asr_hyps = [line.strip().split(maxsplit=1) for line in lines]
        asr_hyps = {uttid: hyp for uttid, hyp in asr_hyps}

    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids(
        "<|startoftranscript|>")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_asr = output_dir / "asr"
    output_st = output_dir / "st"
    do_asr = use_asr_hyp or not disable_asr
    if asr_hyp is not None:
        print(
            f"Skipping ASR inference as the ASR hypothesis is provided at {asr_hyp}...")
        do_asr = False
    total_len = len(ds) if keyfile is None else len(keys)
    if not do_asr:
        print(
            f"Skipping ASR inference as use-asr-hyp is {use_asr_hyp}, disable-asr is {disable_asr} and asr-hyp is {asr_hyp}")
    else:
        all_hyps = {}
        with open(output_asr, "w") as f:
            pbar = tqdm(range(total_len))
            batch = []
            uttids = []
            for uttidx, utt in enumerate(ds):
                uttid = utt["uttid"]
                if keyfile is not None and uttid not in keys:
                    continue
                uttids.append(uttid)
                batch.append(utt)
                if len(batch) < batch_size and uttid != last_uttid:
                    continue
                # Process the batch
                input_speech = [utt["audio"]["array"] for utt in batch]
                samping_rate = batch[0]["audio"]["sampling_rate"]
                input_features = processor(
                    input_speech, sampling_rate=samping_rate, return_tensors="pt").input_features.to(device)

                # Generate token ids
                model.generate = partial(
                    model.generate, language=src_lang, task="transcribe")
                predicted_ids = model.generate(
                    input_features, max_length=128, forced_decoder_ids=asr_forced_decoder_ids)
                # Decode token ids to text
                hyps = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True)

                all_hyps.update(
                    {uttid: hyp for uttid, hyp in zip(uttids, hyps)})

                for i, hyp in enumerate(hyps):
                    print(uttids[i], hyp.strip(), file=f)

                f.flush()
                pbar.update(len(batch))

                # Reset the batch
                batch = []
                uttids = []

        if asr_hyps is None and not mt:
            print("Using on-the-fly ASR hypothesis as the prompt to the ST inference")
            asr_hyps = all_hyps

    with open(output_st, "w") as f_st:
        # Do the conditioned ST inference
        pbar = tqdm(range(total_len))
        batch = []
        uttids = []
        for uttidx, utt in enumerate(ds):
            uttid = utt["uttid"]
            if keyfile is not None and uttid not in keys:
                continue
            if asr_hyps is not None and not mt:
                # Replace the transcript with the ASR hypothesis
                utt["transcript"] = asr_hyps[uttid]
            uttids.append(uttid)
            batch.append(utt)
            if len(batch) < batch_size and uttid != last_uttid:
                continue
            # Process the batch
            input_speech = [utt["audio"]["array"] for utt in batch]
            samping_rate = batch[0]["audio"]["sampling_rate"]
            if not mt:
                input_features = processor(
                    input_speech, sampling_rate=samping_rate, return_tensors="pt").input_features.to(device)
            else:
                _input_features = [{"input_features": torch.zeros(
                    (processor.feature_extractor.feature_size, processor.feature_extractor.nb_max_frames)).numpy()} for _ in input_speech]
                input_features = processor.feature_extractor.pad(
                    _input_features, return_tensors="pt")['input_features']

            st_predicted_ids = []
            model.generate = partial(
                model.generate, language=src_lang, task="translate")
            transcripts = [_["transcript"] for _ in batch]
            # Normalize the reference text, remove things like "[xxxx]", i.e. things in square brackets
            for i, transcript in enumerate(transcripts):
                _ref = re.sub(r"\s+\[.*?\]\s+", "", transcript)
                # Remove the "% " tokens
                _ref = re.sub(r"% ", "", _ref)
                # Remove the "<char> - " tokens where <char> refers to any English character
                _ref = re.sub(r"[a-zA-Z] -\s+", "", _ref)
                _ref = re.sub(r" - ", "", _ref)
                # Remove the extra spaces
                _ref = re.sub(r"\s+", " ", _ref).strip()
                transcripts[i] = _ref

            # Need to explicitly erase the forced_decoder_ids
            model.config.forced_decoder_ids = None
            model.generation_config.forced_decoder_ids = None
            # Set the decoder_start_token_ids to a set of ids, i.e. <|startoftranscript|><|language|><|task|><|notimestamps|>
            # Get the language token id
            model.generation_config.language = src_lang
            if model.generation_config.language in model.generation_config.lang_to_id.keys():
                language_token = model.generation_config.language
            elif model.generation_config.language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[model.generation_config.language]}|>"
            elif model.generation_config.language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{model.generation_config.language}|>"
            else:
                is_language_code = len(model.generation_config.language) == 2
                raise ValueError(
                    f"Unsupported language: {model.generation_config.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )
            language_token_id = model.generation_config.lang_to_id[language_token]

            model.generation_config.task = "translate"
            if model.generation_config.task in TASK_IDS:
                task_token_id = model.generation_config.task_to_id[model.generation_config.task]
            else:
                raise ValueError(
                    f"The `{model.generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                )

            decoder_start_token_ids = [
                startoftranscript_id, language_token_id, task_token_id]

            if hasattr(model.generation_config, "no_timestamps_token_id") and not model.generation_config.return_timestamps:
                decoder_start_token_ids.append(
                    model.generation_config.no_timestamps_token_id)

            model.generation_config.decoder_start_token_id = decoder_start_token_ids
            inputs_st = _create_prompted_inputs(
                input_features=input_features,
                tokenizer=processor,
                prompts=transcripts,
                device=device,
                startofprev_token="<|startofprev|>" if not use_asr_hyp else "<|startoflm|>",
            ).to(device=device)
            raw_st_predicted_ids = model.generate(
                **inputs_st, max_length=256, num_beams=1)  # Currently only supports greedy decoding
            # Need to mask everything before the first <|startoftranscript|> token with the special token
            st_predicted_ids = copy.deepcopy(raw_st_predicted_ids)
            for i, st_predicted_id in enumerate(raw_st_predicted_ids):
                # Mask everything before the first <|startoftranscript|> token
                first_startoftranscript_id = (
                    st_predicted_id == startoftranscript_id).nonzero(as_tuple=True)[0]
                if len(first_startoftranscript_id) > 0:
                    st_predicted_id[:first_startoftranscript_id] = processor.tokenizer.pad_token_id
                st_predicted_ids[i] = st_predicted_id
            st_hyps = processor.tokenizer.batch_decode(
                st_predicted_ids, skip_special_tokens=True)
            for i, hyp in enumerate(st_hyps):
                print(uttids[i], hyp.strip(), file=f_st)
            f_st.flush()
            pbar.update(len(batch))

            # Reset the batch
            batch = []
            uttids = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=Path,
                        default="/home/hltcoe/cxiao/scale23/st/dump/export/cmn.bbn_cts_bolt_test.wav.scp",
                        help="Path to the keyfile")
    parser.add_argument("--dset", type=Path,
                        default="/expscratch/dchakraborty/hf_datasets/scale23/data/multi/cmn.bbn_cts_bolt_test",
                        help="Path to the HF dataset")
    parser.add_argument("--src-lang", type=str, default="cmn",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path,
                        default="exp/st_hf_whisper_tiny/logdir/inference_asr/cmn/bbn_cts_bolt_test/output.1",
                        help="Path to the output directory")
    parser.add_argument("--pretrained-model", type=Path, default=None,
                        help="Path to the pretrained (finetuned) model, if not specified, the model will be loaded from HuggingFace")
    parser.add_argument("--peft-model", type=Path, default=None,
                        help="Path to the PEFT model, note that this will override the pretrained model")
    parser.add_argument("--disable-asr", action="store_true",
                        help="Disable ASR inference, note that this only applies if use-asr-hyp is False")
    parser.add_argument("--model_name", type=str, default="tiny")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-asr-hyp", action="store_true",
                        help="Use the ASR hypothesis as the prompt to the ST inference")
    parser.add_argument("--num-beams", type=int, default=2,
                        help="Number of beams for the inference")
    parser.add_argument("--mt", action="store_true",
                        help="Use the MT mode, i.e. replace input_features with zeros")
    parser.add_argument("--asr-hyp", type=Path, default=None,
                        help="Path to the ASR hypothesis file")
    args = parser.parse_args()
    inference(
        keyfile=args.keyfile,
        dset=args.dset,
        src_lang=LANGS[args.src_lang],
        tgt_lang=LANGS[args.tgt_lang],
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained_model=args.pretrained_model,
        peft_model=args.peft_model,
        batch_size=args.batch_size,
        use_asr_hyp=args.use_asr_hyp,
        disable_asr=args.disable_asr,
        num_beams=args.num_beams,
        mt=args.mt,
        asr_hyp=args.asr_hyp,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import yaml

DEFAULT = {
    "Seq2SeqTrainingArguments": dict(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=1,
        warmup_steps=800,
        max_steps=2400,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=30,
        report_to=["tensorboard"],
        evaluation_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,  # This is important for PEFT
    ),
    "LoraConfig": dict(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",
    ),
    "WhisperConfig": dict(
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.05,
        mask_feature_length=10,
        mask_feature_min_masks=0,
    ),
    "QuantizationConfig": dict(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype="bf16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )}


class Config(object):
    """
    This code is adapted from:
    https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """

    def __init__(self, config_path):
        with open(config_path) as cf_file:
            self._data = yaml.safe_load(cf_file.read())

    def get(self, path=None, default=DEFAULT):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

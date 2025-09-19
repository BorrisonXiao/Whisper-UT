#!/usr/bin/env python3
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import Seq2SeqTrainer
import copy
from typing import Dict, List, Optional
import torch
from functools import partial
from transformers.utils import (
    is_torch_tpu_available,
)
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time
import math
from transformers.trainer_utils import (
    EvalPrediction,
    speed_metrics,
    has_length,
    denumpify_detensorize,
    EvalLoopOutput,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init
from transformers.debug_utils import DebugOption
from torch import nn
from typing import Any, Dict, List, Optional, Union, Tuple
from packaging import version
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.feature_extraction_utils import BatchFeature
from whisper_st.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

logger = logging.get_logger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(
        SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
    "fr": "french",
    "de": "german",
}


def _schedule_dynamic_mtl_weight(
    current_step,
    warmup_steps,
    max_steps,
    min_weight,
    max_weight,
    loss_base,
):
    if min_weight == max_weight:
        return min_weight

    c = warmup_steps
    d = max_steps
    a = min_weight
    b = max_weight
    k = loss_base

    x = np.random.beta(min_weight, max_weight)
    return x


class WhisperTrainer(Seq2SeqTrainer):
    """
    Trainer for Whisper.
    The Seq2SeqTrainer is sufficient to perform monolingual (multitask) training.
    However, in order to extend to multilingual training, we need to override the evaluate()
    method so that the prompts are configured correctly, i.e. with the correct task token and the
    correct language token.
    """

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if self.use_asr_prompt:
            inputs_asr = {}
            inputs_asr['input_features'] = inputs['input_features']
            inputs_asr['labels'] = inputs['labels_src']
            inputs_asr = BatchFeature(inputs_asr)
            inputs_asr = self._prepare_inputs(inputs_asr)

            promptless_train = True

            promptless_prob = .0
            if self.max_promptless_prob > 0.0:
                current_step = self.state.global_step + 1
                max_steps = self.args.max_steps
                # Linearly increase the promptless_prob from min_promptless_prob to max_promptless_prob
                # as the training step increases to max_steps
                promptless_prob = self.min_promptless_prob + \
                    (self.max_promptless_prob - self.min_promptless_prob) * \
                    (current_step - 1) / max_steps

                if np.random.rand() < promptless_prob:
                    promptless_train = True
                else:
                    promptless_train = False
            else:
                promptless_train = False

            # This version of the script supports only masked prompt training. Sampling is no longer supported.
            # The ASR reference is used as the ST prompt
            inputs_st = {}
            inputs_st['input_features'] = inputs['input_features']

            startofprev_token = "<|startofprev|>"
            sot_token = "<|startoftranscript|>"
            sot_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                sot_token)

            if promptless_train:
                inputs_st['labels'] = inputs['labels_tgt']
            else:
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
                max_tgt_len = inputs['labels_tgt'].shape[1]
                # Truncate the ASR transcript so that the new supervision's does not exceed the max length
                _asr_tokens = inputs['labels_src']
                _asr_tokens = _asr_tokens[:, :440 - max_tgt_len]

                asr_texts = self.tokenizer.tokenizer.batch_decode(
                    _asr_tokens, skip_special_tokens=True)

                st_ref_texts = self.tokenizer.tokenizer.batch_decode(
                    inputs['labels_tgt'], skip_special_tokens=False)
                st_ref_texts = [
                    f"<|startoftranscript|>{_text}" for _text in st_ref_texts]
                prompt_texts = [f"{startofprev_token}{asr_text}{st_ref_text}" for asr_text, st_ref_text in zip(
                    asr_texts, st_ref_texts)]
                _labels = self.tokenizer.tokenizer.batch_encode_plus(
                    prompt_texts, add_special_tokens=False).input_ids
                # labels are _labels[1:]
                labels = [{"input_ids": _label[1:]} for _label in _labels]
                # decoder_input_ids are _labels[:-1]
                decoder_input_ids = [{"input_ids": _label[:-1]}
                                     for _label in _labels]

                # Here we apply the mask to the ASR transcript
                similarity_sample_range = 4
                if np.random.rand() < self.batch_mask_prob:
                    mt_idx = [idx for idx, input_feature in enumerate(
                        inputs_st['input_features']) if (input_feature == 0).all().item()]
                    for i, _tokens in enumerate(decoder_input_ids):
                        # Do not apply masks to the MT-only samples
                        if i in mt_idx:
                            continue
                        sop_pos = 1  # The first token is always <|startofprev|>
                        # The end-of-prev token is the sot_token
                        eop_pos = _tokens['input_ids'].index(sot_token_id)
                        # With self.token_mask_prob, mask out the tokens randomly for the transcript portion of the tokens
                        for j in range(sop_pos, eop_pos):
                            if np.random.rand() < self.token_mask_prob:
                                # Here we use the scheme of synthesizing ASR errors instead of naive masking to improve the model's robustness
                                with torch.no_grad():
                                    # Check if the model is wrapped by the DeepSpeedEngine
                                    _model = model.module.model.model if hasattr(
                                        model, 'module') else model.model.model
                                    embedding = _model.decoder.embed_tokens(torch.tensor(
                                        [decoder_input_ids[i]['input_ids'][j]]).to(self.args.device))
                                    # Search for the nearest token to the embedding excluding itself
                                    similarities = torch.nn.functional.cosine_similarity(
                                        _model.decoder.embed_tokens.weight, embedding)
                                    similarities[decoder_input_ids[i]
                                                 ['input_ids'][j]] = -1
                                    # Set the scores for all the special tokens to -1
                                    similarities[50257:] = -1
                                    # Randomly select the closest token from the top similarity_sample_range
                                    closest_token_index = torch.topk(
                                        similarities, similarity_sample_range).indices[np.random.randint(similarity_sample_range)]
                                    decoder_input_ids[i]['input_ids'][j] = closest_token_index.item(
                                    )
                        # Replace the first token (<|startofprev|>) with the mask token indicating that the prompt/transcript is not exact
                        decoder_input_ids[i]['input_ids'][0] = self.tokenizer.tokenizer.mask_token_id

                # Pad the labels and decoder_input_ids to the same length
                labels = self.tokenizer.tokenizer.pad(
                    labels, return_tensors="pt")
                decoder_input_ids = self.tokenizer.tokenizer.pad(
                    decoder_input_ids, return_tensors="pt")
                labels = labels["input_ids"].masked_fill(
                    labels.attention_mask.ne(1), -100)
                decoder_input_ids = decoder_input_ids["input_ids"]
                inputs_st['labels'] = labels
                inputs_st['decoder_input_ids'] = decoder_input_ids

                sos_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                    "<|startoftranscript|>")
                decoder_loss_mask = torch.ones_like(labels)
                # Generate the decoder_loss_mask to mask out everything before the <|startoftranscript|> token (inlusive)
                # when computing the loss
                if self.mask_labels_src:
                    for i, seq in enumerate(labels):
                        for j, label in enumerate(seq):
                            if label != sos_id:
                                decoder_loss_mask[i, j] = 0
                            else:
                                # Disallow BP on the <|startoftranscript|> token
                                decoder_loss_mask[i, j] = 0
                                break
                inputs_st['decoder_loss_mask'] = decoder_loss_mask.to(bool)
            inputs_st = BatchFeature(inputs_st)
            inputs_st = self._prepare_inputs(inputs_st)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            current_step = self.state.global_step + 1
            if self.loss_warmup > 0 and current_step > self.loss_warmup:
                alpha = _schedule_dynamic_mtl_weight(
                    current_step=current_step,
                    warmup_steps=self.loss_warmup,
                    max_steps=self.args.max_steps,
                    min_weight=self.min_alpha,
                    max_weight=self.max_alpha,
                    loss_base=self.loss_base,
                )
            else:
                alpha = self.min_alpha
            if self.use_asr_prompt:
                if alpha < 1.0:
                    loss_asr, outputs_asr = self.compute_loss(
                        model, inputs_asr, return_outputs=True)
                else:
                    loss_asr = 0.0
                loss_st = self.compute_loss(model, inputs_st)
                loss = (1 - alpha) * loss_asr + alpha * loss_st
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        with torch.cuda.amp.autocast(cache_enabled=False):
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def __init__(
        self,
        *args,
        use_asr_prompt: bool = False,
        use_asr_prompt_dev: bool = False,
        min_promptless_prob: Optional[float] = 0.0,
        max_promptless_prob: Optional[float] = 0.0,
        batch_mask_prob: Optional[float] = 0.0,
        token_mask_prob: Optional[float] = 0.0,
        src_lang: Optional[str] = None,
        eval_steps: Optional[int] = None,
        min_alpha: Optional[float] = 0.5,
        max_alpha: Optional[float] = 0.5,
        loss_warmup: Optional[int] = -1,
        loss_base: Optional[float] = 0.25,
        mask_labels_src: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_promptless_prob = min_promptless_prob
        self.max_promptless_prob = max_promptless_prob
        self.batch_mask_prob = batch_mask_prob
        self.token_mask_prob = token_mask_prob
        self.src_lang = src_lang
        self.use_asr_prompt = use_asr_prompt
        self.use_asr_prompt_dev = use_asr_prompt_dev
        self.eval_steps = eval_steps
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.loss_warmup = self.args.warmup_steps if loss_warmup == -1 else loss_warmup
        self.loss_base = loss_base
        self.mask_labels_src = mask_labels_src

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["sample_prob"] = self.sample_prob if hasattr(
                self, "sample_prob") else 0.0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    lang, mode = eval_dataset_name.split("_")
                    if "module" in model.__dict__ and type(model.module).__name__ == "PeftModel":
                        model.module.base_model.generate = partial(
                            model.module.base_model.generate, language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")
                    else:
                        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            model.module.generate = partial(
                                model.module.generate, language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")
                        else:
                            model.generate = partial(
                                model.generate, language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                        nolog=True,
                    )
                    metrics.update(dataset_metrics)
                # Sum the sacrebleu metric over all the datasets evaluated on.
                sum_sacrebleu = 0
                sum_cer = 0
                for key, value in metrics.items():
                    if key.endswith("_sacrebleu"):
                        sum_sacrebleu += value
                    if key.endswith("_cer"):
                        sum_cer += value
                # Average the cer metric over all the datasets evaluated on.
                metrics["eval_cer"] = sum_cer / len(self.eval_dataset)
                metrics["eval_sacrebleu"] = sum_sacrebleu / \
                    len(self.eval_dataset)
                self.log(metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        nolog=False,
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get(
                "num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if not nolog:
            self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def _create_prompted_labels(
        self,
        inputs,
        asr_hyp_texts=None,
        padding_side="right",
        return_labels=True,
        startofprev_token="<|startofprev|>",
    ) -> Dict[str, torch.Tensor]:
        inputs_st = {}
        inputs_st['input_features'] = inputs['input_features']

        tokenizer = copy.deepcopy(self.tokenizer.tokenizer)
        tokenizer.padding_side = padding_side
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
        asr_texts = tokenizer.batch_decode(
            inputs['labels_src'], skip_special_tokens=True) if asr_hyp_texts is None else asr_hyp_texts
        if return_labels:
            st_ref_texts = tokenizer.batch_decode(
                inputs['labels_tgt'], skip_special_tokens=False)
            st_ref_texts = [
                f"<|startoftranscript|>{_text}" for _text in st_ref_texts]
        else:
            st_ref_texts = [f"" for _ in asr_texts]
        prompt_texts = [f"{startofprev_token}{asr_text}{st_ref_text}" for asr_text, st_ref_text in zip(
            asr_texts, st_ref_texts)]
        _labels = tokenizer.batch_encode_plus(
            prompt_texts,
            add_special_tokens=False,
            truncation=True,
        ).input_ids
        if return_labels:
            # labels are _labels[1:]
            labels = [{"input_ids": _label[1:]} for _label in _labels]
            # Pad the labels and decoder_input_ids to the same length
            labels = tokenizer.pad(labels, return_tensors="pt")
            labels = labels["input_ids"].masked_fill(
                labels.attention_mask.ne(1), -100)
            inputs_st['labels'] = labels
        # decoder_input_ids are _labels[:-1] if return_labels is True
        # Otherwise, the decoder_input_ids are _labels
        if return_labels:
            decoder_input_ids = [{"input_ids": _label[:-1]}
                                 for _label in _labels]
        else:
            decoder_input_ids = [{"input_ids": _label} for _label in _labels]
        decoder_input_ids = tokenizer.pad(
            decoder_input_ids, return_tensors="pt")
        # Note that no right padding is applied to the labels since they are set to -100 already in the collator
        decoder_input_ids = decoder_input_ids["input_ids"]
        inputs_st['decoder_input_ids'] = decoder_input_ids

        if return_labels:
            # Generate the decoder_loss_mask to mask out everything before the <|startoftranscript|> token (inlusive)
            # when computing the loss
            sos_id = tokenizer.convert_tokens_to_ids(
                "<|startoftranscript|>")
            decoder_loss_mask = torch.ones_like(labels)
            for i, seq in enumerate(labels):
                for j, label in enumerate(seq):
                    if label != sos_id:
                        decoder_loss_mask[i, j] = 0
                    else:
                        # Disallow BP on the <|startoftranscript|> token
                        decoder_loss_mask[i, j] = 0
                        break
            # Cihan: Now this seems a bit weird, maybe a better solution is to set the labels directly to -100
            inputs_st['decoder_loss_mask'] = decoder_loss_mask.to(bool)

        if padding_side == "left":
            # Add decoder_attention_mask to mask out the left padding
            decoder_attention_mask = torch.zeros_like(decoder_input_ids)
            # Replace all non-padding tokens with 1
            decoder_attention_mask = decoder_attention_mask.masked_fill(
                decoder_input_ids.ne(tokenizer.pad_token_id), 1)
            inputs_st['decoder_attention_mask'] = decoder_attention_mask

        inputs_st = BatchFeature(inputs_st)
        inputs_st = self._prepare_inputs(inputs_st)

        return inputs_st

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, loss_asr, logits and labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get(
                "num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get(
                "synced_gpus") is not None else default_synced_gpus
        )

        use_asr_hyp = hasattr(
            self, "use_asr_prompt_dev") and self.use_asr_prompt_dev

        # if use_asr_hyp:
        if False:
            # Note (Cihan): No longer performs on-the-fly ASR hypotheses generation
            has_labels = True
            inputs_asr = {}
            inputs_asr["input_features"] = inputs["input_features"]
            inputs_asr["labels"] = inputs["labels_src"]
            inputs_asr = self._prepare_inputs(inputs_asr)
        else:
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k !=
                      "decoder_input_ids"}

        if hasattr(self, "use_asr_prompt") and self.use_asr_prompt:
            self.model.generate = partial(
                self.model.generate, language=self.src_lang, task="translate")
            # Need to explicitly erase the forced_decoder_ids
            self.model.config.forced_decoder_ids = None
            self.model.generation_config.forced_decoder_ids = None
            # Set the decoder_start_token_ids to a set of ids, i.e. <|startoftranscript|><|language|><|task|><|notimestamps|>
            startofprev_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                "<|startofprev|>") if not use_asr_hyp else self.tokenizer.tokenizer.convert_tokens_to_ids("<|startoflm|>")
            startoftranscript_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                "<|startoftranscript|>")

            # Get the language token id
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            model.generation_config.language = self.src_lang
            if model.generation_config.language in model.generation_config.lang_to_id.keys():
                language_token = model.generation_config.language
            elif model.generation_config.language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[model.generation_config.language]}|>"
            elif model.generation_config.language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{model.generation_config.language}|>"
            else:
                is_language_code = len(
                    model.generation_config.language) == 2
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

            inputs_st = self._create_prompted_labels(
                inputs=inputs,
                asr_hyp_texts=None,
                padding_side="left",
                return_labels=False,
                startofprev_token="<|startofprev|>" if not use_asr_hyp else "<|startoflm|>",
            )
            with torch.cuda.amp.autocast(cache_enabled=False):
                # Here we generate the ST hypotheses
                st_hyps = self.model.generate(**inputs_st, **gen_kwargs)

            # For MT, use the label_src as the prompt, i.e. things to translate
            asr_hyp_texts = self.tokenizer.tokenizer.batch_decode(
                inputs["labels_src"], skip_special_tokens=True)
            # Here we prepare the teacher-forcing labels for loss computation
            prompted_inputs = self._create_prompted_labels(
                inputs=inputs,
                asr_hyp_texts=asr_hyp_texts,
                startofprev_token="<|startofprev|>" if not use_asr_hyp else "<|startoflm|>",
            )
            inputs = prompted_inputs  # A hack
            has_labels = True

            _generated_tokens = [{"input_ids": st_hyp}
                                 for st_hyp in st_hyps]
            # Pad the tokens to the same length with the padding token
            generated_tokens = self.tokenizer.tokenizer.pad(
                _generated_tokens, return_tensors="pt").to(self.args.device)["input_ids"]
        else:
            with torch.cuda.amp.autocast(cache_enabled=False):
                generated_tokens = self.model.generate(
                    **inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    # if use_asr_hyp:
                    if False:
                        # Note (Cihan): No longer performs on-the-fly ASR hypotheses generation
                        with torch.cuda.amp.autocast(cache_enabled=False):
                            outputs_asr = self.model(**inputs_asr)

                            outputs_st = self.model(**prompted_inputs)
                        outputs = outputs_st
                    else:
                        with torch.cuda.amp.autocast(cache_enabled=False):
                            outputs = model(**inputs)
                if self.label_smoother is not None:
                    # Currently does not really support label smoothing under use_asr_hyp
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
                    # if use_asr_hyp:
                    if False:
                        # Note (Cihan): No longer performs on-the-fly ASR hypotheses generation
                        loss_asr = (outputs_asr["loss"] if isinstance(
                            outputs_asr, dict) else outputs_asr[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        # return (loss, loss_asr, generated_tokens, labels) if use_asr_hyp else (loss, generated_tokens, labels)
        # Note (Cihan): No longer returns the ASR loss
        return (loss, generated_tokens, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        # Customized for storing multiple losses
        losses_asr_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        use_asr_hyp = hasattr(
            self, "use_asr_prompt_dev") and self.use_asr_prompt_dev

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # To limit eval steps to a specific number
            if hasattr(self, "eval_steps") and self.eval_steps is not None and step >= self.eval_steps:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            if False:
                loss, loss_asr, logits, labels = self.prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            else:
                loss, logits, labels = self.prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(
                inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics(
                    (loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(
                    losses_host, losses, padding_index=-100)
                # Keep track of th additional ASR loss
                if False:
                    losses_asr_host = loss_asr if losses_asr_host is None else nested_concat(
                        losses_asr_host, loss_asr, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode)
                inputs_decode = self.accelerator.gather_for_metrics(
                    (inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if losses_asr_host is not None:
                    losses_asr = nested_numpify(losses_asr_host)
                    all_losses_asr = (
                        losses_asr if all_losses_asr is None else np.concatenate(
                            (all_losses_asr, losses_asr), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                if False:
                    losses_asr_host = None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            all_losses = nested_numpify(losses_host)
        if preds_host is not None:
            all_preds = nested_numpify(preds_host)
        if inputs_host is not None:
            all_inputs = nested_numpify(inputs_host)
        if labels_host is not None:
            all_labels = nested_numpify(labels_host)
        if False:
            if losses_asr_host is not None:
                all_losses_asr = nested_numpify(losses_asr_host)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds,
                                   label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(
                    predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if False:
            if all_losses_asr is not None:
                metrics[f"{metric_key_prefix}_loss_asr"] = all_losses_asr.mean(
                ).item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

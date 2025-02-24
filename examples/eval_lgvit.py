#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import datasets
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalLoopOutput,
    get_last_checkpoint,
    has_length,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

import transformers.utils as trans_utils

from transformers.data.data_collator import DataCollator
from transformers.integrations import WandbCallback, rewrite_logs, is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.versions import require_version
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback

from timm.data.auto_augment import RandAugment
from timm.data.random_erasing import RandomErasing

if trans_utils.is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# Local/Own
from lgvit_utils.args_utils import new_get_args, setup_environment, print_args, get_env_paths
from lgvit_utils import evaluate_interactive, check_conda_env

setup_environment() # Need to run this so that the 'models' library can be imported
from models.deit_highway import DeiTImageProcessor, DeiTConfig, DeiTHighwayForImageClassification
from models.deit_highway.configuration_deit import configure_logger

""" Fine-tuning a 🤗 Transformers model for image classification"""

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

logger = configure_logger(logging.getLogger("main"))

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class TrainerwithExits(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

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

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

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
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_exit_layers = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, exit_layer = self.prediction_step(model, inputs, prediction_loss_only,
                                                                    ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if exit_layer is not None:
                exit_layer = np.array([exit_layer])
                all_exit_layers = exit_layer if all_exit_layers is None else np.concatenate(
                    [all_exit_layers, exit_layer])

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
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
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

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

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if all_exit_layers is not None:
            for i in range(model.config.num_hidden_layers):
                metrics[f"counts_exit_layer_{i + 1}"] = 0
            unique, counts = np.unique(all_exit_layers, return_counts=True)
            for i in range(len(unique)):
                metrics[f"counts_exit_layer_{unique[i]}"] = int(counts[i])

            metrics["average_exit_layer"] = all_exit_layers.mean().item()

            metrics["speed-up"] = 12 / all_exit_layers.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference",
                                      ['entropies', 'exit_layer', 'hidden_states', 'attentions'])
            else:
                ignore_keys = ['entropies', 'exit_layer', 'hidden_states', 'attentions']

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if trans_utils.is_sagemaker_mp_enabled():  # false
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                        print(outputs['exit_layer'])
                    else:
                        logits = outputs[1]
                        exit_layer = outputs[-1]
                # else:
                #     loss = None
                #     with self.compute_loss_context_manager():
                #         outputs = model(**inputs)
                #     if isinstance(outputs, dict):
                #         logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                #     else:
                #         logits = outputs
                #     # TODO: this needs to be fixed and made cleaner later.
                #     if self.args.past_index >= 0:
                #         self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1 and type(logits) == tuple:
            logits = logits[0]

        return (loss, logits, labels, exit_layer)

def hf_transformers_setup(**kwargs):
    """
    Setup for Huggingface Transformers library
    """
    transformers.utils.logging.set_verbosity(kwargs["verbosity"])
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def get_parsed_args(parser):
    """
    Parse arguments using the provided parser
    """
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Single argument; json path
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args

def retrieve_last_checkpoint(output_dir:str, overwrite_output_dir:bool, resume_from_checkpoint:str):
    if os.path.isdir(output_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)

        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        
        if resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        return last_checkpoint

# Dataset fns
def get_dataset_dict(data_args, model_args):
    task_arg = datasets.ImageClassification(image_column='img', label_column='fine_label')
    if data_args.dataset_name is not None:
        logger.info(f"Loading dataset '{data_args.dataset_name}' with config '{data_args.dataset_config_name}'")
        dataset = datasets.load_dataset(
            path=data_args.dataset_name,
            name=data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            task=task_arg,
            token=True if model_args.use_auth_token else None,
            # use_auth_token=True if model_args.use_auth_token else None,
            # ignore_verifications=True,
        )
    else:
        logger.info(f"Loading dataset from directories. Train '{data_args.train_dir}', Validation '{data_args.validation_dir}'")
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
            
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task=task_arg,
        )
    
    # If we don't have a validation split, split off a percentage of train as validation.
    # data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    
    if "validation" in dataset.keys():
        data_args.train_val_split = None
    elif "valid" in dataset.keys():
        data_args.train_val_split = None
        dataset["validation"] = dataset["valid"]
    elif "test" in dataset.keys():
        data_args.train_val_split = None
        dataset["validation"] = dataset["test"]
    else:
        pass
    
    
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
        logger.info(f"Split train data into train/val with {len(dataset['train'])} samples in train and {len(dataset['validation'])} samples in validation")

    return dataset

def get_label_mappings(labels):
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    return label2id, id2label

# image processor fns
def get_image_processor(model_args):
    logger.info("Loading image processor")
    
    image_processor = DeiTImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    # Define torchvision transforms to be applied to each image.
    image_processor.size['shortest_edge'] = 224
    
    return image_processor

def add_transforms(dataset:datasets.DatasetDict, image_processor:DeiTImageProcessor, training_args, data_args):
    logger.info("Adding transforms to dataset")
    
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        logger.warning("No 'shortest_edge' found in image_processor.size. Using 'height' and 'width' instead.")
        size = (image_processor.size["height"], image_processor.size["width"])
    logger.debug(f"Image size: {size}")
    # transforms
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    logger.debug(f"Image mean: {image_processor.image_mean}, Image std: {image_processor.image_std}")
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch
    
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    return dataset

# Model fns
def get_model_config(model_args, label2id:dict, id2label:dict, do_train:bool, tot_optim_steps:int, save_config:bool=False):
    logger.info(f"Loading DeiTConfig with '{model_args.backbone}' backbone")

    if do_train:
        config = DeiTConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path,
            num_labels=len(label2id.keys()),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            backbone=model_args.backbone,
            threshold=model_args.threshold,
            exit_strategy=model_args.exit_strategy,
            train_strategy=model_args.train_strategy,
            num_early_exits=model_args.num_early_exits,
            position_exits=model_args.position_exits,
            highway_type=model_args.highway_type,
            loss_coefficient=model_args.loss_coefficient,
            homo_loss_coefficient=model_args.homo_loss_coefficient,
            hete_loss_coefficient=model_args.hete_loss_coefficient,
            output_hidden_states=model_args.output_hidden_states,
            # use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = DeiTConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(label2id.keys()),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            backbone=model_args.backbone,
            threshold=model_args.threshold,
            exit_strategy=model_args.exit_strategy,
            # train_strategy=model_args.train_strategy,
            # num_early_exits=model_args.num_early_exits,
            # position_exits=model_args.position_exits,
            # highway_type=model_args.highway_type,
            # loss_coefficient=model_args.loss_coefficient,
            # homo_loss_coefficient=model_args.homo_loss_coefficient,
            # hete_loss_coefficient=model_args.hete_loss_coefficient,
            # feature_loss_coefficient=model_args.feature_loss_coefficient,
            # output_hidden_states=model_args.output_hidden_states,
            # use_auth_token=True if model_args.use_auth_token else None,
        )
    
    if save_config:
        config.save_pretrained("./config_pretrained.json")
        logger.info("config saved")

    config.total_optimization_steps = tot_optim_steps

    return config


def _early_exit():
    logger.info("Exiting program EARLY")
    exit()

def _compare(dataclass1, dataclass2, name1:str, name2:str):
    if dataclass1 != dataclass2:
        logger.warning("dataclasses do not match!")
        at_least_one_found = False
        for arg in vars(dataclass1):
            if getattr(dataclass1, arg) != getattr(dataclass2, arg):
                print(f"Difference: {arg=}: {getattr(dataclass1, arg)} != {getattr(dataclass2, arg)}")
                at_least_one_found = True
        if not at_least_one_found:
            logger.error("No differences found.")
        _early_exit()
    else:
        logger.info(f"{name1} and {name2} match.")

# See all possible arguments in src/transformers/training_args.py or by passing the --help flag to this script.
def main():
    check_conda_env("lgvit")
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # model_args, data_args, training_args = get_parsed_args(parser)

    base_path, checkpoint_path, model_path = get_env_paths()
    new_model_args, new_data_args, new_training_args = new_get_args(base_path, checkpoint_path)

    # # Compare model_args with new_model_args
    # _compare(model_args, new_model_args, "model_args", "new_model_args")
    # _compare(data_args, new_data_args, "data_args", "new_data_args")
    # _compare(training_args, new_training_args, "training_args", "new_training_args")
    # del model_args, data_args, training_args
    model_args, data_args, training_args = new_model_args, new_data_args, new_training_args
    # _early_exit()

    hf_transformers_setup(verbosity=training_args.get_process_log_level())

    # Log on each process the small summary:
    logger.debug(
        f"Process Summary:" + \
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} " +\
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )


    # Detecting last checkpoint.
    last_checkpoint = None
    if training_args.do_train:
        last_checkpoint = retrieve_last_checkpoint(training_args.output_dir, training_args.overwrite_output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info("Initializing dataset")
    dataset_dict = get_dataset_dict(data_args, model_args)
    
    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    label2id, id2label = get_label_mappings(dataset_dict["train"].features["labels"].names)
    
    image_processor = get_image_processor(model_args)
    
    dataset_dict = add_transforms(dataset_dict, image_processor, training_args, data_args)

    logger.info("Dataset initialized")

    total_optimization_steps = int(len(dataset_dict['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs)
    config = get_model_config(model_args, label2id, id2label, training_args.do_train, total_optimization_steps)
    logger.info("Model config loaded")

    print_args([config], ["id2label", "label2id"], [
        "attention_probs_dropout_prob",
        "backbone",
        "encoder_ensemble",
        "encoder_stride",
        "exit_strategy",
        "hete_coefficient",
        "hidden_act",
        "hidden_dropout_prob",
        "hidden_size",
        "highway_type",
        "homo_coefficient",
        "image_size",
        "initializer_range",
        "intermediate_size",
        "is_encoder_decoder",
        "layer_norm_eps",
        "loss_coefficient",
        "num_attention_heads",
        "num_channels",
        "num_early_exits",
        "num_hidden_layers",
        "output_hidden_states",
        "patch_size",
        "position_exits",
        "qkv_bias",
        "threshold",
        "train_strategy",
    ])

    logger.info(f"Loading 'DeiTHighwayForImageClassification' model with {model_args.backbone} backbone")

    test_loader = DataLoader(
        dataset_dict["test"],
        batch_size=1,  # Can be made configurable
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = DeiTHighwayForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        train_highway=True,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    evaluate_interactive(model, test_loader,id2label)

    
    # str_match = "" # 
    # for key, value in model.state_dict().items():
    #     if str_match in key:
    #         print(f"{key}: {value.shape}")

    ###### Trainer ######
    # actions = []
    # if training_args.do_train:
    #     actions.append("train")
    # if training_args.do_eval:
    #     actions.append("eval") 
    # logger.info(f"Initializing 'Trainer' for {actions}") 
    
    # evaluate_interactive(model, test_loader)

    # # Load the accuracy metric from the datasets package
    # metric = evaluate.load("accuracy")

    # def compute_metrics(p):
    #     # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    #     # predictions and label_ids field) and has to return a dictionary string to float.
    #     """Computes accuracy on a batch of predictions"""
    #     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # trainer = TrainerwithExits(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset_dict["train"] if training_args.do_train else None,
    #     eval_dataset=dataset_dict["validation"] if training_args.do_eval else None,
    #     compute_metrics=compute_metrics,
    #     tokenizer=image_processor,
    #     data_collator=collate_fn,
    # )

    # # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()
    #     trainer.log_metrics("train", train_result.metrics)
    #     trainer.save_metrics("train", train_result.metrics)
    #     trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate()
    #     metrics[model.exit_strategy] = model.deit.encoder.early_exit_threshold[0]
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
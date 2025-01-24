# coding=utf-8
# Copyright 2021 Facebook AI Research (FAIR) and The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.
""" DeiT model configuration"""

import logging
from collections import OrderedDict
from typing import Mapping

from packaging import version

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig

def configure_logger(logger: logging.Logger,log_level=logging.DEBUG, log_filepath=None) -> logging.Logger:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s][%(name)s][%(asctime)s]: %(message)s')
    formatter.default_time_format = '%H:%M:%S'
    formatter.default_msec_format = '%s.%03d'
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    # if log_filepath:
    #     file_handler = logging.FileHandler(log_filepath)
    #     file_handler.setFormatter(formatter)
    #     logger.addHandler(file_handler)

    return logger

DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/deit-base-distilled-patch16-224": (
        "https://huggingface.co/facebook/deit-base-patch16-224/resolve/main/config.json"
    ),
    # See all DeiT models at https://huggingface.co/models?filter=deit
}


class DeiTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeiTModel`]. It is used to instantiate an DeiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeiT
    [facebook/deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, `optional`, defaults to 16):
            Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import DeiTConfig, DeiTModel

    >>> # Initializing a DeiT deit-base-distilled-patch16-224 style configuration
    >>> configuration = DeiTConfig()

    >>> # Initializing a model (with random weights) from the deit-base-distilled-patch16-224 style configuration
    >>> model = DeiTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "deit"

    def __init__(
        self,
        attention_probs_dropout_prob=0.0,
        backbone = 'ViT',
        encoder_ensemble=False,
        encoder_stride=16,
        exit_strategy="entropy",  # entropy, confidence, patience, patient_and_confident
        hete_coefficient=0.01,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        hidden_size=768,
        highway_type='linear',
        homo_coefficient=0.01,
        image_size=224,
        initializer_range=0.02,
        intermediate_size=3072,
        is_encoder_decoder=False,
        layer_norm_eps=1e-12,
        loss_coefficient=0.3,
        num_attention_heads=12,
        num_channels=3,
        num_early_exits=4,
        num_hidden_layers=12,
        output_hidden_states=False,
        patch_size=16,
        position_exits=None,
        qkv_bias=True,
        threshold = None,
        train_strategy="normal",  # weighted, alternating
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
        self.num_early_exits = num_early_exits
        self.position_exits = position_exits
        self.encoder_ensemble = encoder_ensemble
        self.exit_strategy = exit_strategy
        self.train_strategy = train_strategy
        self.highway_type = highway_type
        self.loss_coefficient = loss_coefficient
        self.homo_coefficient = homo_coefficient
        self.hete_coefficient = hete_coefficient
        self.output_hidden_states = output_hidden_states
        self.backbone = backbone
        self.threshold = threshold

class DeiTOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

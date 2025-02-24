# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from logging import getLogger, StreamHandler, DEBUG
from typing import TYPE_CHECKING

from transformers.utils import(
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

logger = getLogger(__name__)
stream_handler = StreamHandler()
stream_handler.setLevel(DEBUG)
logger.addHandler(stream_handler)

_import_structure = {"configuration_deit": ["DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeiTConfig", "DeiTOnnxConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
    logger.warning("transformers.utils.is_vision_available() is false.")
else:
    _import_structure["feature_extraction_deit"] = ["DeiTFeatureExtractor"]
    _import_structure["image_processing_deit"] = ["DeiTImageProcessor"]

try: # Check Torch being available
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
    logger.warning("transformers.utils.is_torch_available() is false.")
else:
    _import_structure["modeling_deit"] = [
        "DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeiTSelfAttention",
        "DeiTLayer",
        "DeiTPooler",
        "DeiTForImageClassification",
        "DeiTForImageClassificationWithTeacher",
        "DeiTForMaskedImageModeling",
        "DeiTPreTrainedModel",
    ]
    _import_structure['modeling_highway_deit'] = [
        "DeiTHighwayForImageClassification_distillation",
        "DeiTHighwayForImageClassification_frozenfinetune",
        "DeiTHighwayForImageClassification",
        "DeiTModel",
    ]
    _import_structure['highway'] = [
        "DeiTHighway",
        "DeiTHighway_v2",
        "ViT_EE_Highway",
    ]

try: # Check TensorFlow being available
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_deit"] = [
        "TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDeiTForImageClassification",
        "TFDeiTForImageClassificationWithTeacher",
        "TFDeiTForMaskedImageModeling",
        "TFDeiTModel",
        "TFDeiTPreTrainedModel",
    ]

from .configuration_deit import DeiTConfig, DeiTOnnxConfig

if TYPE_CHECKING:
    from configuration_deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig, DeiTOnnxConfig

    try: # PIL being available
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from feature_extraction_deit import DeiTFeatureExtractor
        from image_processing_deit import DeiTImageProcessor

    try: # Torch being available
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from modeling_deit import (
            DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeiTLayer,
            DeiTForImageClassification,
            DeiTForImageClassificationWithTeacher,
            DeiTForMaskedImageModeling,
            DeiTModel,
            DeiTPreTrainedModel,
        )
        from modeling_highway_deit import (
            DeiTHighway,
            DeiTHighwayForImageClassification
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

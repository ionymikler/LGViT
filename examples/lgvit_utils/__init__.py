from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def print_args(args_list):
    for args in args_list:
        for arg in vars(args):
            print(f"* {arg}: {getattr(args, arg)}")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    
    backbone: str = field(
        default='ViT',
        metadata={
            "help": "choose one backbone: ViT, DeiT"
        }
    )
    
    train_highway: bool = field(
        default=True,
        metadata={
            "help": "train highway"
        }
    )

    threshold: float = field(
        default=0.8,
        metadata={
            "help": "threshold"
        }
    ) 
    
    exit_strategy: str = field(
        default='entropy',
        metadata={
            "help": "choose one exit_strategy: entropy, confidence, patience"
        }
    )

    train_strategy: str = field(
        default='normal',
        metadata={
            "help": "choose one train_strategy: normal, weighted, alternating"
        }
    )

    num_early_exits: int = field(
        default=4,
        metadata={
            "help": "number of exits"
        }
    )

    position_exits: Optional[str] = field(
        default=None,
        metadata={"help": "The position of the exits"}
    )

    highway_type: Optional[str] = field(
        default='linear',
        metadata={
            "help": "choose one highway_type: linear, conv1_1, conv1_2, conv1_3, conv2_1, attention"
        }
    )

    loss_coefficient: float = field(
        default=0.3,
        metadata={
            "help": "the coefficient of the prediction distillation loss"
        }
    )

    homo_loss_coefficient: float = field(
        default=0.01,
        metadata={
            "help": "the coefficient of the homogeneous distillation loss"
        }
    )
    
    hete_loss_coefficient: float = field(
        default=0.01,
        metadata={
            "help": "the coefficient of the heterogeneous distillation loss"
        }
    )
    
    output_hidden_states: bool = field(
        default=False,
        metadata={"help": "whether output_hidden_states and use feature distillation" }
    )
        
    model_name_or_path: str = field(
        default="facebook/deit-base-distilled-patch16-224",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

import logging
import torch
import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# local
from examples.lgvit_utils import (
    ModelArguments,
    DataTrainingArguments,
)

from models.deit_highway import DeiTImageProcessor, DeiTConfig, DeiTHighwayForImageClassification

logger = logging.getLogger(__name__)

# Taken from the bash running scripts
BACKBONE = "ViT" # ViT, DeiT
EXIT_STRATEGY="confidence" # entropy, confidence, patience, patient_and_confident
HIGHWAY_TYPE="LGViT" # linear, LGViT, vit, self_attention, conv_normal
PAPER_NAME="LGViT"  # base, SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT
TRAIN_STRATEGY="no_train" # no_train, normal, weighted, alternating, distillation, alternating_weighted


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
args = [
"--run_name" ,f"${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME}",
"--image_processor_name" ,"facebook/deit-base-distilled-patch16-224",
"--config_name" ,"facebook/deit-base-distilled-patch16-224",
"--model_name_or_path" ,"/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100",
"--dataset_name" ,"uoft-cs/cifar100",
"--output_dir" ,"../outputs/DeiT-base/uoft-cs/cifar100/LGViT/confidence/",
"--remove_unused_columns" ,"False",
"--backbone" ,"ViT", # ViT, DeiT
"--exit_strategy" ,"confidence",
"--do_train" ,"False",
"--do_eval", "True",
"--per_device_eval_batch_size" ,"1",
"--seed" ,"777",
"--report_to" ,"wandb",
"--use_auth_token" ,"False",
"--ignore_mismatched_sizes" ,"False",
]
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

task_arg = datasets.ImageClassification(image_column='img', label_column='fine_label')

dataset_dict = datasets.load_dataset(
    path=data_args.dataset_name,
    name=data_args.dataset_config_name,
    cache_dir=model_args.cache_dir,
    task=task_arg, #Deprecated in 2.13.0
    token=True if model_args.use_auth_token else None,
    # use_auth_token=True if model_args.use_auth_token else None,
    # ignore_verifications=True,
)

image_processor = DeiTImageProcessor.from_pretrained(
    model_args.image_processor_name or model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
)

# attaching the data transformation to the dataset


size = 224
_train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        
    ]
)

def train_transforms(example_batch:dict):
    """Apply _train_transforms across a batch."""
    example_batch["pixel_values"] = [
        _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
    ]
    del example_batch["image"]

    return example_batch

dataset_dict["train"].set_transform(train_transforms)

# Prepare label mappings.
# We'll include these in the model's config to get human readable labels in the Inference API.
labels = dataset_dict["train"].features["labels"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# only eval case
config = DeiTConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=len(labels),
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

total_optimization_steps = int(len(dataset_dict['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs)
config.total_optimization_steps = total_optimization_steps

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



X = dataset_dict["train"][0]["pixel_values"]
X = torch.unsqueeze(X, 0) # make X into batch shape
y = model(X)

print(X.shape)
print(y[0].shape)

torch.onnx.export(model=model,args=X, f="model.onnx", input_names=["image_batch"], output_names=["labels_pred"])

print("Model exported to model.onnx")
import torch
import logging
from typing import Dict
from torch.utils.data import DataLoader
from .logger_cfg import configure_logger

logger = configure_logger(logging.getLogger("lgvit_utils"))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_interactive(model: torch.nn.Module, test_loader: DataLoader, id2label:Dict[int,str]) -> None:
    """
    Interactive evaluation of the model, showing detailed results for each image.
    """
    model.to(_DEVICE)
    model.eval()

    logger.info("Starting interactive evaluation...")
    logger.info("Press Enter to continue to next image, or 'q' to quit")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Wait for user input
            user_input = input("\nPress Enter for next image, or 'q' to quit: ")
            if user_input.lower() == "q":
                print("\nExiting interactive evaluation...")
                break

            images = batch["pixel_values"].to(_DEVICE)
            labels = batch["labels"].to(_DEVICE)
            label_name = id2label[str(labels.item())] # Since batch size is 1. Will need to change this to support multiple images

            # Get model predictions
            logits, confidence, exit_layer = model(images) # logits, confidence, exit_layer
            confidence = confidence[0] # original_score, highway_score. NOTE: original_score can also come from highway. Not sure how this works
            predictions = torch.softmax(logits, dim=1)

            # Get predicted class and confidence
            _, predicted_class = torch.max(predictions, dim=1)
            predicted_name = id2label[str(predicted_class.item())]

            # Print results
            print("\n" + "=" * 50)
            print(f"Image {batch_idx + 1}")
            print(f"True label: {label_name} (id: {labels.item()})")
            print(f"Predicted label: {predicted_name} (id: {predicted_class})")
            print(f"Confidence: {confidence.item():.2%}")
            print(f"Exit layer: {exit_layer}")
            print("=" * 50)


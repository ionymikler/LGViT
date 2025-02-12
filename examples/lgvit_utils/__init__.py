import torch
import logging
from torch.utils.data import DataLoader
from .logger_cfg import configure_logger

logger = configure_logger(logging.getLogger("lgvit_utils"))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_interactive(model: torch.nn.Module, test_loader: DataLoader) -> None:
    """
    Interactive evaluation of the model, showing detailed results for each image.
    """
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
            label_name = batch["label_names"][0]  # Since batch size is 1

            # Get model predictions
            outputs = model(images)
            predictions = outputs[:, :-1]  # Remove exit layer index
            exit_layer = outputs[:, -1].item()  # Get exit layer

            # Get predicted class and confidence
            confidence, predicted_class = torch.max(predictions, dim=1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            predicted_name = test_loader.dataset.features["fine_label"].names[
                predicted_class
            ]

            # Print results
            print("\n" + "=" * 50)
            print(f"Image {batch_idx + 1}")
            print(f"True label: {label_name} (class {labels.item()})")
            print(f"Predicted: {predicted_name} (class {predicted_class})")
            print(f"Confidence: {confidence:.2%}")
            print(f"Exit layer: {exit_layer if exit_layer != -1 else 'Final layer'}")
            print("=" * 50)


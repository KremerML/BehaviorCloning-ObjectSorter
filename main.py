import torch
import argparse
import model.training as training

def main(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Notify about logging
    print("Training started. Check log file for details.")

    # Perform training and evaluation
    accuracy = training.train_and_evaluate(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        probs_file=args.probs_file,
        counts_file=args.counts_file,
        num_classes=args.num_classes,
        device=device,
        epochs=args.epochs
    )

    # Final results to console
    print(f"Final Accuracy: {accuracy}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10H with human-annotated labels")

    parser.add_argument('--csv_file', type=str, required=True, help='Path to the cifar10h-raw.csv file')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for image files')
    parser.add_argument('--probs_file', type=str, required=True, help='Path to the cifar10h-probs.npy file')
    parser.add_argument('--counts_file', type=str, required=True, help='Path to the cifar10h-counts.npy file')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes to classify')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    args = parser.parse_args()
    main(args)

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image

import glob
import os

transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(p=0.5)]
)

train_dataset = torchvision.datasets.ImageFolder(root="Images", transform=transform)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.image_paths = []
        for ext in ["png"]:
            self.image_paths += glob.glob(os.path.join(root_dir, "*", f"*.{ext}"))
        self.image_paths.sort()
        class_set = set()
        for path in self.image_paths:
            class_set.add(os.path.dirname(path))
        self.class_lbl = {cls: i for i, cls in enumerate(sorted(list(class_set)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx], ImageReadMode.RGB).float()
        cls = os.path.dirname(self.image_paths[idx])
        label = self.class_lbl[cls]

        return self.transform(img), torch.tensor(label)


def create_model():
    """Creates a model with a pretrained ResNet50 backbone and a custom head"""
    # Instantiate the dataset
    dataset = CustomDataset(root_dir="Images/", transform=transform)
    print("Dataset instantiated")

    # Create the splits
    splits = [0.8, 0.1, 0.1]
    split_sizes = []
    for split in splits[:-1]:
        split_sizes.append(int(split * len(dataset)))
    split_sizes.append(len(dataset) - sum(split_sizes))
    print("Splits created")

    # Split the dataset using torch.utils.data.random_split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, split_sizes
    )
    print(test_dataset)
    print("Dataset split")

    # Define the data loaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True),
        "val": torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False),
    }
    print("Data loaders defined")

    # Initalize the model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    print("Model initialized")

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 13),
    )

    # Move the model to the GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model moved to", device)

    # Freeze parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    print("Loss function defined")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

    # Define metrics
    metrics = {
        "train": {"loss": [], "acc": []},
        "val": {"loss": [], "acc": []},
    }

    # Train the model
    for epoch in range(300):
        epoch_metrics = {
            "train": {"loss": 0, "accuracy": 0, "count": 0},
            "val": {"loss": 0, "accuracy": 0, "count": 0},
        }
        print(f"Epoch {epoch + 1}")

        for phase in ["train", "val"]:
            print(f"Phase: {phase}")
            for images, labels in dataloaders[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = model(images.to(device))
                    # One-hot encode the label
                    ohe_label = torch.nn.functional.one_hot(labels, num_classes=13)

                    loss = criterion(output, ohe_label.float().to(device))

                    correct_preds = labels.to(device) == torch.argmax(output, dim=1)
                    accuracy = (correct_preds).sum() / len(labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                epoch_metrics[phase]["loss"] += loss.item()
                epoch_metrics[phase]["accuracy"] += accuracy.item()
                epoch_metrics[phase]["count"] += 1

            ep_loss = epoch_metrics[phase]["loss"] / epoch_metrics[phase]["count"]
            ep_acc = epoch_metrics[phase]["accuracy"] / epoch_metrics[phase]["count"]

            print(f"Loss: {ep_loss}, Accuracy: {ep_acc}")

            metrics[phase]["loss"].append(ep_loss)
            metrics[phase]["acc"].append(ep_acc)

    """# Visualize the metrics
    for phase in metrics:
        for metric in metrics[phase]:
            metric_data = metrics[phase][metric]
            plt.plot(range(len(metric_data)), metric_data)
            plt.xlabel("Epoch")
            plt.ylabel(f"{phase} {metric}")
            plt.show()"""

    # Test the model
    preds = []
    actual = []

    total_loss = total_acc = count = 0

    for images, labels in dataloaders["test"]:
        with torch.set_grad_enabled(False):
            output = model(images.to(device))
            # One-hot encode the label
            ohe_label = torch.nn.functional.one_hot(labels, num_classes=13)
            out_labels = torch.argmax(output, dim=1)

            total_loss += criterion(output, ohe_label.float().to(device))
            total_acc += (labels.to(device) == out_labels).sum() / len(labels)
            count += 1

        preds += out_labels.tolist()
        actual += labels.tolist()

    print(f"Test loss: {total_loss/count}, Test accuracy: {total_acc/count}")

    """# Confusion matrix
    class_labels = ["bishop", "king", "knight", "pawn", "queen", "rook"]
    cm = sklearn.metrics.confusion_matrix(actual, preds)
    disp = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_labels
    )

    disp.plot()
    plt.show()"""

    # Save the model
    torch.save(model, "chess_piece_classifier_refined.pt")
    print("Model saved")


def load_model():
    """Loads the model"""
    model = torch.load("chess_piece_classifier_refined.pt")
    model.eval()
    return model


def rotate_board(chess_board):
    """Rotates the board 90 degrees counter-clockwise"""
    R, C = len(chess_board), len(chess_board[0])
    rotated_board = [[None] * R for _ in range(C)]
    for c in range(C):
        for r in range(R - 1, -1, -1):
            rotated_board[C - c - 1][r] = chess_board[r][c]

    return rotated_board


def convert_to_fen(chess_board):
    fen_piece_map = {
        "bishop_dark": "b",
        "bishop_light": "B",
        "king_dark": "k",
        "king_light": "K",
        "knight_dark": "n",
        "knight_light": "N",
        "pawn_dark": "p",
        "pawn_light": "P",
        "queen_dark": "q",
        "queen_light": "Q",
        "rook_dark": "r",
        "rook_light": "R",
    }

    fen = ""
    for row in chess_board:
        empty_count = 0
        for square in row:
            if square == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += fen_piece_map[square]
        if empty_count > 0:
            fen += str(empty_count)
        fen += "/"
    return fen[:-1]


def main():
    # create_model()
    # create_model()

    chess_board = []
    # Load the model
    model = load_model()
    print("Model loaded")

    # Current chess board
    chess_board = []

    # Create dataset
    my_dataset = CustomDataset(root_dir="data/", transform=transform)
    print("Dataset instantiated")

    # Create dataloader
    my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=False)
    # Print labels
    my_dataloader.dataset
    print("Dataloader instantiated")

    # Move the model to the GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_labels = [
        "bishop_dark",
        "bishop_light",
        "empty",
        "king_dark",
        "king_light",
        "knight_dark",
        "knight_light",
        "pawn_dark",
        "pawn_light",
        "queen_dark",
        "queen_light",
        "rook_dark",
        "rook_light",
    ]
    # Get batch
    for images, labels in my_dataloader:
        # Get predictions
        with torch.no_grad():
            output = model(images.to(device))
            out_labels = torch.argmax(output, dim=1)
            chess_board.append(class_labels[out_labels])
            # print(class_labels[out_labels])
    # Additions
    chess_board = [chess_board[i : i + 8] for i in range(0, 64, 8)]
    # Rotate the board
    chess_board = rotate_board(chess_board)
    print("Rotated board:")
    for row in chess_board:
        print(row)

    # Convert to FEN
    fen = convert_to_fen(chess_board)
    print(fen)
    print("Done")

    return 0


if __name__ == "__main__":
    main()

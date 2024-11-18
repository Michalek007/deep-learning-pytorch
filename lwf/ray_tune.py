from functools import partial
import tempfile
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from lwf_dataset import LWFDataset
from model import NeuralNetwork
import os
from pathlib import Path
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from core import test_loop

IN_CHANNELS = 3
IMAGE_H = 160
IMAGE_W = 160


def load_data(data_dir="./data"):
    train_data = LWFDataset(
        root=data_dir,
        split="train",
        download=True,
        transform=ToTensor()
    )

    test_data = LWFDataset(
        root=data_dir,
        # split="test",
        download=True,
        transform=ToTensor()
    )
    return train_data, test_data


def train_lwf(config, data_dir=None):
    net = NeuralNetwork(IN_CHANNELS, (IMAGE_H, IMAGE_W), config["c1"], config["f2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.TripletMarginLoss(margin=config["m"])
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["b"]), shuffle=True, num_workers=6
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["b"]), shuffle=True, num_workers=6
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, (anchor, positive, negative) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            anchor_pred = net(anchor)
            positive_pred = net(positive)
            negative_pred = net(negative)
            loss = criterion(anchor_pred, positive_pred, negative_pred)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, (anchor, positive, negative) in enumerate(valloader, 0):
            with torch.no_grad():
                anchor_pred = net(anchor)
                positive_pred = net(positive)
                negative_pred = net(negative)

                total += positive_pred.size(0) + negative_pred.size(0)
                distance = (anchor_pred-positive_pred).norm(dim=1)
                correct += (distance <= config["m"]).sum().item()
                distance = (anchor_pred-negative_pred).norm(dim=1)
                correct += (distance > config["m"]).sum().item()

                loss = criterion(anchor_pred, positive_pred, negative_pred)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished Training")


def test_accuracy(net, margin, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=2
    )
    accuracy, loss = test_loop(testloader, net, None, margin)

    return accuracy


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "c1": tune.choice([2**i for i in range(2, 7)]),
        "f2": tune.choice([2**i for i in range(2, 9)]),
        "m": tune.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "lr": tune.loguniform(1e-2, 1e-1),
        "b": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_lwf, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = NeuralNetwork(IN_CHANNELS, (IMAGE_H, IMAGE_W), best_trial.config["c1"], best_trial.config["f2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, best_trial.config["m"], device)
        print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

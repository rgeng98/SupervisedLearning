import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import GengAI.Networks.Builder as B
import logging
from argparse import Namespace
import numpy as np

log = logging.getLogger(__name__)

def getAccuracy(t1: torch.Tensor, t2: torch.Tensor) -> float:
    acc = torch.argmax(t1, dim = -1) == t2
    return acc.float().detach().mean().cpu().item()

class Model(nn.Module):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.device = torch.device(config.Device if torch.cuda.is_available() else "cpu")
        
        self.NN = B.BuildFromConfig(config.ModelConfig).to(self.device)

        self.load_data(config)

        self.optimizer = torch.optim.Adadelta(self.NN.parameters(), lr = config.LearningRate)
        self.loss = nn.CrossEntropyLoss()
        self.init_metric()
        
    def init_metric(self):
        log.info(f"Initializing metrics for tracking.")
        self.N = 1
        self.metrics = {
            "training_epoch_loss":       [],
            "training_epoch_accuracy":   [],
            "validation_epoch_loss":     [],
            "validation_epoch_accuracy": []
        }

    def load_data(self, config: Namespace) -> None:
        log.info(f"Initializing Data Loaders.")
        trnsfrm = transforms.Compose([
            transforms.Resize((config.ImageDimX, config.ImageDimY)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        training_dataset = datasets.ImageFolder(root=config.TrainDir, transform=trnsfrm)
        validation_dataset = datasets.ImageFolder(root=config.ValidationDir, transform=trnsfrm)
        self.idx_identifier = training_dataset.class_to_idx

        log.info(class_mapping := self.idx_identifier)

        self.training_data_loader = DataLoader(
            training_dataset,
            batch_size=config.BatchSize,
            shuffle=True,
            num_workers=config.NumWorkers,
            pin_memory=True
        )
        self.validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=config.BatchSize,
            shuffle=True,
            num_workers=config.NumWorkers,
            pin_memory=True
        )

    def training_epoch(self) -> None:
        log.info(f"Training Epoch {self.N}")
        self.NN.train()
        error_tracker = []
        acc_tracker = []

        for data, idx in self.training_data_loader:
            data = data.to(self.device)
            idx  = idx.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.NN(data)
            err = self.loss(outputs, idx)
            acc_tracker.append(getAccuracy(outputs, idx))
            err.backward()
            error_tracker.append(err.detach().cpu().item())
            self.optimizer.step()
        avg_error = np.mean(error_tracker)
        avg_acc   = np.mean(acc_tracker)

        log.info(f"Average training error after epoch:    {avg_error}")
        log.info(f"Average training accuracy after epoch: {avg_acc}")

        self.metrics["training_epoch_loss"].append(avg_error)
        self.metrics["training_epoch_accuracy"].append(avg_acc)

        self.N = self.N + 1

    def validation_epoch(self) -> None:
        log.info(f"Validating...")
        self.NN.eval()
        with torch.no_grad():
            error_tracker = []
            acc_tracker = []
            for data, idx in self.validation_data_loader:
                data = data.to(self.device)
                idx  = idx.to(self.device)
                outputs = self.NN(data)
                acc_tracker.append(getAccuracy(outputs, idx))
                err = self.loss(outputs, idx)
                error_tracker.append(err.cpu().item())
        avg_error = np.mean(error_tracker)
        avg_acc   = np.mean(acc_tracker)

        log.info(f"Average validation error after epoch:    {avg_error}")
        log.info(f"Average validation accuracy after epoch: {avg_acc}")

        self.metrics["validation_epoch_loss"].append(avg_error)
        self.metrics["validation_epoch_accuracy"].append(avg_acc)


    def display_model_performance(self) -> None:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.metrics["training_epoch_loss"], label="Training", color="purple")
        plt.plot(self.metrics["validation_epoch_loss"], label="Validation", color="green")
        
        plt.plot(self.metrics["training_epoch_accuracy"], color="purple", linestyle="dashed")
        plt.plot(self.metrics["validation_epoch_accuracy"], color="green", linestyle="dashed")
        plt.xlabel("Epoch [-]")
        plt.ylabel("Loss [-]")
        plt.legend()
        plt.show()



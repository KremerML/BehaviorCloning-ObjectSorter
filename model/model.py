import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import vgg11, VGG11_Weights
import torch.nn.functional as F
import wandb

# Base class for models handling human-annotated labels
class HumanAnnotatedModel:
    def __init__(self, device, num_classes=2):
        # Initialize common elements like device and number of classes
        self.device = device
        self.num_classes = num_classes
        self.model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        # Modify the classifier
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        self.model = self.model.to(device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def process_labels(self, labels):
        # Process labels according to the specific model's approach
        pass

    def train(self, train_loader, epochs=10, log_file=None):
        wandb.init(project='human-anno-model-comparison', entity='mnkbone')

        self.model.train()
        for epoch in range(epochs):
            with tqdm(train_loader, unit='batch', desc=f'Training Epoch {epoch+1}/{epochs}') as tepoch:
                total_loss = 0
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    tepoch.set_postfix(loss=total_loss / (tepoch.n + 1))

                avg_loss = total_loss / len(train_loader)
                wandb.log({"epoch": epoch, "loss": avg_loss})

                if log_file:
                    log_file.write(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}\n')

    def evaluate(self, test_loader, log_file=None):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(test_loader, unit='batch', desc='Evaluating') as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    tepoch.set_postfix(accuracy=f'{100 * correct / total:.2f}%')

        accuracy = 100 * correct / total
        if log_file:
            log_file.write(f'Accuracy: {accuracy}%\n')
        return accuracy


class PooledMultinomialModel(HumanAnnotatedModel):
    def __init__(self, device, num_classes=2):
        super().__init__(device, num_classes)
        # Change the loss function to KLDivLoss for handling probabilities
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def process_labels(self, labels):
        # Convert labels to probability distributions
        # Assuming labels are already normalized probabilities
        return labels

    def train(self, train_loader, epochs=10, log_file=None):
        wandb.init(project='human-anno-model-comparison', entity='mnkbone')
        self.model.train()
        for epoch in range(epochs):
            with tqdm(train_loader, unit='batch', desc=f'Training Epoch {epoch+1}/{epochs}') as tepoch:
                total_loss = 0
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels = self.process_labels(labels)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    outputs = F.log_softmax(outputs, dim=1)
                    loss = self.criterion(outputs, labels)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    tepoch.set_postfix(loss=total_loss / (tepoch.n + 1))

                avg_loss = total_loss / len(train_loader)
                wandb.log({"epoch": epoch, "loss": avg_loss})

                if log_file:
                    log_file.write(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}\n')

# Dawid-Skene Model
class DawidSkeneModel(HumanAnnotatedModel):
    def __init__(self, device, num_classes=2):
        super().__init__(device, num_classes)
        # Additional initialization specific to this model

    def process_labels(self, labels):
        # Process labels for the Dawid-Skene Model
        pass

# Raykar et al. Model
class RaykarModel(HumanAnnotatedModel):
    def __init__(self, device, num_classes=2):
        super().__init__(device, num_classes)
        # Additional initialization specific to this model

    def process_labels(self, labels):
        # Process labels for the Raykar et al. Model
        pass
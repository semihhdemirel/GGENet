import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from tqdm import tqdm
from utils.data_loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

class Train:
    def __init__(self, args):
        self.args = args
        self.model = self.get_model()  # Initialize model

    def get_model(self):
        if self.args.model == 'GGENet_S':
            from models.GGENet import GGENet_S
            return GGENet_S()
        elif self.args.model == 'GGENet_T':
            from models.GGENet import GGENet_T
            return GGENet_T()
        else:
            raise ValueError(f"Invalid model name: {self.args.model}")

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def get_data_loader(self):
        return DataLoader(self.args.fold_dir, self.args.batch_size, self.args.fold)

    def metrics(self, labels, preds):
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        return accuracy, f1, recall, precision
    
    def save_model(self, model, optimizer, val_accuracy, epoch):
        save_path = f"{self.args.output_dir}/Fold{self.args.fold}/{type(model).__name__.lower()}/epoch{epoch}_{type(model).__name__.lower()}_{val_accuracy:.4f}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_accuracy,
        }, save_path)

    def train_one_epoch(self, model, epoch, device, optimizer, criterion):
        model.train()
        train_labels = []
        train_preds = []
        total_loss = 0

        with tqdm(self.get_data_loader().train_loader(), unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1} [Training]")
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()  # Reset gradients
                outputs, _ = model(inputs)  # Forward pass

                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Optimize

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

                _, preds = torch.max(outputs, 1)
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(tepoch)
        # Compute training metrics
        train_accuracy, train_f1, train_recall, train_precision = self.metrics(train_labels, train_preds)
        LOGGER.info(f"Epoch {epoch+1}: Training Loss: {avg_loss:.4f}")  
        LOGGER.info(f'Training - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}')

    def validate(self, model, epoch, device, optimizer, criterion):
        model.eval()  # Set the model to evaluation mode
        val_labels = []
        val_preds = []
        total_loss = 0

        with torch.no_grad():  # Disable gradient computation during validation
            with tqdm(self.get_data_loader().val_loader(), unit="batch") as vepoch:
                for inputs, labels in vepoch:
                    vepoch.set_description(f"Epoch {epoch+1} [Validation]")
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs, _ = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate validation loss
                    total_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())

                    vepoch.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(vepoch)
        # Compute validation metrics
        val_accuracy, val_f1, val_recall, val_precision = self.metrics(val_labels, val_preds)
        self.save_model(model, optimizer, val_accuracy, epoch)
        LOGGER.info(f"Epoch {epoch+1}: Validation Loss: {avg_loss:.4f}")
        LOGGER.info(f'Validation - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}')

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold in range(self.args.fold, self.args.fold + 1):
            LOGGER.info(f"Starting Fold {self.args.fold}/{self.args.num_folds}")
            model = self.get_model().to(device)
            optimizer = self.get_optimizer()
            criterion = nn.CrossEntropyLoss()
            for epoch in range(self.args.num_epochs):
                self.train_one_epoch(model, epoch, device, optimizer, criterion)
                self.validate(model, epoch, device, optimizer, criterion)
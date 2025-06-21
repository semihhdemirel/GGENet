import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from tqdm import tqdm
from utils.data_loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from models.GGENet import GGENet_S, GGENet_T
import torch.nn.functional as F

def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

class Train_Knowledge_Distillation:
    def __init__(self, args):
        self.args = args

    def get_data_loader(self):
        return DataLoader(self.args.fold_dir, self.args.batch_size, self.args.fold)

    def metrics(self, labels, preds):
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        return accuracy, f1, recall, precision
    
    def save_model(self, model, optimizer, val_accuracy, epoch):
        save_path = f"{self.args.output_dir}/Fold{self.args.fold}/{type(model).__name__.lower()}_akd/epoch{epoch}_{type(model).__name__.lower()}_akd_{val_accuracy:.4f}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_accuracy,
        }, save_path)

    def FeatureMapDistillationLoss(self, teacher_features, student_features):
        loss = 0
        num_features = len(teacher_features)
        for tf, sf in zip(teacher_features, student_features):
            loss += F.mse_loss(sf, tf)
        return loss / num_features

    def train_knowledge_distillation(self, teacher, student, feature_map_weight, ce_loss_weight, epoch, optimizer, ce_loss, device):
        teacher.to(device)
        student.to(device)
        teacher.eval()  # Teacher set to evaluation mode
        student.train() # Student to train mode
        train_labels = []
        train_preds = []
        total_loss = 0

        with tqdm(self.get_data_loader().train_loader(), unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1} [Training]")
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()  # Reset gradients
                with torch.no_grad():
                    _, teacher_feature_map = teacher(inputs)
                
                student_logits, student_feature_map = student(inputs)

                hidden_rep_loss = self.FeatureMapDistillationLoss(teacher_feature_map, student_feature_map)

                label_loss = ce_loss(student_logits, labels)

                feature_map_weight = hidden_rep_loss / (hidden_rep_loss + label_loss)

                feature_map_weight = max(0.25, min(feature_map_weight, 0.75))

                ce_loss_weight = 1- feature_map_weight
                loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

                loss.backward()  # Backpropagation
                optimizer.step()  # Optimize

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                _, preds = torch.max(student_logits, 1)
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
            teacher_model = GGENet_T().to(device)
            teacher_checkpoint = torch.load(self.args.teacher_model_path)
            if "model_state_dict" in teacher_checkpoint:
                teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
            else:
                teacher_model.load_state_dict(teacher_checkpoint)
            teacher_model = teacher_model.to(device)
            student_model = GGENet_S().to(device)
            optimizer = optim.Adam(student_model.parameters(), lr=self.args.learning_rate)
            ce_loss = nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for epoch in range(self.args.num_epochs):
                self.train_knowledge_distillation(teacher=teacher_model, student=student_model, feature_map_weight=0.25, ce_loss_weight=0.75, epoch=epoch, optimizer=optimizer, ce_loss=ce_loss, device = device)
                self.validate(student_model, epoch, device, optimizer, ce_loss)

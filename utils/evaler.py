import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from tqdm import tqdm
from utils.data_loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models.GGENet import GGENet_T, GGENet_S

plt.rcParams["font.serif"] = "Times New Roman"

def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

class Evaler:
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
    
    def roc(self, Y_true, Y_pred, modelName):
        plt.figure()
        y_true = label_binarize(Y_true, classes=[0, 1, 2, 3, 4, 5])
        y_pred = label_binarize(Y_pred, classes=[0, 1, 2, 3, 4, 5])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(6):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], label=os.path.splitext(modelName)[0] + '(AUC = {0:0.4f})'''.format(roc_auc["micro"]), linewidth=4)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate', fontsize=36)
        plt.ylabel('True positive rate', fontsize=36)
        plt.title(f'Receiver operating characteristic curve of {os.path.splitext(modelName)[0]} in Fold {self.args.fold}', fontsize=32)
        plt.legend(loc="lower right", fontsize=36)
        plt.tick_params(axis='both', labelsize=32)
        plt.show()

    def plot_confusion_matrix(self, cm, class_names, model_name):
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot_kws={"size": 36})
        plt.xlabel('Predicted label', fontsize=36)
        plt.ylabel('True label', fontsize=36)
        plt.title(f'Confusion matrix of {os.path.splitext(model_name)[0]} in Fold {self.args.fold}', fontsize=32)
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
        plt.show()
    
    def evaluation(self, model, device):
        Y_true = []
        Y_pred = []
        total_samples = 0
        total_correct = 0
        with torch.no_grad():
            tqdm_loader = tqdm(self.get_data_loader().test_loader(), desc='Testing', unit='batch')
            for inputs, labels in tqdm_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                Y_true.extend(labels.cpu().numpy())
                Y_pred.extend(predicted.cpu().numpy())
        return Y_true, Y_pred

    def eval(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        main_path = os.path.join(self.args.fold_path, "Fold" + str(self.args.fold))
        models = os.listdir(main_path)
        for model_name in models:
            model_path = os.path.join(main_path, model_name)
            checkpoint = torch.load(model_path)
            model = GGENet_T() if model_name == "GGENet-T.pth" else GGENet_S()
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            start_time = time.time()
            Y_true, Y_pred = self.evaluation(model, device)
            elapsed_time = time.time() - start_time
            if self.args.metrics:
                accuracy, f1, recall, precision = self.metrics(Y_true, Y_pred)
                LOGGER.info(f"Model: {model_name}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Elapsed Time (seconds): {elapsed_time:.4f}")
            if self.args.roc:
                self.roc(Y_true, Y_pred, model_name)
            if self.args.cm:
                cm = confusion_matrix(Y_true, Y_pred)
                self.plot_confusion_matrix(cm, class_names=['FA', 'FB', 'FO', 'RA', 'RB', 'RO'], model_name = model_name)


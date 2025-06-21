import logging
import argparse
from utils.trainer_knowledge_distillation import Train_Knowledge_Distillation 

def get_args_parser(add_help = True):
    parser = argparse.ArgumentParser(description='Train the student model.')
    parser.add_argument('--fold-dir', type=str, default='./fold_dataset', help='Path to dataset folds.')
    parser.add_argument('--teacher-model-path', type=str, default='./ckpt/Fold1/GGENet_T/Epoch99_GGENet_T_0.9000.pth', help='Path to the teacher model checkpoint.')
    parser.add_argument('--fold', type=int, default=1, help='Fold to train the model. Default: 1')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--output-dir', type=str, default='./ckpt', help='Directory to save the model checkpoints.')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds to train the model.')
    return parser

def main(args):
    print(args)
    trainer_kd = Train_Knowledge_Distillation(args)
    trainer_kd.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
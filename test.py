import logging
import argparse
from utils.evaler import Evaler 

def get_args_parser(add_help = True):
    parser = argparse.ArgumentParser(description='Train the student model.')
    parser.add_argument('--fold-dir', type=str, default='./fold_dataset', help='Path to dataset folds.')
    parser.add_argument('--fold-path', type=str, default='./ckpt', help='Path to the fold folders.')
    parser.add_argument('--fold', type=int, default=1, help='Fold to train the model. Default: 1')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for testing.')
    parser.add_argument('--metrics', action = 'store_true', help='Compute the accuracy, f1, recall, precision and elapsed time')
    parser.add_argument('--roc', action = 'store_true', help='Compute the ROC curve')
    parser.add_argument('--cm', action = 'store_true', help='Compute the confusion matrix')
    return parser

def main(args):
    print(args)
    test = Evaler(args)
    test.eval()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
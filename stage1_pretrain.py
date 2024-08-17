import time
import argparse
import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from data_proc import FBanksCrossEntropyDataset
from models import FBankCrossEntropyNetV2
from utils import restore_objects, save_model, save_objects, restore_model
from trainer.cross_entropy_train import train, test


def main(args):
    model_path = 'saved_models_cross_entropy/{args.num_layers}/'
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

    import multiprocessing
    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    train_dataset = FBanksCrossEntropyDataset(args.train_folder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = FBanksCrossEntropyDataset(args.test_folder)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = FBankCrossEntropyNetV2(num_layers=args.num_layers, reduction='mean').to(device)
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(model_path, (0, 0, [], [], [], []))
    start = last_epoch + 1 if max_accuracy > 0 else 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(start, args.epochs):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, 500)
        test_loss, test_accuracy = test(model, device, test_loader)
        print('After epoch: {}, train_loss: {}, test loss is: {}, train_accuracy: {}, '
              'test_accuracy: {}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies), epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FBank Cross Entropy Training Script')

    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--train_folder', type=str, default='fbanks_train', help='Training dataset folder')
    parser.add_argument('--test_folder', type=str, default='fbanks_test', help='Testing dataset folder')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer')

    args = parser.parse_args()

    main(args)

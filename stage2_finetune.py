import time
import numpy as np
import torch
import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trainer.triplet_loss_train import train, test
from utils.pt_util import restore_model, restore_objects, save_model, save_objects
from data_proc.triplet_loss_dataset import FBanksTripletDataset
from models.triplet_loss_model import FBankTripletLossNet
import argparse


def main(num_layers, lr, epochs, batch_size, pretrained_model_path, output_model_path, train_data, test_data):
    use_cuda = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', device)

    import multiprocessing
    print('Number of CPUs:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    train_dataset = FBanksTripletDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_dataset = FBanksTripletDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    model = FBankTripletLossNet(num_layers=num_layers, margin=0.2).to(device)
    model = restore_model(model, pretrained_model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies, train_negative_accuracies, \
    test_positive_accuracies, test_negative_accuracies = restore_objects(pretrained_model_path, (0, 0, [], [], [], [], [], []))

    start = last_epoch + 1 if max_accuracy > 0 else 0

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(start, start + epochs):
        train_loss, train_positive_accuracy, train_negative_accuracy = train(model, device, train_loader, optimizer,
                                                                             epoch, 500)
        test_loss, test_positive_accuracy, test_negative_accuracy = test(model, device, test_loader)
        print('After epoch: {}, train loss is : {}, test loss is: {}, '
              'train positive accuracy: {}, train negative accuracy: {}, '
              'test positive accuracy: {}, and test negative accuracy: {}'
              .format(epoch, train_loss, test_loss, train_positive_accuracy, train_negative_accuracy,
                      test_positive_accuracy, test_negative_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_positive_accuracies.append(train_positive_accuracy)
        test_positive_accuracies.append(test_positive_accuracy)

        train_negative_accuracies.append(train_negative_accuracy)
        test_negative_accuracies.append(test_negative_accuracy)

        test_accuracy = (test_positive_accuracy + test_negative_accuracy) / 2

        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, output_model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies,
                          train_negative_accuracies, test_positive_accuracies, test_negative_accuracies),
                         epoch, output_model_path)
            print('Saved epoch: {} as checkpoint'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FBankTripletLossNet model.')

    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--pretrained_model_path', type=str, default='siamese_fbanks_saved/', help='Path to the pretrained model')
    parser.add_argument('--output_model_path', type=str, default='siamese_fbanks_saved/', help='Path to save the trained model')
    parser.add_argument('--train_data', type=str, default='fbanks_train', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='fbanks_test', help='Path to testing data')

    args = parser.parse_args()

    main(args.num_layers, args.lr, args.epochs, args.batch_size, args.pretrained_model_path, 
         args.output_model_path, args.train_data, args.test_data)

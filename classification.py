from trainer.fbankcross_classification import train_classification, test_classification, inference_speaker_classification
from utils.pt_util import restore_objects, save_model, save_objects, restore_model
import torch
from data_proc.cross_entropy_dataset import FBanksCrossEntropyDataset, DataLoader
import json
from torch import optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from models.classifier import DynamicLinearClassifier 


async def train_csf(
    train_dataset_path: str = 'dataset-speaker-csf/fbanks-train',
    test_dataset_path: str = 'dataset-speaker-csf/fbanks-test',
    model_name: str = 'fbanks-net-classification',
    num_layers : int = 2 ,
    epoch: int = 2,
    lr: float = 0.0005,
    batch_size: int = 2,
    labId: str = '',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import multiprocessing
    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if torch.cuda.is_available() else {}
    try:
        train_dataset = FBanksCrossEntropyDataset(train_dataset_path)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test_dataset = FBanksCrossEntropyDataset(test_dataset_path)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    except:
        return 'path dataset test or train is not exist'

    try:

        assert train_dataset.num_classes == test_dataset.num_classes

    except:
        return "The number of speakers in test and training sets must be equal "
    if model_name == 'fbanks-net-classification':
        try:
            model = DynamicLinearClassifier(num_layers= num_layers,
                output_size=train_dataset.num_classes).to(device)
        except:
            print('cuda load is error')
            device = torch.device("cpu")
            model = DynamicLinearClassifier(num_layers = num_layers,
                output_size=train_dataset.num_classes).to(device)
    else:
        model = None
        return {"model not exist in lab"}
    model_path = f'./modelDir/{labId}/log_train/{model_name}/'
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(
        model_path, (0, 0, [], [], [], []))
    start = last_epoch + 1 if max_accuracy > 0 else 0

    models_path = []
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(start, epoch):
        train_loss, train_accuracy = train_classification(
            model, device, train_loader, optimizer, epoch, 500)
        test_loss, test_accuracy = test_classification(
            model, device, test_loader)
        print('After epoch: {}, train_loss: {}, test loss is: {}, train_accuracy: {}, '
              'test_accuracy: {}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            model_path = save_model(model, epoch, model_path)
            models_path.append(model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses,
                         train_accuracies, test_accuracies), epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))
    train_history = {
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "model_path": models_path
    }
    return {
        'history': json.dumps(train_history)
    }


async def test_csf(
    test_dataset_path: str = 'dataset-speaker-csf/fbanks-test',
    model_name: str = 'fbanks-net-classification',
    num_layers : int = 2,
    batch_size: int = 2,
    labId: str = '',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import multiprocessing
    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if torch.cuda.is_available() else {}
    try:
        test_dataset = FBanksCrossEntropyDataset(test_dataset_path)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    except:
        return 'path dataset test is not exist'
    model_folder_path = f'./modelDir/{labId}/log_train/{model_name}/{num_layers}/'
    for file in os.listdir(model_folder_path):
        if file.endswith(".pth"):
            model_path = os.path.join(model_folder_path, file)
    if model_name == 'fbanks-net-classification':
        try:
            model = DynamicLinearClassifier(num_layers=num_layers, output_size=test_dataset.num_classes)
            cpkt = torch.load(model_path)
            model.load_state_dict(cpkt)
            model.to(device)
        except:
            print('cuda load is error')
            device = torch.device("cpu")
            model = DynamicLinearClassifier(num_layers=num_layers,output_size=test_dataset.num_classes)
            cpkt = torch.load(model_path)
            model.load_state_dict(cpkt)
            model.to(device)
    else:
        model = None
        return {"model not exist in lab"}
    test_loss, accurancy_mean = test_classification(model, device, test_loader)
    print(accurancy_mean)
    return {
        'test_loss': test_loss,
        'test_accuracy': accurancy_mean
    }


def infer_csf(
    speech_file_path: str = './sample.wav',
    model_name: str = 'fbanks-net-classification',
    num_layers : int = 2,
    
    labId: str = '',
):
    model_folder_path = f'./modelDir/{labId}/log_train/{model_name}/'
    for file in os.listdir(model_folder_path):
        if file.endswith(".pth"):
            model_path = os.path.join(model_folder_path, file)
    rs = inference_speaker_classification(
        file_speaker=speech_file_path, model_path=model_path, num_layers = num_layers)
    return {
        "result": rs
    }
    
if __name__ == '__main__':
    result = infer_csf()
    print(result)
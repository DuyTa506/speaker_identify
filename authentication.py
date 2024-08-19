import glob
from predictions import get_embeddings, get_cosine_distance
from utils.pt_util import restore_objects, save_model, save_objects, restore_model
from utils.preprocessing import extract_fbanks
from models.cross_entropy_model import FBankCrossEntropyNetV2
from trainer.cross_entropy_train import test, train
import numpy as np
import torch
from data_proc.cross_entropy_dataset import FBanksCrossEntropyDataset, DataLoader
import json
from torch import optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


async def train_auth(
    train_dataset_path: str = 'dataset-speaker-csf/fbanks-train',
    test_dataset_path: str = 'dataset-speaker-csf/fbanks-test',
    model_name: str = 'fbanks-net-auth',
    model_layers : int = 2,
    epochs: int = 3,
    lr: float = 0.0005,
    batch_size: int = 16,
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
    if model_name == 'fbanks-net-auth':
        model = FBankCrossEntropyNetV2(num_layers= model_layers, reduction='mean').to(device)
    else:
        model = None
        return {"model not exist in lab"}
    pretrain_path = f"./weights/{model_layers}/"
    model_path = f'./modelDir/{labId}/log_train/{model_name}/{model_layers}/'
    chk_file = glob.glob(model_path + '*.pth')
    if chk_file:    
        model = restore_model(model, model_path, device)
    else:
        model = restore_model(model, pretrain_path, device)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(
        model_path, (0, 0, [], [], [], []))
    start = last_epoch + 1 if max_accuracy > 0 else 0

    models_path = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(start, epochs):
        train_loss, train_accuracy = train(
            model, device, train_loader, optimizer, epoch, 500)
        test_loss, test_accuracy = test(model, device, test_loader)
        print('After epoch: {}, train_loss: {}, test loss is: {}, train_accuracy: {}, '
              'test_accuracy: {}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            saved_path = save_model(model, epoch, model_path)
            models_path.append(saved_path)
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


async def test_auth(
        test_dataset_path: str = 'dataset-speaker-csf/fbanks-test',
        model_name: str = 'fbanks-net-auth',
        model_layers : int = 2,
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

    model_folder_path = f'./modelDir/{labId}/log_train/{model_name}/{model_layers}/'
    if not os.path.exists(model_folder_path):
        return {"Model not exist in lab"}
    for file in os.listdir(model_folder_path):
        if file.endswith(".pth"):           
            model_path = os.path.join(model_folder_path, file)
    if model_name == 'fbanks-net-auth':
        try:
                model = FBankCrossEntropyNetV2(num_layers=model_layers, reduction= "mean")
                cpkt = torch.load(model_path)
                model.load_state_dict(cpkt)
                model.to(device)
        except:
                print('cuda load is error')
                device = torch.device("cpu")
                model = FBankCrossEntropyNetV2(num_layers=model_layers,reduction= "mean")
                cpkt = torch.load(model_path)
                model.load_state_dict(cpkt)
                model.to(device)
    else:
        model = None
        return {"model not exist in lab"}
    test_loss, accurancy_mean = test(model, device, test_loader)

    return {
        'test_loss': test_loss,
        'test_accuracy': accurancy_mean
    }


async def infer_auth(
        speech_file_path: str = 'sample.m4a',
        model_name: str = 'fbanks-net-auth',
        model_layers : int = 2,
        name_speaker: str = 'DuyTa',
        threshold: float = 0.8,
        labId: str = '',
):
    speaker_path = f'./modelDir/{labId}/speaker/'
    dir_ = speaker_path + name_speaker
    if not os.path.exists(dir_):
        return {'message': 'name speaker is not exist,please add speaker'}

    model_folder_path = f'./modelDir/{labId}/log_train/{model_name}/{model_layers}/'
    for file in os.listdir(model_folder_path):
        if file.endswith(".pth"):           
            model_path = os.path.join(model_folder_path, file)
    if model_name == 'fbanks-net-auth':
        try:    
                device = torch.device("cuda")
                model = FBankCrossEntropyNetV2(num_layers=model_layers, reduction= "mean")
                cpkt = torch.load(model_path)
                model.load_state_dict(cpkt)
                model.to(device)
        except:
                print('cuda load is error')
                device = torch.device("cpu")
                model = FBankCrossEntropyNetV2(num_layers=model_layers,reduction= "mean")
                cpkt = torch.load(model_path, map_location=device)
                model.load_state_dict(cpkt)
                model.to(device)
    else:
        model = None
        return {"model not exist in lab"}
    
    fbanks = extract_fbanks(speech_file_path)
    obj_embeddings = get_embeddings(fbanks, model)

    stored_embedding = np.mean(get_embeddings(np.load(
        speaker_path + name_speaker + '/fbanks.npy'), model),axis=0).reshape(1,-1)
    
    distances = get_cosine_distance(obj_embeddings, stored_embedding)

    print('mean distances', np.mean(distances), flush=True)
    positives = distances < 0.45
    positives_mean = np.mean(positives)
    if positives_mean >= threshold:
        return {
            "positives_mean": positives_mean,
            "name_speaker": name_speaker,
            "auth": True,
        }
    else:
        return {
            "positives_mean": positives_mean,
            "name_speaker": name_speaker,
            "auth": False,
        }

if __name__ == '__main__':
    result = infer_auth()
    print(result)
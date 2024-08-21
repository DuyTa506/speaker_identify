import glob
from typing import Literal
from trainer.cross_entropy_train import test, train
from data_proc.cross_entropy_dataset import FBanksCrossEntropyDataset, DataLoader
from utils.pt_util import restore_objects, save_model, save_objects, restore_model
from speaker import load_data_speaker
from utils.preprocessing import extract_fbanks
from models.cross_entropy_model import FBankCrossEntropyNetV2
from predictions import get_embeddings
import faiss
import numpy as np
import json
import torch
from torch import optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


async def train_id(
    train_dataset_path: str = 'speaker_id\\dataset-speaker-auth\\fbanks_train',
    test_dataset_path: str = 'speaker_id\\dataset-speaker-auth\\fbanks_test',
    model_name: str = 'fbanks-net-identity',
    model_layers : int = 3,
    epoch: int = 10,
    lr: float = 0.0003,
    batch_size: int = 32,
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
    if model_name == 'fbanks-net-identity':
        model = FBankCrossEntropyNetV2(num_layers= model_layers,reduction='mean').to(device)
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
    for epoch in range(start, epoch):
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


async def test_id(
    test_dataset_path: str = 'dataset-speaker-csf/fbanks-test' ,
    model_name: str = 'fbanks-net-identity',
    model_layers : int =2,
    batch_size: int = 32,
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
    if model_name == 'fbanks-net-identity':
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
    print(accurancy_mean)
    return {
        'test_loss': test_loss,
        'test_accuracy': accurancy_mean
    }


async def infer_id(
    speech_file_path: str = 'vivos_sample\\15\\VIVOSDEV16_273.wav',
    model_name :str = "fbanks-net-identity",
    model_layers : int = 3,
    num_speaker: int = 3,
    distance_metric : Literal["l2", "cosine"] = "cosine",
    labId: str = '',
):  
    model_folder_path = f'./modelDir/{labId}/log_train/{model_name}/{model_layers}'
    for file in os.listdir(model_folder_path):
        if file.endswith(".pth"):
            model_path = os.path.join(model_folder_path, file)
    if model_name == 'fbanks-net-identity':
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
    
    fbanks = extract_fbanks(speech_file_path)
    embeddings = get_embeddings(fbanks, model)
    mean_embeddings = np.mean(embeddings, axis=0)
    mean_embeddings = mean_embeddings.reshape((1, -1)).astype(np.float32)
    encodes = []
    person_ids = []
    ##Embedding and cat the user's embeddings##
    rs = load_data_speaker(labId)
    for key, vectors in rs.items():
        for _, vector in vectors.items():
            vector =np.mean(get_embeddings(vector, model), axis=0).reshape((1,-1))
            encodes.append(np.array(vector, dtype=np.float32))
            person_ids.append(key)
    encodes = np.vstack(encodes).astype(np.float32)

    if distance_metric == "l2" :
        index = faiss.IndexFlatL2(encodes.shape[1]) 
               
    elif distance_metric == "cosine" :
        faiss.normalize_L2(mean_embeddings)
        faiss.normalize_L2(encodes) 
        index = faiss.index_factory(encodes.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    else :
        return {"Please choose distance stragegy between l2 distance and cosine distance"}
    index.add(encodes)
    distances, indices = index.search(mean_embeddings, k = num_speaker)
    rs_speaker = []    
    for i in range(num_speaker):
        # rs_speaker.append(f"speaker {i+1}: {person_ids[indices[0][i]]}, distances: {distances[0][i]}")
        rs_speaker.append({
            "speaker_name": person_ids[indices[0][i]],
            "distance": str( 1- distances[0][i]) if distance_metric == "cosine" else str(distances[0][i])
        })
    return {
        'result': rs_speaker
    }

if __name__ == '__main__':
    result = infer_id()
    print(result)
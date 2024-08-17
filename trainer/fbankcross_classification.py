import torch
from models import FBankCrossEntropyNet
import tqdm
import multiprocessing
import time
import numpy as np
from models import DynamicLinearClassifier
MODEL_PATH = './weights/triplet_loss_trained_model.pth'
model_instance = FBankCrossEntropyNet()
model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))

use_cuda = False
kwargs = {'num_workers': multiprocessing.cpu_count(),
            'pin_memory': True} if use_cuda else {}


def train_classification(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    accuracy = 0
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        x = model_instance(x)
        optimizer.zero_grad()
        out = model(x)
        loss = model.loss(out, y)

        with torch.no_grad():
            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == y))

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    accuracy_mean = (100. * accuracy) / len(train_loader.dataset)

    return np.mean(losses), accuracy_mean.item()




def test_classification(model, device, test_loader, log_interval=None):
    model.eval()
    losses = []

    accuracy = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
            x, y = x.to(device), y.to(device)
            x = model_instance(x)
            out = model(x)
            test_loss_on = model.loss(out, y).item()
            losses.append(test_loss_on)

            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == y))

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(x), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss = np.mean(losses)
    accuracy_mean = (100. * accuracy) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} , ({:.4f})%\n'.format(
        test_loss, accuracy, len(test_loader.dataset), accuracy_mean))
    return test_loss, accuracy_mean.item()



def speaker_probability(tensor):
    counts = {}
    total = 0
    for value in tensor:
        value = int(value)
        counts[value] = counts.get(value, 0) + 1
        total += 1

    probabilities = {}
    for key, value in counts.items():
        probabilities['speaker '+str(key)] = value / total

    return probabilities



def inference_speaker_classification(
        file_speaker,
        num_class=3,
        num_layers= 2,
        model_instance=model_instance,
        model_path='saved_models_cross_entropy_classification/0.pth'
        ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from utils.preprocessing import extract_fbanks
    fbanks = extract_fbanks(file_speaker)
    model = DynamicLinearClassifier(num_layers =num_layers ,output_size=num_class)
    cpkt = torch.load(model_path)
    model.load_state_dict(cpkt)
    model = model.double()
    model.to(device)
    model_instance = model_instance.double()
    model_instance.eval()
    model_instance.to(device)
    with torch.no_grad():
        x = torch.from_numpy(fbanks)
        embedings = model_instance(x.to(device))
        # print(embedings.shape)  
        # embedings=embedings.unsqueeze(0)
        output = model(embedings)
        output = torch.argmax(output,dim=-1) 
        speaker_pro = speaker_probability(output)
        print(speaker_pro)
    return speaker_pro


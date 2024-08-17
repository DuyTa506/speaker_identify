from torch import nn
from abc import abstractmethod

import torch

class FBankResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.network(x)
        out = out + x
        out = self.relu(out)
        return out
class FBankNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(256, 250)
        )

    @abstractmethod
    def forward(self, *input_):
        raise NotImplementedError('Call one of the subclasses of this class')


class FBankCrossEntropyNet(FBankNet):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        n = x.shape[0]
        out = self.network(x)
        out = out.reshape(n, -1)
        out = self.linear_layer(out)
        return out

    
    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val

class FBankNetV2(nn.Module):
    def __init__(self, num_layers=4, embedding_size = 250):
        super().__init__()
        layers = []
        in_channels = 1
        out_channels = 32

        for i in range(num_layers):
            #print("In: " ,in_channels )
            #print("Out: ", out_channels)
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=(5 - 1) // 2, stride=2))
            layers.append(FBankResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3))
            if i < num_layers - 1  :
                in_channels = out_channels
                out_channels *= 2
            #print("After in: " ,in_channels )
            #print("After Out: ", out_channels)
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.network = nn.Sequential(*layers)
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=embedding_size)
        )

    @abstractmethod
    def forward(self, *input_):
        raise NotImplementedError('Call one of the subclasses of this class')





class FBankCrossEntropyNetV2(FBankNetV2):
    def __init__(self, num_layers=3, reduction='mean'):
        super().__init__(num_layers=num_layers)
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        n = x.shape[0]
        out = self.network(x)
        out = out.reshape(n, -1)
        out = self.linear_layer(out)
        return out

    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val

def main():
    num_layers = 1
    model = FBankCrossEntropyNetV2(num_layers = num_layers, reduction='mean')
    print(model)
    input_data = torch.randn(8, 1, 64, 64)

    output = model(input_data)

    print("Output shape:", output.shape)
    labels = torch.randint(0, 250, (8,))

    loss = model.loss(output, labels)

    print("Loss:", loss.item())

if __name__ == "__main__":
    main()
import torch.nn as nn
import torch.nn.functional as F

#### Additional DynamicLinearClassifier Layer for training ####
class DynamicLinearClassifier(nn.Module):
    def __init__(self,output_size, input_size=250, num_layers=3, dropout_prob=0.5):
        super(DynamicLinearClassifier, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers=num_layers
        
        cls_layers = int(num_layers/2)
        layer_sizes = [int(input_size - i * (input_size - output_size) / (cls_layers + 1)) for i in range(1, cls_layers + 1)]

        self.hidden_layers.append(nn.Linear(input_size, layer_sizes[0]))
        self.batch_norms.append(nn.BatchNorm1d(layer_sizes[0]))

        for i in range(1, cls_layers):
            self.hidden_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i]))

        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.loss_layer = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val

class LinearClassifier(nn.Module):
    def __init__(self, output_size,input_size=250):
        super(LinearClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, 1)
        self.linear2 = nn.Linear(1,output_size)
        self.loss_layer = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, x):
        input = self.linear1(x)
        return self.linear2(input)
    
    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val
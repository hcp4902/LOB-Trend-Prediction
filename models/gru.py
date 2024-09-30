import torch
import torch.nn as nn

class GRU_model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU_model,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)
        self.name = "GRU"
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # define gru units]
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, hn = self.gru(x, h0)
        # print("Original Out shape: ",out.shape)
        out = out[:, -1, :]  # Use the last hidden state from the last layer
        # print("Last hidden state shape: ",out.shape)
        out = self.fc1(out)
        # print("Out shape: ",out.shape)
        prediction = torch.softmax(out, dim=1)
        # print("Prediction shape: ",prediction.shape)
        return prediction



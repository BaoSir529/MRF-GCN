import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn_model import GCN


class LSTM(nn.Module):
    def __init__(self, args, lstm_input_dim, lstm_hidden_dim, lstm_output_dim):
        super(LSTM, self).__init__()
        self.args = args
        self.lstm = torch.nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(lstm_hidden_dim, lstm_output_dim)
        pass

    def forward(self, x):  # x = n_point * width_dim
        r_out, (h_n, c_n) = self.lstm(x, None)
        output = self.linear(r_out).to(self.args.device)
        output = F.relu(output)
        return output  # output = n_point * gcn_input_dim


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.lstm = LSTM(self.args, self.args.width, self.args.lstm_hidden_dim, self.args.gcn_hidden_dim)
        self.gcn = GCN(self.args, self.args.gcn_input_dim, self.args.gcn_hidden_dim)
        self.linear = torch.nn.Linear(self.args.gcn_hidden_dim, 1, bias=True)

    def forward(self, inputs, graph):  # input = n_point * batch * width_dim
        lstm_feature = self.lstm(inputs)
        gcn_feature = self.gcn(lstm_feature, graph.expand(inputs.size(0), graph.size(1), graph.size(1)))
        logit = self.linear(gcn_feature)
        return logit

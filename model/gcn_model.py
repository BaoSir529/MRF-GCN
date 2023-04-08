import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class GCNlayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        demon = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / demon
        output = F.relu(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, args, gcn_input, gcn_hidden_dim):
        super(GCN, self).__init__()
        self.args = args
        self.GCN_layer = GCNlayer(gcn_input, gcn_hidden_dim).to(self.args.device)

    def forward(self, input, graph):
        output = self.GCN_layer(input, graph)
        return output


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gcn(nn.Module):
    def __init__(self, nfeatures, nclasses, hidden_size=16, nhidden_layers=0, dropout=0.0,
                 use_skip_connection=False):
        super().__init__()

        self.use_skip_connection = use_skip_connection
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        self.nhidden_layers = nhidden_layers

        self.input_layer = GraphConvolution(nfeatures, hidden_size, self.dropout)

        # do not use dropout in hidden layers
        self.hidden_layers = self.create_conv_sequence(self.nhidden_layers, self.hidden_size,
                                                       use_skip_connection=self.use_skip_connection)
        self.output_layer = GraphConvolution(hidden_size, nclasses, self.dropout)

    @staticmethod
    def create_conv_sequence(num_of_layers, hidden_size, use_skip_connection=False):
        seq_conv = nn.Sequential()
        for i in range(num_of_layers):
            seq_conv.add_module("Conv " + str(i),
                                GraphConvolution(hidden_size,
                                                 hidden_size,
                                                 use_skip_connection=use_skip_connection))
        return seq_conv

    def forward(self, features, adj_matrix):
        x = self.input_layer([features, adj_matrix])
        x = self.hidden_layers(x)
        output = self.output_layer(x)[0]
        return output


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, use_activation=True, bias=False,
                 use_skip_connection=False):
        super().__init__()

        self.dropout = dropout
        self.use_activation = use_activation
        self.use_skip_connection = use_skip_connection
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        # PyTorch initialization (see docs)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X):
        node_features, adj_matrix = X

        x = torch.mm(node_features, self.weight)
        x = torch.spmm(adj_matrix, x)

        # Add bias if necessary
        if self.bias is not None:
            x = x + self.bias

        # Apply activation
        if self.use_activation:
            x = F.relu(x)

        if self.use_skip_connection:
            x += node_features

        x = F.dropout(x, self.dropout, training=self.training)
        return [x, adj_matrix]
    
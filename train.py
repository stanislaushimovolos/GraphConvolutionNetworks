import torch
import torch.nn as nn
import torch.optim as optim

from utils import accuracy


def train(model, dataset, optimizer, epochs):
    adj, features, labels, idx_train, idx_val, idx_test = dataset
    
    model.train()
    loss = nn.CrossEntropyLoss()
    
    train_history = {"valid_loss": [],
                     "train_loss": [],
                     "valid_accuracy": [],
                     }
    
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = loss(output[idx_train], labels[idx_train])
        
        # save current loss
        train_history["train_loss"].append(loss_train.item())
        
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        output = model(features, adj)
        
        loss_val = loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        # save current loss and accuracy
        train_history["valid_loss"].append(loss_val.item())
        train_history["valid_accuracy"].append(acc_val.item())
   
    return train_history
    

def test(model, dataset):
    adj, features, labels, idx_train, idx_val, idx_test = dataset
    
    model.eval()
    loss = nn.CrossEntropyLoss()
    output = model(features, adj)
    
    loss_test = loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test
    
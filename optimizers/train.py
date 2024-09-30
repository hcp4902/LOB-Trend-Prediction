import torch
import yaml
import sys
import os
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from loaders.dataLoader import DataSet
from models.gru import GRU_model
from optimizers.batch_gd import batch_gd
from logs import logger


def __get_dataset__(model_id, dataset_type, auction, normalisation, T, k, stocks, train_till_days):

    if dataset_type == 'fi2010':
        print("Fetching Train data")
        dataset_train = DataSet('Training',auction,normalisation,train_till_days,stocks,T,k)
        print("Fetching Test data")
        dataset_test = DataSet('Testing',auction,normalisation,train_till_days,stocks,T,k)
    else:
        print("Error: wrong dataset type")

    
    print(f"Training Data Size : {dataset_train.__len__()}")
    print(f"Validation Data Size : {dataset_test.__len__()}")

    # update logs with dataset information
    dataset_info = {
        'dataset_type': dataset_type,
        'normalization': normalisation,
        'T': T,
        'k': k,
        'stock': stocks,
        'train_till_days': train_till_days
    }
    logger.logger(model_id, 'dataset_info', dataset_info)

    return dataset_train, dataset_test

def __get_hyperparams__(name):
    root_path = sys.path[0]
    with open(os.path.join(root_path, 'optimizers', 'hyperparameters.yaml'), 'r') as stream:
        hyperparams = yaml.safe_load(stream)
    return hyperparams[name]

def train(model_id, dataset_type, auction, normalisation, T, k, stocks, train_till_days, model_type):
    # get train and validation set
    dataset_train, dataset_test = __get_dataset__(model_id, dataset_type, auction, normalisation, T, k, stocks, train_till_days)
    
    if model_type == 'GRU':
        model = GRU_model(144, 128, 4, 3) #change input size(first parameter) according to no of features used for classification

    model.to(model.device)

    feature_size = 10

    summary(model) # , (1, 1, 100, feature_size)

    # Hyperparameter setting
    hyperparams = __get_hyperparams__(model.name)

    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epoch = hyperparams['epoch']
    num_workers = hyperparams['num_workers']

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loss_funtion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_gd(model_id = model_id, model = model, loss_funtion = loss_funtion, optimizer = optimizer,
             train_loader = train_loader, test_loader = test_loader, epochs=epoch)
    return

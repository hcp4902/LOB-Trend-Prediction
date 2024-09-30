from logs import logger
from optimizers import train

if __name__ == '__main__':
    # experiment parameter setting
    dataset_type = 'fi2010'
    normalisation = 'Zscore'
    auction = 'NoAuction'
    model_type = 'GRU'

    T = 10
    k = 10
    stocks = [0, 1, 2, 3, 4]
    train_till_days = 8

    # generate model id
    model_id = logger.generate_id(model_type)
    print(f"Model ID: {model_id}")

    
    train.train(model_id, dataset_type, auction, normalisation, T, k, stocks, train_till_days, model_type)


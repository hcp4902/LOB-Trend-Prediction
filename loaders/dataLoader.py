import os
import numpy as np
import torch

class DataSet:
    def __init__(self,mode,auction,normalisation,train_till_days,stocks,T,k):
        self.mode = mode
        self.auction = auction
        self.normalisation = normalisation
        self.train_till_days = train_till_days
        self.stocks = stocks
        self.T = T
        self.k = k

        x, y = self.__init_dataset__()
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)
        print("Unsqueezed shape X:",x.shape)
        print("Unsqueezed shape y:",y.shape)
        self.xshape = x.shape
        self.length = len(y)

    def __init_dataset__(self):
        x_cat = np.array([])
        y_cat = np.array([])

        for stock in self.stocks:
            print("Stock: ",stock)
            if self.mode == 'Training':
                # day = train-till-day -> conatins training data for all days before that day
                raw_data = __separate_stocks__(__read_raw__(self.mode,self.auction,self.normalisation,self.train_till_days),stock)
                x_raw,y_raw = __create_x_y__(raw_data)
                x,y = __create_data_label__(x_raw,y_raw,self.T,self.k,0.002)
                
                # append
                x_cat = __join__(x_cat,x)
                y_cat = __join__(y_cat,y)

            elif self.mode == 'Testing':
                # day >=train-till-day -> contains testing data for remaining days
                curr_day = self.train_till_days
                while curr_day < 10: # run over from curr_day to 9
                    raw_data = __separate_stocks__(__read_raw__(self.mode,self.auction,self.normalisation,curr_day),stock)
                    x_raw,y_raw = __create_x_y__(raw_data)
                    x,y = __create_data_label__(x_raw,y_raw,self.T,self.k,0.002)

                    # append
                    x_cat = __join__(x_cat,x)
                    y_cat = __join__(y_cat,y)

                    curr_day +=1

        return x_cat, y_cat
    
    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Fetch data, used while training"""
        return self.x[index], self.y[index]

def __read_raw__(mode,auction,normalisaton,day):
    """"
    Reading .txt file as ip

    mode: Training/Testing
    auction: Auction/NoAuction
    normalisation: {'Zscore', 'MinMax', 'DecPre'}
    day = {1,2,3 .... 9}

    """
    root = os.getcwd()
    data_path = "data"

    norm_path = ''
    if normalisaton == 'Zscore':
        norm_path = '1.'
    elif normalisaton == 'MinMax':
        norm_path = '2.'
    elif normalisaton == 'DecPre':
        norm_path = '3.'

    folder= f"{norm_path}{auction}_{normalisaton}"
    typefolder = auction+"_"+normalisaton+"_"+mode

    if mode == 'Training':
        file = f"Train_Dst_{auction}_{normalisaton}_CF_{str(day)}.txt"
    else:
        file = f"Test_Dst_{auction}_{normalisaton}_CF_{str(day)}.txt"
    
    file_path = os.path.join(root,data_path,auction,folder,typefolder,file)

    data = np.loadtxt(file_path)
    print(file_path)
    return data

def __separate_stocks__(raw_data,stock):
    """
    raw_data: array
    stock = {1,2,3,4,5}
    """

    boundary_count = 4 # split in 4 sections [5 parts] where there is a max GAP in consecutive difference indicating different stocks
    boundaries = np.sort(np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-boundary_count - 1:])
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(boundary_count + 1))
    return split_data[stock]

def __create_x_y__(raw_data):
    """
    raw_data: array
    Select features you want from first 144 features in dataset. Next 5 are labels
    Important features: No.86, No.88, No.127, No.87, No.89, No.92, No.1, No.128, No.93, and No.91 //[86,87,88,89,90,91,92,93,127,128]
    """

    x = raw_data[:144, :].T # derivatives of basic features
    # # Method -1 set y as midpoint price
    # y = (raw_data[0, :].T+raw_data[2, :].T)/2 # y = (BestAsk+BestBid)/2

    # Method -2 take y as provided by data
    y = raw_data[-5:, :].T

    # Method -3: First select Best Bid and Best Ask at each time
    return x, y

def __create_data_label__(x_raw,y_raw,T,k,alpha):
    """
    x_raw: raw input data
    y_raw: mid-point price
    T: window size of input data
    k: window size of estimation

    """

    [N, D] = x_raw.shape
    # print("y raw shape: ", y_raw.shape)
    # print("x_raw shape: ",N,"x",D)
    x = np.zeros((N-T+1,T,D))
    y = np.zeros(N-T+1)
    for i in range(T,N):
        x[i-T] = x_raw[i-T:i,:]
        # Custom Label
        sell_chnage = (x_raw[i,2]-x_raw[i+T,0])/x_raw[i,2]  # first get bid(while selling) then ask price(while buying)
        buy_chnage = (x_raw[i+T,2]-x_raw[i,0])/x_raw[i,0]  # first ask then bid price
        
        if sell_chnage > alpha:
            y[i-T] = 0 #sell now, buy later
        elif buy_chnage > alpha:
            y[i-T] = 2 #buy now, sell later
        else:
            y[i-T] = 1 #do nothing

        print("Input:",x[i-T])
        
    # # Labels provided in dataset for Method-2
    # y = y_raw[T - 1:N]
    # y = y[:, 4] - 1 # 5 columns, k = 1, 2, 3, 5, or 10
    return x,y


def __join__(old, new):
    if len(old) == 0:
        old = new
    else:
        old = np.concatenate((old, new), axis=0)
    return old  

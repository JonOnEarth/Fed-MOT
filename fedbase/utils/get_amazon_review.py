# generate amazon review dataset
import numpy as np
import os
import random
from scipy.sparse import coo_matrix
from os import path
from fedbase.utils.data_utils import split_data, group_split, split_dataset
import torch


# https://github.com/FengHZ/KD3A/blob/master/datasets/AmazonReview.py
def load_amazon(base_path):
    dimension = 5000
    amazon = np.load(path.join(base_path, "amazon.npz"))
    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :dimension]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    # Partition the data into four categories and for each category partition the data set into training and test set.
    data_name = ["books", "dvd", "electronics", "kitchen"]
    num_data_sets = 4
    data_insts, data_labels, num_insts = [], [], []
    for i in range(num_data_sets):
        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i + 1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :])
        num_insts.append(amazon_offset[i + 1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
        data_insts[i] = data_insts[i].todense().astype(np.float32)
        data_labels[i] = data_labels[i].ravel().astype(np.int64)
    return data_insts, data_labels

random.seed(1)
np.random.seed(1)
data_path = "data/AmazonReview/"
dir_path = "data/AmazonReview/"

train_size = 0.75

# Allocate data to users
def generate_AmazonReview(client_group=1, method='iid',alpha=0.5):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata"
    
    # Get AmazonReview data
    if not os.path.exists(root):
        os.makedirs(root)
        
    # Download the Amazon Review dataset if it doesn't exist
    amazon_file = os.path.join(root, 'amazon.npz')
    if not os.path.exists(amazon_file):
        zip_file = os.path.join(root, 'AmazonReview.zip')
        if not os.path.exists(zip_file):
            print("Downloading Amazon Review dataset...")
            file_id = '1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W'
            # Use proper Google Drive download URL with confirmation
            download_url = f'https://drive.usercontent.google.com/download?id={file_id}&confirm=t'
            os.system(f'wget --no-check-certificate "{download_url}" -O "{zip_file}"')
        
        # Extract the zip file
        if os.path.exists(zip_file):
            print("Extracting Amazon Review dataset...")
            os.system(f'unzip -o "{zip_file}" -d "{root}"')
            # Move files from extracted subdirectory to root
            extracted_dir = os.path.join(root, 'AmazonReview')
            if os.path.exists(extracted_dir):
                import shutil
                # Move all files from extracted_dir to root
                for item in os.listdir(extracted_dir):
                    src = os.path.join(extracted_dir, item)
                    dst = os.path.join(root, item)
                    shutil.move(src, dst)
                # Remove the empty extracted directory
                os.rmdir(extracted_dir)
            # Remove the zip file to save space
            os.remove(zip_file)

    X, y = load_amazon(root)

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    # statistic = [[] for _ in range(num_clients)]
    # for client in range(num_clients):
    #     for i in np.unique(y[client]):
    #         statistic[client].append((int(i), int(sum(y[client]==i))))
    # turn X and y to pytorch tensor
    for i in range(len(X)):
        X[i] = torch.from_numpy(X[i])
        y[i] = torch.from_numpy(y[i])
    train_data_groups, test_data_groups = split_data(X, y, train_size=train_size)
    train_data, test_data = group_split(train_data_groups, test_data_groups, client_group, method=method, alpha=alpha)
    
    return train_data, test_data, 'amazon' +'_'+ str(client_group)+'_'+ str(alpha)+'_'+ str(method)
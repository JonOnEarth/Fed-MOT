import numpy as np
import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data
# from utils.dataset_utils import split_data, save_file
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ChainDataset, ConcatDataset
from fedbase.utils.tools import get_targets
from itertools import accumulate as _accumulate
from sklearn.model_selection import train_test_split
from fedbase.utils.data_utils import split_data, group_split, split_dataset

train_size = 0.75
train_sample_size = 25000 #800 #25000
test_sample_size = 9000 #200 #9000
# https://github.com/FengHZ/KD3A/blob/master/datasets/DigitFive.py
def load_mnist(base_path):
    print("load mnist")
    mnist_data = loadmat(path.join(base_path, "mnist_data.mat"))
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    # turn to the 3 channel image with C*H*W
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train[:train_sample_size]
    train_label = train_label[:train_sample_size]
    mnist_test = mnist_test[:test_sample_size]
    test_label = test_label[:test_sample_size]
    return mnist_train, train_label, mnist_test, test_label


def load_mnist_m(base_path):
    print("load mnist_m")
    mnistm_data = loadmat(path.join(base_path, "mnistm_with_label.mat"))
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    mnistm_train = mnistm_train[:train_sample_size]
    train_label = train_label[:train_sample_size]
    mnistm_test = mnistm_test[:test_sample_size]
    test_label = test_label[:test_sample_size]
    return mnistm_train, train_label, mnistm_test, test_label


def load_svhn(base_path):
    print("load svhn")
    svhn_train_data = loadmat(path.join(base_path, "svhn_train_32x32.mat"))
    svhn_test_data = loadmat(path.join(base_path, "svhn_test_32x32.mat"))
    svhn_train = svhn_train_data['X']
    svhn_train = svhn_train.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_test = svhn_test_data['X']
    svhn_test = svhn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = svhn_train_data["y"].reshape(-1)
    test_label = svhn_test_data["y"].reshape(-1)
    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train = svhn_train[inds]
    train_label = train_label[inds]
    svhn_train = svhn_train[:train_sample_size]
    train_label = train_label[:train_sample_size]
    svhn_test = svhn_test[:test_sample_size]
    test_label = test_label[:test_sample_size]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return svhn_train, train_label, svhn_test, test_label


def load_syn(base_path):
    print("load syn")
    syn_train_data = loadmat(path.join(base_path, "synth_train_32x32.mat"))
    syn_test_data = loadmat(path.join(base_path, "synth_test_32x32.mat"))
    syn_train = syn_train_data["X"]
    syn_test = syn_test_data["X"]
    syn_train = syn_train.transpose(3, 2, 0, 1).astype(np.float32)
    syn_test = syn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = syn_train_data["y"].reshape(-1)
    test_label = syn_test_data["y"].reshape(-1)
    syn_train = syn_train[:train_sample_size]
    syn_test = syn_test[:test_sample_size]
    train_label = train_label[:train_sample_size]
    test_label = test_label[:test_sample_size]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return syn_train, train_label, syn_test, test_label


def load_usps(base_path):
    print("load usps")
    usps_dataset = loadmat(path.join(base_path, "usps_28x28.mat"))
    usps_dataset = usps_dataset["dataset"]
    usps_train = usps_dataset[0][0]
    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0
    usps_test = usps_dataset[1][0]
    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0
    usps_train = usps_train * 255
    usps_test = usps_test * 255
    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    usps_train = np.tile(usps_train, (4, 1, 1, 1))
    train_label = np.tile(train_label,4)
    usps_train = usps_train[:train_sample_size]
    train_label = train_label[:train_sample_size]
    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    return usps_train, train_label, usps_test, test_label

class Digit5Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(Digit5Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]
    

def digit5_dataset_read(base_path, domain):
    if domain == "mnist":
        train_image, train_label, test_image, test_label = load_mnist(base_path)
    elif domain == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m(base_path)
    elif domain == "svhn":
        train_image, train_label, test_image, test_label = load_svhn(base_path)
    elif domain == "syn":
        train_image, train_label, test_image, test_label = load_syn(base_path)
    elif domain == "usps":
        train_image, train_label, test_image, test_label = load_usps(base_path)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(28), #32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # raise train and test data loader
    train_dataset = Digit5Dataset(data=train_image, labels=train_label, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = Digit5Dataset(data=test_image, labels=test_label, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "data/Digit5/"
dir_path = "data/Digit5/"

# Allocate data to users
def generate_Digit5(domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn'], client_group=1, method='iid',alpha=0.5):
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
    
    # Get Digit5 data - improved download mechanism
    if not os.path.exists(root):
        os.makedirs(root)
    
    # Download individual datasets with proper error handling
    download_digit5_datasets(root)

    X, y = [], []
    # domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
    for d in domains:
        train_loader, test_loader = digit5_dataset_read(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))
    

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


    train_data_groups, test_data_groups = split_data(X, y, train_size=train_size)
    # split train data and test data into different clients
    train_data, test_data = group_split(train_data_groups, test_data_groups, client_group, method=method, alpha=alpha)

    # save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
    #     statistic, None, None, None)

    
    return train_data, test_data, 'digit5' +'_'+ str(client_group)+'_'+ str(alpha)+'_'+ str(method)


def download_digit5_datasets(root):
    """
    Download the complete Digit5 dataset from Google Drive
    """
    import urllib.request
    import zipfile
    import shutil
    import subprocess
    
    # Check if required files already exist
    required_files = [
        'mnist_data.mat',
        'mnistm_with_label.mat', 
        'svhn_train_32x32.mat',
        'svhn_test_32x32.mat',
        'synth_train_32x32.mat',
        'synth_test_32x32.mat',
        'usps_28x28.mat'
    ]
    
    all_exist = all(os.path.exists(os.path.join(root, f)) for f in required_files)
    if all_exist:
        print("All Digit5 dataset files already exist.")
        return
    
    print("Downloading Digit5 dataset...")
    
    # Try to download the complete Digit5 dataset from the new Google Drive link
    try:
        # Download the complete Digit5 dataset (7z file)
        digit5_7z = os.path.join(root, 'DigitFive.7z')
        if not os.path.exists(digit5_7z):
            print("Downloading Digit5 dataset from Google Drive...")
            
            # First, get the confirmation page to extract the UUID
            file_id = '1QvC6mDVN25VArmTuSHqgd7Cf9CoiHvVt'
            confirm_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0'
            
            try:
                import subprocess
                import re
                
                # Get the confirmation page
                result = subprocess.run([
                    'wget', '--no-check-certificate', '-O', digit5_7z, confirm_url
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    print(f"wget failed: {result.stderr}")
                    raise Exception("Download failed with wget")
                
                # Read the confirmation page and extract UUID
                with open(digit5_7z, 'r') as f:
                    html_content = f.read()
                
                # Check if we got HTML (confirmation page) or the actual file
                if 'html' in html_content.lower() and 'uuid' in html_content:
                    print("Got confirmation page, extracting UUID...")
                    # Extract UUID from the HTML
                    uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
                    if uuid_match:
                        uuid = uuid_match.group(1)
                        print(f"Found UUID: {uuid}")
                        
                        # Now download with the UUID
                        download_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0&confirm=t&uuid={uuid}'
                        
                        print("Downloading actual file...")
                        result = subprocess.run([
                            'wget', '--no-check-certificate', '-O', digit5_7z, download_url
                        ], capture_output=True, text=True, timeout=600)
                        
                        if result.returncode != 0:
                            print(f"wget failed: {result.stderr}")
                            raise Exception("Download failed with wget")
                        
                        print("Digit5 dataset downloaded successfully.")
                    else:
                        print("Could not find UUID in confirmation page")
                        raise Exception("Could not extract UUID from confirmation page")
                else:
                    print("Got file directly (no confirmation needed)")
                    
            except Exception as e:
                print(f"Failed to download: {e}")
                # Clean up any partial download
                if os.path.exists(digit5_7z):
                    os.remove(digit5_7z)
                raise e
        
        # Extract the 7z file if it exists
        if os.path.exists(digit5_7z):
            print("Extracting Digit5 dataset (7z file)...")
            try:
                # Use 7z to extract the file
                result = subprocess.run([
                    '7z', 'x', digit5_7z, f'-o{root}', '-y'
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"7z extraction failed: {result.stderr}")
                    raise Exception("7z extraction failed")
                    
                print("Digit5 dataset extracted successfully.")
                
                # Move files from extracted subdirectory to root if needed
                extracted_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in ['__MACOSX', '.', '..']]
                for extracted_dir in extracted_dirs:
                    extracted_path = os.path.join(root, extracted_dir)
                    if os.path.exists(extracted_path):
                        # Move all files from extracted directory to root
                        for item in os.listdir(extracted_path):
                            src = os.path.join(extracted_path, item)
                            dst = os.path.join(root, item)
                            if os.path.isfile(src) and not os.path.exists(dst):
                                shutil.move(src, dst)
                        # Remove empty directory if it's empty
                        try:
                            os.rmdir(extracted_path)
                        except:
                            pass
                            
                # Remove the 7z file to save space
                os.remove(digit5_7z)
                
            except Exception as e:
                print(f"7z extraction failed: {e}")
                print("Make sure 7z is installed: sudo apt-get install p7zip-full")
                raise e
        
        # Check if download was successful
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(root, f))]
        
        if missing_files:
            print(f"\nSome dataset files are still missing after download: {missing_files}")
            raise FileNotFoundError(f"Required dataset files are missing: {missing_files}")
                
    except Exception as e:
        print(f"Download failed: {e}")
        raise



if __name__ == "__main__":
    generate_Digit5(dir_path)
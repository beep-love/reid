import torch
import torchvision
import time
from tqdm import tqdm
from triplet_batch_dataset import VehicleTripletDataset
import os
from torch.utils.data import DataLoader
from triplet_loss import soft_margin_batch_hard_triplet_loss
# Device configuration for code and data

root_dir = '/home/biplav/AI_center/dataset/VERI-1.0/VERI-1'
list_file= 'train_test_split/train_list.txt'
info_file= 'train_test_split/vehicle_info.txt'

##########---------------------------------------------################
# Hyperparameters
# Batch size = P * K 
# By default, P = 8, K = 4 in vehicle triplet dataset! 
# Please modify the batch in dataset using P and K where,
# P = Number of identities in a batch
# K = Number of images per identity in a batch
##########---------------------------------------------################


class InfiniteDataloader:
    def __init__(self, dataset, num_workers=0, shuffle=False, batch_size=None):
        self.dataloader = DataLoader(dataset, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)
        self.data_iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)
        return data

def get_dataloaders():
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.RandomCrop((128, 128), padding=4, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.226, 0.226, 0.226])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.226, 0.226, 0.226])
    ])

    if os.path.isfile('train-pair-dataset.pt'):
        print('Loading pickled train dataset from train-pair-dataset.pt...')
        train_dataset = torch.load('train-pair-dataset.pt')
    else:
        start = time.time()
        print('\nPreparing training dataset. This may take a while...')
        train_dataset = VehicleTripletDataset(root_dir=root_dir, list_file=list_file, info_file= info_file, mode='train', transform=transform_train, P=8, K=4)
        torch.save(train_dataset, 'train-pair-dataset.pt')
        end = time.time()
        print("time:{:.2f}s".format(end - start))

    # if os.path.isfile('val-pair-dataset.pt'):
    #     print('Loading pickled validation dataset from val-pair-dataset.pt...')
    #     val_dataset = torch.load('val-pair-dataset.pt')
    
    # else:
    #     start = time.time()
    #     print('\nPreparing validation dataset. This may take a while...')
    #     val_dataset = VehicleTripletDataset(root_dir=root_dir, list_file=list_file, info_file= info_file, mode='val', transform=transform_test, P=8, K=4)
    #     torch.save(val_dataset, 'val-pair-dataset.pt')
    #     end = time.time()
    #     print("time:{:.2f}s".format(end - start))
    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=0, shuffle=False)
    # valloader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=0, shuffle=False)

    return trainloader #, valloader



def train(epoch, num_iters, net, trainloader, optimizer, device, interval):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    train_loss = 0.
    total = 0
    start = time.time()

    for it in tqdm(range(num_iters)):
        labels, anchors, positives, negatives = next(iter(trainloader))
        P, K = positives.shape[0], positives.shape[1]
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

        # Extract representative embeddings
        anchor_outputs = net(anchors.view(P * K, 3, *anchors.shape[2:]))
        positive_outputs = net(positives.view(P * K, 3, *positives.shape[2:]))
        negative_outputs = net(negatives.view(P * K, 3, *negatives.shape[2:]))

        # Calculate loss
        loss = soft_margin_batch_hard_triplet_loss(anchor_outputs, positive_outputs, negative_outputs, margin=0.1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate stats
        train_loss += loss.item()
        total += P * K

    end = time.time()
    print("time:{:.2f}s Loss:{:12.8g}".format(end - start, train_loss / total))
    return train_loss / total



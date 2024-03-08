import torch
import torchvision
import time
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import argparse
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

# Custom imports
from utils import load_weights, draw_curve, lr_decay, InfiniteDataloader
from model import Net
from triplet_loss import soft_margin_batch_hard_triplet_loss, soft_margin_batch_all_triplet_loss
from triplet_batch_dataset import VehicleTripletDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train on VERI WILD using triplet loss")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--wd", default=0.01, type=float)
    parser.add_argument("--interval", '-i', default=20, type=int)
    parser.add_argument('--resume', '-r', action='store_true')
    args = parser.parse_args()
    return args


# Device configuration for code and data
root_dir = '/home/biplav/AI_center/dataset/VERI-1.0/VERI-1/'  #IN MY PC # CHANGE FOR QNAP AND BAKENEKO
list_file= 'train_test_split/train_list_start0.txt'
info_file= 'train_test_split/vehicle_info.txt'

##########---------------------------------------------################
# Hyperparameters
# Batch size = P * K 
# By default, P = 8, K = 4 in vehicle triplet dataset! 
# Please modify the batch in dataset using P and K where,
# P = Number of identities in a batch
# K = Number of images per identity in a batch
##########---------------------------------------------################

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

    if os.path.isfile('val-pair-dataset.pt'):
        print('Loading pickled validation dataset from val-pair-dataset.pt...')
        val_dataset = torch.load('val-pair-dataset.pt')
    
    else:
        start = time.time()
        print('\nPreparing validation dataset. This may take a while...')
        val_dataset = VehicleTripletDataset(root_dir=root_dir, list_file=list_file, info_file= info_file, mode='val', transform=transform_test, P=8, K=4)
        torch.save(val_dataset, 'val-pair-dataset.pt')
        end = time.time()
        print("time:{:.2f}s".format(end - start))
    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=0, shuffle=False)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=0, shuffle=False)

    return trainloader , valloader

def train(epoch, num_iters, net, trainloader, optimizer, device, interval):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    train_loss = 0.
    total = 0
    start = time.time()

    for it in tqdm(range(num_iters)):
        labels, anchors, positives, negatives = next(iter(trainloader))  #[P*K, 3, 128, 128]
        # P, K = positives.shape[0], positives.shape[1]
        # print(P, K)
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        # print(anchors.shape, positives.shape, negatives.shape)

        anchor_outputs = net(anchors)
        positive_outputs = net(positives)
        negative_outputs = net(negatives)

        # Calculate loss
        loss = soft_margin_batch_hard_triplet_loss(anchor_outputs, positive_outputs, negative_outputs, margin=0.1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate stats
        train_loss += loss.item()
        total += positives.shape[0]

    end = time.time()
    print("time:{:.2f}s Loss:{:12.8g}".format(end - start, train_loss / total))
    return train_loss / total

def test(epoch, net, device, testloader, best_loss):
    net.eval()
    test_loss = 0.
    total = 0
    start = time.time()

    with torch.no_grad():
        print('Testing on validation set...')
        for labels, anchors, positives, negatives in tqdm(testloader):
            P, K = positives.shape[0], positives.shape[1]
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            # Feed forward
            anchor_outputs = net(anchors)
            positive_outputs = net(positives)
            negative_outputs = net(negatives)


            # Calculate loss
            loss = soft_margin_batch_all_triplet_loss(anchor_outputs, positive_outputs, negative_outputs)

            # Accumulate stats
            test_loss += loss.item()
            total += P * K

    end = time.time()
    avg_loss = test_loss / total
    print("Time: {:.2f}s, Loss: {:12.8g}".format(end - start, avg_loss))

    # Saving checkpoint
    if best_loss is None or avg_loss < best_loss:
        best_loss = avg_loss
        print('Saving parameters for epoch', epoch, 'to checkpoint/ckpt.reid_new.t7')
        checkpoint = {
            'net_dict': net.state_dict(),
            'loss': avg_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.reid_new.t7')
    else:
        print('No improvement of {:.6f} over {:.6f}'.format(avg_loss, best_loss))

    return avg_loss


def main():

    # Parse args and move to the right directory

    args = get_args()
    os.chdir(os.path.dirname(os.path.abspath( __file__)))  # Change directory to the location of this script

    # Identify CUDA device

    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True
    print('Device:', device)

    # Get model object

    num_pretrain_classes = 1261
    net = Net(num_classes=num_pretrain_classes, reid=True, square=True, embedding_size=128)
    net = torch.nn.DataParallel(net, device_ids=[4, 5, 6, 7])
    if args.resume:
        checkpoint = load_weights(net, './checkpoint/ckpt.finetune-pair-biplav-epoch55.t7')
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        best_epoch = start_epoch
    else:
        load_weights(net, './checkpoint/ckpt-mars.t7')
        best_loss = None
        start_epoch = 0
        
    net.to(device)

    # Set up optimizer

    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)
    # optimizer = torch.optim.RMSprop(net.parameters(), args.lr, weight_decay=args.wd, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    # Plot loss/accuracy figure
    x_epoch = []
    record = {'train_loss':[], 'test_loss':[]}
    fig = plt.figure()
    ax0 = fig.add_subplot(111, title="loss with margin 0.1")

    trainloader, testloader = get_dataloaders()

    # Run test set (validation set) before training to get initial loss
    num_train_iters = 50         # --> README.md explains why this is 5000
    
    test_loss = test(start_epoch-1, net, device, testloader, best_loss)

    if best_loss is None or test_loss < best_loss:
        best_loss = test_loss
        best_epoch = start_epoch
    draw_curve(start_epoch-1, None, test_loss, best_loss, best_epoch, record, x_epoch, ax0, fig)

    # Run 200 epochs

    for epoch in range(start_epoch, start_epoch+200):

        # Get data loaders in each epoch --> Ensure that the logic lets you get 
        # new data loaders in each epoch when there is existing saved data in the directory

        # trainloader, testloader = get_dataloaders()

        train_loss = train(epoch, num_train_iters, net, trainloader, optimizer, device, args.interval)
        print (f"{epoch+1}:{train_loss}")
        test_loss = test(epoch, net, device, testloader, best_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch+1
        draw_curve(epoch+1, train_loss, test_loss , best_loss, best_epoch, record, x_epoch, ax0, fig)
        if (epoch+1)%10==0:
            lr_decay(optimizer)

if __name__ == '__main__':
    main()

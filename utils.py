import os
import torch
from torch.utils.data import DataLoader

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


def load_weights(net, checkpoint_file):
    assert os.path.isfile(checkpoint_file), "Error: checkpoint file not found!"
    print('Loading from', checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    #checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda:0'))
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    return checkpoint

def draw_curve(epoch, train_loss, test_loss, best_loss, best_epoch, record, x_epoch, ax0, fig):
    record['train_loss'].append(train_loss)
    record['test_loss'].append(test_loss)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train BH')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val BA')
    for txt in ax0.texts:
        txt.remove()
    ax0.text(1, record['test_loss'][0], 'Best val loss %f @ epoch %d' % (best_loss, best_epoch))
    if epoch == 0:
        ax0.legend()
    fig.savefig("train-new-VERI.jpg")

# Reduce learning rate

def lr_decay(optimizer):
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))
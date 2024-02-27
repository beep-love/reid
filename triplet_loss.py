import torch
import torch.nn as nn
import sys

def soft_margin_batch_hard_triplet_loss(anchors, positives, negatives, margin=0.1):
    P = positives.shape[0]
    K = positives.shape[1]
    loss = 0.0
    for p in range(P):
        m1 = anchors[p,:,:]
        m2 = positives[p,:,:]
        m3 = negatives[p,:,:]
        # pos_dists = 1.0 - torch.matmul(m1, m1.transpose(0, 1))
        # neg_dists = 1.0 - torch.matmul(m1, m2.transpose(0, 1))
        pos_dists = torch.cdist(m1, m2)
        neg_dists = torch.cdist(m1, m3)
        pn_dists = torch.cdist(m2, m3)
        hardest_pos = torch.max(pos_dists, 1)[0]
        hardest_neg = torch.min(neg_dists, 1)[0]
        hardest_pn = torch.min(pn_dists, 1)[0]
        
        #=======================================================================================#
        # print(hardest_pos.min().item(), hardest_pos.max().item(), hardest_neg.min().item(), hardest_neg.max().item())
        #=======================================================================================#
        loss += torch.sum(torch.log1p(torch.exp(margin + 2*hardest_pos - hardest_neg - hardest_pn )))
        #=======================================================================================#
        
        if torch.isnan(loss):
            print('Got nan hardest pos:', hardest_pos, 'hardest neg:', hardest_neg, 'm2:', m2, 'neg_dists:', neg_dists)
            sys.exit()
    
    return loss

def soft_margin_batch_all_triplet_loss(anchors, positives, negatives):
    P = positives.shape[0]
    K = positives.shape[1]
    loss = 0
    for p in range(P):
        m1 = anchors[p,:,:]
        m2 = positives[p,:,:]
        m3 = negatives[p,:,:]

        # pos_dists = 1.0 - torch.matmul(m1, m1.transpose(0, 1))
        # neg_dists = 1.0 - torch.matmul(m1, m2.transpose(0, 1))

        pos_dists = torch.cdist(m1, m2)
        neg_dists = torch.cdist(m1, m3)
        pn_dists = torch.cdist(m2, m3)
        
        #print("pos_dists: ", pos_dists.shape)
        # sum over anchors a (sum over positives p (sum over negatives n ( D(a,p) - D(a,n) )))

        diffs = (2*pos_dists.repeat_interleave(K, dim=1) - neg_dists.repeat([1,K]) - pn_dists.repeat_interleave(K, dim=1) ).reshape([-1])
        
        #print("pos: ", pos_dists.repeat_interleave(K, dim=1).shape)
        #print("neg: ", neg_dists.repeat([1,K]).shape)
        #print("diffs: ", diffs.shape)

        loss += torch.sum(torch.log1p(torch.exp(diffs)))
        if torch.isnan(loss):
            print('Got nan pos_diffs:', diffs)
            sys.exit()
    return loss
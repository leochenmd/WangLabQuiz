import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pandas as pd

from datetime import datetime
from torch.autograd import Variable

from loader_stanford_vnet128x64n import load_data   #
from models.vnet import VNet
from models.loss.dice_loss import DiceLoss


## HYPERPARAMETERS
# learning_rate = 1e-3
# momentum = 0.9
# weight_decay = 1e-3
# epochs = 10
# dropout = 0   # 0.5
# batch_size = 4
# num_slices = 16
# crop_dim = 128



# https://arxiv.org/pdf/1707.03237.pdf
def generalized_dice_loss(logits, targets):
    probs = F.sigmoid(logits)
    
    batch_size = probs.size(0)
    probs = probs.view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    weight0 = 1 / (torch.sum(1-targets, dim=1)+1)**2 # +1 for numerical stability
    weight1 = 1 / (torch.sum(targets, dim=1)+1)**2
    
    numerator = weight0 * torch.sum((1-probs) * (1-targets)) + weight1 * torch.sum(probs * targets)
    denominator = weight0 * torch.sum((1-probs) + (1-targets)) + weight1 * torch.sum(probs + targets)

    loss = 1. - 2 * numerator / denominator

    loss = loss.mean()
    
    return loss

def run_model(model, loader, train=False, optimizer=None):
    labels = []
    
    dice_loss_fn = DiceLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0
    
    for batch in loader:
        if train:
            optimizer.zero_grad()

        vol, label, accession = batch
        if loader.dataset.args.gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)
        
        probs = torch.sigmoid(logit).cpu().detach().numpy()

#         loss = dice_loss_fn(logit, label) + bce_loss_fn(logit, label)
        loss = generalized_dice_loss(logit, label)
        
#         print('%08d' % label.sum(), '%08d' % preds.sum(), '%.4f' % preds.mean(), '%.4f' % loss.item(), '%.4f' % dice_loss_fn(logit, label).item())
        total_loss += loss.item()
        
        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1
        
    avg_loss = total_loss / num_batches

    return avg_loss

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--image_dir', default='/data/prostate/stanfordT2ADC_128x64n/', type=str)
    parser.add_argument('--label_dir', default='/data/prostate/stanfordlabelvnet/', type=str)
    parser.add_argument('--gpu', action='store_false') # defaults to True
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_slices', default=64, type=int)
    parser.add_argument('--crop_dim', default=128, type=int)
    return parser


def train(args):
    outdata = []
    
    if not os.path.exists(args.rundir):
        os.mkdir(args.rundir)
    
    with open(os.path.join(args.rundir, 'args.json'), 'w') as out:
        json.dump(vars(args), out, indent=4)
    
    train_loader, valid_loader, test_loader = load_data(args)
    
    model = VNet(model_depth=5, num_channels=2, dropout_prob=args.dropout, init_method='xavier')

    if args.gpu:
        model = model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)

    best_val_loss = float('inf')

    start_time = datetime.now()

    for epoch in range(args.epochs):
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
#         print(f'train AUC: {train_auc:0.4f}')

        val_loss = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
#         print(f'valid AUC: {val_auc:0.4f}')

        # scheduler.step(val_loss)

        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss

        file_name = f'epoch{epoch+1}_train{train_loss:0.4f}_val{val_loss:0.4f}'
        save_path = args.rundir + '/' + file_name
        
        torch.save(model.state_dict(), save_path)

        outdata.append(dict(zip(['epoch', 'train_loss', 'val_loss'], [epoch+1, train_loss, val_loss])))
        outdf = pd.DataFrame(outdata)
        outdf.to_csv(args.rundir + '/trainingcurve.csv')
            
if __name__=='__main__':
    args = get_parser().parse_args()
    train(args)

#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import datetime as dt
#Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
import random
import numpy as np
from dataset import CompositionDataset
best_auc = 0
best_hm = 0
best_attr = 0
best_obj = 0
best_seen = 0
best_unseen = 0
latest_changes = 0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random_seed = random.randint(0, 10000)
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

def main():
    # Get arguments and start logging
    now_time = dt.datetime.now().strftime('%F %T')
    print('new time is' + now_time)
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name+'_'+now_time)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only= args.train_only,
        open_world=args.open_world
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    train = train_normal

    evaluator_val =  Evaluator(testset, model)

    print(model)

    start_epoch = 0
    # Load checkpoint
    
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train(epoch, image_extractor, model, trainloader, optimizer)

        if epoch % args.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                test(epoch, image_extractor, model, testloader, evaluator_val, args, logpath)

    '''Final Outputs'''
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)
    print('best_seen', best_seen)
    print('best_unseen', best_unseen)
    print('best_attr', best_attr)
    print('best_obj', best_obj)
    print('Latest Changed Epoch is ', latest_changes)


def train_normal(epoch, image_extractor, model, trainloader, optimizer):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train() # Let's switch to training

    train_loss = 0.0 
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])
        loss, _ ,= model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
    train_loss = train_loss/len(trainloader)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, args, logpath):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, best_obj,best_attr,best_seen,best_unseen,latest_changes

    def save_checkpoint(filename):
        if image_extractor:
            resnet_save_path = os.path.join(logpath, 'Best_HM_ResNet_{}.pth'.format(args.dataset))
            torch.save(image_extractor.state_dict(),resnet_save_path)
        embedding_save_path = os.path.join(logpath, 'Best_HM_Embedding.pth')
        torch.save(model.state_dict(),embedding_save_path)

    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _, predictions = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch
    print(f'Test Epoch: {epoch}')
    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        latest_changes = epoch
        print('New best AUC ', best_auc)
    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        latest_changes = epoch
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')
    if stats['closed_attr_match'] > best_attr:
        best_attr = stats['closed_attr_match']
        latest_changes = epoch
        print('New Best attr', best_attr)
    if stats['closed_obj_match'] > best_obj:
        best_obj = stats['closed_obj_match']
        latest_changes = epoch
        print('best_obj', best_obj)
    if stats['best_seen'] > best_seen:
        best_seen = stats['best_seen']
        latest_changes = epoch
        print('New Best seen', best_seen)
    if stats['best_unseen'] > best_unseen:
        best_unseen = stats['best_unseen']
        latest_changes = epoch
        print('New Best unseen', best_unseen)
    print(latest_changes)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
        print('best_seen', best_seen)
        print('best_unseen', best_unseen)
        print('best_attr', best_attr)
        print('best_obj', best_obj)
        print('Latest Changed Epoch is', latest_changes)
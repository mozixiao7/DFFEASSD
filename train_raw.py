import argparse
import collections
import os
import json
import shutil
from ASV2021.raw_dataset import ASVspoof2019Raw
from dataset import ASVspoof2019Rawaug
from rawnet2.model import RawNet,setup_seed,Rawtf,RawNet2D
from rawnet2.model3d import Raw3D
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data2/tzc/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder",default='/data2/tzc/ocspoof/raw/3/contact/train_aug1/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=10, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=int, help="GPU index", default=7)
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    parser.add_argument('--add_loss', type=str, default="softmax",
                        help="loss for one-class training:softmax, amsoftmax,ocsoftmax'")

    # parser.add_argument('--continue_training', action='store_true', help="continue training with previously trained model")

    args = parser.parse_args()

    # Change this to specify GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

    # Path for input data
    assert os.path.exists(args.path_to_features)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")

    # assign device
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    # args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.set_default_tensor_type(torch.FloatTensor)
    dir_yaml = os.path.splitext('./rawnet2/model_config_RawNet2')[0] + '.yaml'
    print(dir_yaml)
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    print('cuda:',device)

    # initialize model
    raw_model = Raw3D(parser1['model'],device).to(device)


    # if args.continue_training:
    #     lfcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt')).to(args.device)

    lfcc_optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)

    training_set = ASVspoof2019Rawaug('train')
    validation_set = ASVspoof2019Rawaug('dev')
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=validation_set.collate_fn)

    criterion = nn.CrossEntropyLoss()

    early_stop_cnt = 0
    prev_eer = 1e8
    for epoch_num in tqdm(range(args.num_epochs)):
        raw_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        print('\nEpoch: %d ' % (epoch_num + 1))
        for i, (waveform, filename, tag, label) in enumerate(tqdm(trainDataLoader)):
            waveform = waveform.to(device)
            # print('wave.shape:',waveform.shape)
            label = label.to(device)
            batch_out = raw_model(waveform)
            lfcc_loss = criterion(batch_out, label)

            lfcc_optimizer.zero_grad()
            trainlossDict[args.add_loss].append(lfcc_loss.item())
            lfcc_loss.backward()
            lfcc_optimizer.step()


        # Val the model
        raw_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (waveform, filename, tag, label) in enumerate(tqdm(valDataLoader)):
                # print('wave.shape:', waveform.shape)
                # print('label.shape', label.shape)
                waveform = waveform.to(device)
                label = label.to(device)
                batch_out = raw_model(waveform)
                lfcc_loss = criterion(batch_out, label)
                score = F.softmax(batch_out, dim=1)[:, 0]

                devlossDict["softmax"].append(lfcc_loss.item())
                idx_loader.append(label)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            print("Val EER: {}".format(val_eer))

        torch.save(raw_model.state_dict(), os.path.join(args.out_fold, 'checkpoint',
                                            'anti-spoofing_raw_model_%d.pt' % (epoch_num + 1)))

        if val_eer < prev_eer:
            # Save the model checkpoint
            torch.save(raw_model.state_dict(), os.path.join(args.out_fold, 'anti-spoofing_raw_model.pt'))
            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            break

    return raw_model


if __name__ == "__main__":
    args = initParams()
    _ = train(args)
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_raw_model.pt'))


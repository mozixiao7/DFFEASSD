import argparse
import os
import json
import shutil
from resnet_fca import setup_seed, ResNet_FCA, ResNet_FCA_ELA
import pickle
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.FloatTensor)

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, path_to_protocol, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat'):
        self.access_type = access_type
        # self.ptd = path_to_database
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        # self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.access_type == 'LA':
            # self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
            #           "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
            #           "A19": 19}
            self.tag = {"-": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                        "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                        "A16": 16, "A17": 17,
                        "A18": 18,
                        "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        try:
            with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                 feat_mat = pickle.load(feature_handle)
            # with open(self.ptf + '/'+ filename  + '.npy', 'rb') as feature_handle:
            #     feat_mat = np.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            def the_other(train_or_dev):
                assert train_or_dev in ["train", "dev"]
                res = "train" if train_or_dev == "train" else "train"
                return res
            with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
            # print(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + '.pkl')
            # with open(os.path.join(self.path_to_features,self.part) + '/'+ filename + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
            # with open(self.ptf + '/'+ filename  + '.npy', 'rb') as feature_handle:
            #     feat_mat = np.load(feature_handle)

        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return feat_mat, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/tzc/ocspoof/pre/lfcc/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/data2/tzc/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", default='/data2/tzc/ocspoof/train_fca/am/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=120, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="14")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    parser.add_argument('--add_loss', type=str, default="softmax",
                        help="loss for one-class training:softmax, amsoftmax,ocsoftmax'")

    # parser.add_argument('--continue_training', action='store_true', help="continue training with previously trained model")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device('cuda:{}'.format(args.gpu))

    # initialize model
    lfcc_model = ResNet_FCA_ELA(3, args.enc_dim, resnet_type='18', nclasses=2).to(device)


    lfcc_optimizer = torch.optim.Adam(lfcc_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)

    training_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'train',
                                'LFCC', feat_len=args.feat_len, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'dev',
                                  'LFCC', feat_len=args.feat_len, padding=args.padding)
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               collate_fn=validation_set.collate_fn)

    # feat, _, _, _ = training_set[29]

    criterion = nn.CrossEntropyLoss()


    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        lfcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)

        print('\nEpoch: %d ' % (epoch_num + 1))
        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(trainDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            feats, lfcc_outputs = lfcc_model(lfcc)
            lfcc_loss = criterion(lfcc_outputs, labels)

            lfcc_optimizer.zero_grad()
            trainlossDict[args.add_loss].append(lfcc_loss.item())
            lfcc_loss.backward()
            lfcc_optimizer.step()

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(np.nanmean(trainlossDict[monitor_loss])) + "\n")

        # Val the model
        lfcc_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(valDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(device)
                labels = labels.to(device)

                feats, lfcc_outputs = lfcc_model(lfcc)

                lfcc_loss = criterion(lfcc_outputs, labels)
                score = F.softmax(lfcc_outputs, dim=1)[:, 0]

                devlossDict["softmax"].append(lfcc_loss.item())
                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) +"\n")
            print("Val EER: {}".format(val_eer))

        torch.save(lfcc_model, os.path.join(args.out_fold, 'checkpoint',
                                            'anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))


        if val_eer < prev_eer:
            # Save the model checkpoint
            torch.save(lfcc_model, os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))

            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            break

    return lfcc_model


if __name__ == "__main__":
    args = initParams()
    _ = train(args)
    model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))



import random

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
from torch.utils.data.dataloader import default_collate
import librosa
import soundfile as sf
import audioread
from torch.utils.data import DataLoader
from RawBoost.RawBoost import LnL_convolutive_noise,ISD_additive_noise,SSI_additive_noise
from feature_exactor import process_aug
import warnings
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

class ASVspoof2021LAeval(Dataset):
    def __init__(self, path_to_features="/dataNVME/neil/ASVspoof2021LAFeatures", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021LAeval, self).__init__()
        self.path_to_features = path_to_features
        # print(self.path_to_features)
        self.ptf = path_to_features
        # print(self.ptf)
        self.feat_len = feat_len
        # print(self.feat_len)
        self.feature = feature
        # print(self.feature)
        self.pad_chop = pad_chop
        # print(self.pad_chop)
        self.padding = padding
        # print(self.padding)
        protocol = '/data2/tzc/LA2021eval/ASVspoof2021.LA.cm.eval.trl.txt'
        with open(protocol,'r') as f:
            self.all_files = [info.strip().split() for info in f.readlines()]
        print('eval speech number:{}'.format(len(self.all_files)))
        # print(self.all_files)
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17,"A18": 18,"A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        # print(filepath)
        basename = os.path.basename(filepath[1])
        # print(basename)
        assert len(filepath) == 4
        file_path = os.path.join(self.path_to_features+basename+'LFCC.pkl')
        # print(filepath)
        with open(file_path,'rb') as f:
            featurenumpy= pickle.load(f)
        # print('featurenumpy:{}'.format(featurenumpy.shape),type(featurenumpy))
        featureTensor = torch.from_numpy(featurenumpy)
        # print(featureTensor.shape)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = basename
        # print(featureTensor.shape)
        return featureTensor, filename,self.tag[filepath[2]],self.label[filepath[3]]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)

class ASVspoof2021evalRaw(Dataset):
    def __init__(self, path_to_database="/data2/tzc/LA2021eval/flac/"):
        super(ASVspoof2021evalRaw, self).__init__()
        self.ptd = path_to_database
        self.path_to_audio = self.ptd
        protocol = '/data2/tzc/LA2021eval/ASVspoof2021.LA.cm.eval.trl.txt'
        with open(protocol,'r') as f:
            self.all_files = [info.strip().split() for info in f.readlines()]
        # self.all_files = librosa.util.find_files(self.path_to_audio, ext="flac")
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17, "A18": 18, "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        filedir = os.path.join(self.ptd+filepath[1]+'.flac')
        waveform, sr = torchaudio_load(filedir)
        filename = filepath[1]
        feat_mat = waveform
        # print(feat_mat.shape)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > 64000:
            startp = np.random.randint(this_feat_len - 64000)
            feat_mat = feat_mat[:, startp:startp + 64000]
        if this_feat_len < 64000:
            feat_mat = repeat_padding(feat_mat, 64000)
        return feat_mat, filename,self.tag[filepath[2]],self.label[filepath[3]]

    def collate_fn(self, samples):
        return default_collate(samples)

class ASVspoof2021DFeval(Dataset):
    def __init__(self, path_to_features="/dataNVME/neil/ASVspoof2021LAFeatures", feature='LFCC', feat_len=750, pad_chop=True, padding='repeat'):
        super(ASVspoof2021DFeval, self).__init__()
        self.path_to_features = path_to_features
        # print(self.path_to_features)
        self.ptf = path_to_features
        # print(self.ptf)
        self.feat_len = feat_len
        # print(self.feat_len)
        self.feature = feature
        # print(self.feature)
        self.pad_chop = pad_chop
        # print(self.pad_chop)
        self.padding = padding
        # print(self.padding)
        protocol = '/data2/tzc/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt'
        with open(protocol,'r') as f:
            self.all_files = [info.strip().split() for info in f.readlines()]
        print('eval speech number:{}'.format(len(self.all_files)))
        # print(self.all_files)
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17,"A18": 18,"A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        # print(filepath)
        basename = os.path.basename(filepath[1])
        # print(basename)
        assert len(filepath) == 4
        file_path = os.path.join(self.path_to_features+basename+'LFCC.pkl')
        # print(filepath)
        with open(file_path,'rb') as f:
            featurenumpy= pickle.load(f)
        # print('featurenumpy:{}'.format(featurenumpy.shape),type(featurenumpy))
        featureTensor = torch.from_numpy(featurenumpy)
        # print(featureTensor.shape)
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        filename = basename
        # print(featureTensor.shape)
        return featureTensor, filename,filepath[2],self.label[filepath[3]]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)

class ASVspoof2021DFevalRaw(Dataset):
    def __init__(self, path_to_database="/data2/tzc/ASVspoof2021_DF_eval/ASVspoof2021_DF_eval/flac/"):
        super(ASVspoof2021DFevalRaw, self).__init__()
        self.ptd = path_to_database
        self.path_to_audio = self.ptd
        protocol = '/data2/tzc/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt'
        with open(protocol,'r') as f:
            self.all_files = [info.strip().split()  for info in f.readlines()]
        # self.all_files = librosa.util.find_files(self.path_to_audio, ext="flac")
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17, "A18": 18, "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        filedir = os.path.join(self.ptd+filepath[1]+'.flac')
        waveform, sr = torchaudio_load(filedir)
        filename = filepath[1]
        feat_mat = waveform
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > 64000:
            startp = np.random.randint(this_feat_len - 64000)
            feat_mat = feat_mat[:, startp:startp + 64000]
        if this_feat_len < 64000:
            feat_mat = repeat_padding(feat_mat, 64000)
        return feat_mat, filename,filepath[2],self.label[filepath[3]]

    def collate_fn(self, samples):
        return default_collate(samples)

class ASVspoof2019aug(Dataset):
    def __init__(self,part='train', feature='LFCC', feat_len=750, padding='repeat'):
        self.tag = {"-": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17,"A18": 18,"A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.padding = padding
        self.feat_len = feat_len
        self.feature = feature
        self.part = part
        self.total_audio_info = []
        self.feature_path = '/data2/tzc/ocspoof/pre/lfcc'
        origion_protocol = '/data2/tzc/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trl.txt'.format(self.part)
        aug_protocal = '/data2/tzc/ocspoof/pre/lfcc/{}_aug.txt'.format(self.part)
        with open(origion_protocol, 'r') as f:
            origion_audio_info = [info.strip().split() for info in f.readlines()]
            self.origion_audio_info = origion_audio_info
        with open(aug_protocal,'r') as f:
            aug_audio_info = [info.strip().split() for info in f.readlines()]
            self.aug_audio_info = aug_audio_info
        self.total_audio_info = self.origion_audio_info+self.aug_audio_info
        # random.shuffle(self.total_audio_info)
        # self.total_audio_info = self.total_audio_info[:len(self.origion_audio_info)]
        print("{} dataset numbers:{}".format(self.part,len(self.total_audio_info)))
    def __len__(self):
        return len(self.total_audio_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.total_audio_info[idx]
        if filename.endswith('aug'):
            with open(self.feature_path + '/'+ self.part +'_aug' + '/' +filename + self.feature + '.pkl', 'rb') as feature_handle:
                 feat_mat = pickle.load(feature_handle)
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
        else:
            with open(self.feature_path + '/'+self.part + '/' +filename + self.feature + '.pkl', 'rb') as feature_handle:
                 feat_mat = pickle.load(feature_handle)
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

class ASVspoof2019Rawaug(Dataset):
    def __init__(self,part='train'):
        super(ASVspoof2019Rawaug, self).__init__()
        params = {}
        params['nBands'] = 5
        params['minF'] = 20
        params['maxF'] = 8000
        params['minBW'] = 100
        params['maxBW'] = 1000
        params['minCoeff'] = 10
        params['maxCoeff'] = 100
        params['minG'] = 0
        params['maxG'] = 0
        params['minBiasLinNonLin'] = 5
        params['maxBiasLinNonLin'] = 20
        params['N_f'] = 5
        params['P'] = 10
        params['g_sd'] = 2
        params['SNRmin'] = 10
        params['SNRmax'] = 40
        self.params = params
        self.part = part
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                  "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                  "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.total_audio_info = []
        origion_protocol = '/data2/tzc/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trl.txt'.format(self.part)
        aug_protocal = '/data2/tzc/ocspoof/pre/lfcc/{}_aug.txt'.format(self.part)
        with open(origion_protocol, 'r') as f:
            origion_audio_info = [info.strip().split() for info in f.readlines()]
            self.origion_audio_info = origion_audio_info
        with open(aug_protocal, 'r') as f:
            aug_audio_info = [info.strip().split() for info in f.readlines()]
            self.aug_audio_info = aug_audio_info
        self.total_audio_info = self.origion_audio_info + self.aug_audio_info
        # self.total_audio_info = self.aug_audio_info

    def __len__(self):
        return len(self.total_audio_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.total_audio_info[idx]
        if filename.endswith('aug'):
            filepath = os.path.join('/data2/tzc/LA/ASVspoof2019_LA_{}/flac/'.format(self.part), filename[:-4] + ".flac")
            waveform, sr = torchaudio_load(filepath)
            feat_mat = process_aug(waveform,sr,self.params)
            feat_mat = torch.Tensor(np.expand_dims(feat_mat, axis=0))
            this_feat_len = feat_mat.shape[1]
            if this_feat_len > 64000:
                startp = np.random.randint(this_feat_len - 64000)
                feat_mat = feat_mat[:, startp:startp + 64000]
            if this_feat_len < 64000:
                feat_mat = repeat_padding(feat_mat, 64000)
            return feat_mat, filename, self.tag[tag], self.label[label]
        else:
            filepath = os.path.join('/data2/tzc/LA/ASVspoof2019_LA_{}/flac/'.format(self.part), filename + ".flac")
            waveform, sr = torchaudio_load(filepath)
            feat_mat = torch.Tensor(np.expand_dims(waveform, axis=0))
            this_feat_len = feat_mat.shape[1]
            if this_feat_len > 64000:
                startp = np.random.randint(this_feat_len - 64000)
                feat_mat = feat_mat[:, startp:startp + 64000]
            if this_feat_len < 64000:
                feat_mat = repeat_padding(feat_mat, 64000)

            return feat_mat, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)


def torchaudio_load(filepath):
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        waveform, sr = librosa.load(filepath, sr=16000)
    except:
        # print(filepath)
        waveform, sr = sf.read(filepath)
        # wave, sr = audioread.audio_open(filepath)
        # print(sr == 16000)
    waveform = torch.Tensor(np.expand_dims(waveform, axis=0))
    return [waveform, sr]

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


if __name__ == "__main__":
    from tqdm import tqdm
    # test_set = ASVspoof2021evalRaw()
    # test_set = ASVspoof2019aug('train')
    test_set = ASVspoof2019Rawaug('train')
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0,collate_fn=test_set.collate_fn)
    for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
        print(i,lfcc.shape,audio_fn,tags,labels)
        # print(i)
        # j = 1


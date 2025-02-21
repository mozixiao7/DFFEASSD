import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
from torch.utils.data.dataloader import default_collate, DataLoader
from tqdm import tqdm
import soundfile as sf
import warnings

torch.set_default_tensor_type(torch.FloatTensor)

class ASVspoof2019Two(Dataset):
    def __init__(self, access_type, path_to_features,path_to_database,path_to_protocol, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat'):
        self.access_type = access_type
        # self.ptd = path_to_database
        self.path_to_features = path_to_features
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, access_type,'ASVspoof2019_' + access_type + '_' + self.part + '/flac/')
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
                 feat_mat_lfcc = pickle.load(feature_handle)
            # with open(self.ptf + '/'+ filename  + '.npy', 'rb') as feature_handle:
            #     feat_mat = np.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            def the_other(train_or_dev):
                assert train_or_dev in ["train", "dev"]
                res = "train" if train_or_dev == "train" else "train"
                return res
            with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
            # with open(self.ptf + '/'+ filename  + '.npy', 'rb') as feature_handle:
            #     feat_mat = np.load(feature_handle)

        feat_mat_lfcc = torch.from_numpy(feat_mat_lfcc)
        this_feat_len_lfcc = feat_mat_lfcc.shape[1]
        if this_feat_len_lfcc > self.feat_len:
            startp = np.random.randint(this_feat_len_lfcc-self.feat_len)
            feat_mat_lfcc = feat_mat_lfcc[:, startp:startp+self.feat_len]
        if this_feat_len_lfcc < self.feat_len:
            if self.padding == 'zero':
                feat_mat_lfcc = padding(feat_mat_lfcc, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat_lfcc = repeat_padding(feat_mat_lfcc, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        filepath_wave = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath_wave)
        feat_mat_wave = waveform
        this_feat_len_wave = feat_mat_wave.shape[1]
        if this_feat_len_wave > 64000:
            startp = np.random.randint(this_feat_len_wave - 64000)
            feat_mat_wave = feat_mat_wave[:, startp:startp + 64000]
        if this_feat_len_wave < 64000:
            feat_mat_wave = repeat_padding(feat_mat_wave, 64000)

        return feat_mat_lfcc,feat_mat_wave ,filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)

class ASVspoof2021Two(Dataset):
    def __init__(self, path_to_features='/data2/tzc/ocspoof/pre/lfcc/eval2021/', feature='LFCC', feat_len=750,
                 path_to_database="/data2/tzc/LA2021eval/flac/",pad_chop=True, padding='repeat'):
        super(ASVspoof2021Two, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.path_to_database = path_to_database
        protocol = '/data2/tzc/LA2021eval/ASVspoof2021total.LA.cm.eval.trl.txt'
        with open(protocol, 'r') as f:
            self.all_files = [info.strip().split() for info in f.readlines()]
        print('eval speech number:{}'.format(len(self.all_files)))
        # print(self.all_files)
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17, "A18": 18, "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath[1])
        assert len(filepath) == 4
        file_path = os.path.join(self.path_to_features + basename + 'LFCC.pkl')
        with open(file_path, 'rb') as f:
            featurenumpy = pickle.load(f)
        featureTensor = torch.from_numpy(featurenumpy)
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
        filedir = os.path.join(self.path_to_database + filepath[1] + '.flac')
        waveform, sr = torchaudio_load(filedir)
        filename = filepath[1]
        raw_mat = waveform
        this_raw_len = raw_mat.shape[1]
        if this_raw_len > 64000:
            startp = np.random.randint(this_raw_len - 64000)
            raw_mat = raw_mat[:, startp:startp + 64000]
        if this_raw_len < 64000:
            raw_mat = repeat_padding(raw_mat, 64000)
        return featureTensor,raw_mat,filename, self.tag[filepath[2]], self.label[filepath[3]]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)

class ASVspoofDFTwo(Dataset):
    def __init__(self, path_to_features='/data2/tzc/ocspoof/pre/lfcc/DF/', feature='LFCC', feat_len=750,
                 path_to_database="/data2/tzc/ASVspoof2021_DF_eval/ASVspoof2021_DF_eval/flac/",pad_chop=True, padding='repeat'):
        super(ASVspoofDFTwo, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.path_to_database = path_to_database
        protocol = '/data2/tzc/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.score.txt'
        with open(protocol, 'r') as f:
            self.all_files = [info.strip().split() for info in f.readlines()]
        print('eval speech number:{}'.format(len(self.all_files)))
        # print(self.all_files)
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17, "A18": 18, "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath[1])
        assert len(filepath) == 4
        file_path = os.path.join(self.path_to_features + basename + 'LFCC.pkl')
        with open(file_path, 'rb') as f:
            # featurenumpy = pickle.load(f)
            featureTensor=pickle.load(f)
        # featureTensor = torch.from_numpy(featurenumpy)
        featureTensor = featureTensor.squeeze(0)
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
        filedir = os.path.join(self.path_to_database + filepath[1] + '.flac')
        waveform, sr = torchaudio_load(filedir)
        filename = filepath[1]
        raw_mat = waveform
        this_raw_len = raw_mat.shape[1]
        if this_raw_len > 64000:
            startp = np.random.randint(this_raw_len - 64000)
            raw_mat = raw_mat[:, startp:startp + 64000]
        if this_raw_len < 64000:
            raw_mat = repeat_padding(raw_mat, 64000)
        return featureTensor,raw_mat,filename, filepath[2], self.label[filepath[3]]

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)

class IWDTwo(Dataset):
    def __init__(self, path_to_features='/data2/tzc/ocspoof/pre/lfcc/IWD/', feature='LFCC', feat_len=750,
                 path_to_database="/data2/tzc/release_in_the_wild/release_in_the_wild/",pad_chop=True, padding='repeat'):
        super(IWDTwo, self).__init__()
        self.path_to_features = path_to_features
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.path_to_database = path_to_database
        protocol = '/data2/tzc/release_in_the_wild/IWDdataset.txt'
        with open(protocol, 'r') as f:
            self.all_files = [info.strip().split() for info in f.readlines()]
        print('eval speech number:{}'.format(len(self.all_files)))
        # print(self.all_files)
        self.tag = {"bonafide": 20, "A01": 0, "A02": 1, "A03": 4, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15,
                    "A16": 16, "A17": 17, "A18": 18, "A19": 19}
        self.label = {"spoof": 1, "bona-fide": 0}

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        # print(filepath)
        basename = os.path.basename(filepath[1]).split('.')[0]
        # assert len(filepath) == 4
        file_path = os.path.join(self.path_to_features +'IWD'+basename + 'LFCC.pkl')
        with open(file_path, 'rb') as f:
            # featurenumpy = pickle.load(f)
            featureTensor=pickle.load(f)
        # featureTensor = torch.from_numpy(featurenumpy)
        featureTensor = featureTensor.squeeze(0)
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
        filedir = os.path.join(self.path_to_database + filepath[1])
        waveform, sr = torchaudio_load(filedir)
        filename = filepath[1].split('.')[0]
        raw_mat = waveform
        this_raw_len = raw_mat.shape[1]
        if this_raw_len > 64000:
            startp = np.random.randint(this_raw_len - 64000)
            raw_mat = raw_mat[:, startp:startp + 64000]
        if this_raw_len < 64000:
            raw_mat = repeat_padding(raw_mat, 64000)
        return featureTensor,raw_mat,filename, filepath[2], self.label[filepath[-1]]

    def collate_fn(self, samples):
        if self.pad_chop:
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
def torchaudio_load(filepath):
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        wave, sr = librosa.load(filepath, sr=16000)
    except:
        # print(filepath)
        wave, sr = sf.read(filepath)
        # print(sr == 16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

if __name__ == "__main__":
    training_set = ASVspoof2019Two(access_type='LA', path_to_features='/data2/tzc/ocspoof/pre/lfcc/',path_to_database='/data2/tzc/',
                                   path_to_protocol='/data2/tzc/LA/ASVspoof2019_LA_cm_protocols/', part='train')
    # waveform,filename,tag,label = training_set[0]
    # print(waveform.shape)

    trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True)
    # print('0')
    for i, (lfcc,waveform, filename, tag, label) in enumerate(tqdm(trainDataLoader)):
        #     print(type(waveform))
        #     print('waveform',waveform)
        print('lfcc.shape',lfcc.shape)
        print('waveform.shape', waveform.shape)
        print('label.shape', label.shape)


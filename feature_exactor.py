import pickle
import librosa
import numpy as np
import torch
from LFCC.feature_extraction import LFCC
# from RawBoost.data_utils_rawboost import process_Rawboost_feature
from RawBoost.RawBoost import LnL_convolutive_noise,ISD_additive_noise,SSI_additive_noise

def process_aug(wave,sr,params):
    wave = LnL_convolutive_noise(wave, params['N_f'],params['nBands'], params['minF'], params['maxF'], params['minBW'], params['maxBW'],
                                    params['minCoeff'], params['maxCoeff'], params['minG'], params['maxG'], params['minBiasLinNonLin'],
                                    params['maxBiasLinNonLin'], sr)
    wave = ISD_additive_noise(wave, params['P'], params['g_sd'])
    wave = SSI_additive_noise(wave, params['SNRmin'], params['SNRmax'],params['nBands'], params['minF'],
                                 params['maxF'], params['minBW'], params['maxBW'], params['minCoeff'], params['maxCoeff'], params['minG'], params['maxG'],
                                 sr)
    return wave
if __name__ == "__main__":
    params = {}
    params['nBands']=5
    params['minF']=20
    params['maxF']=8000
    params['minBW']=100
    params['maxBW']=1000
    params['minCoeff']=10
    params['maxCoeff']=100
    params['minG']=0
    params['maxG']=0
    params['minBiasLinNonLin']=5
    params['maxBiasLinNonLin']=20
    params['N_f'] = 5
    params['P'] =10
    params['g_sd'] =2
    params['SNRmin'] =10
    params['SNRmax'] =40
    train_dir = '/data2/tzc/ocspoof/pre/lfcc/train_aug.txt'
    dev_dir = '/data2/tzc/ocspoof/pre/lfcc/dev_aug.txt'
    with open(dev_dir, 'r') as f:
        infos = [info.strip().split() for info in f.readlines()]
    # print(infos[0])
    # print(infos[0][1][:-4])
    for i, info in enumerate(infos):
        wav, sr = librosa.load("/data2/tzc/LA/ASVspoof2019_LA_dev/flac/{}.flac".format(info[1][:-4]),
                               sr=16000)
        wav = process_aug(wav,sr,params)
        wav = torch.Tensor(np.expand_dims(wav, axis=0))
    #     # name = info[1].split('.')[0]
        name = info[1]
        lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
        wav_lfcc = lfcc(wav)
        wav_lfcc = torch.transpose(wav_lfcc, 2, 1).squeeze(0)
        # print(wav_lfcc.shape)
        # print(type(wav_lfcc))
        with open('/data2/tzc/ocspoof/pre/lfcc/dev_aug/' + name + "LFCC" + '.pkl', 'wb') as feature_handle:
            # print('/data2/tzc/ocspoof/pre/lfcc/DF/'+name + "LFCC" + '.pkl')
            pickle.dump(wav_lfcc, feature_handle, protocol=pickle.HIGHEST_PROTOCOL)
        if i % 1000 == 0:
            print('have successfully done {} numbers'.format(i))



import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset_two import ASVspoof2019Two,ASVspoof2021Two,ASVspoofDFTwo,IWDTwo
from rawnet2.model3d import Raw3D
from tqdm import tqdm
import eval_metrics as em
import numpy as np


def test_model(feat_model_path, raw_model_path,loss_model_path, part, add_loss, device):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    dir_yaml = os.path.splitext('./rawnet2/model_config_RawNet2')[0] + '.yaml'
    print(dir_yaml)

    lfcc_model = torch.load('/data2/tzc/ocspoof/train_fca/contact/train_aug2/anti-spoofing_lfcc_model.pt', map_location="cuda")

    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    raw_model = Raw3D(parser1['model'],device)
    raw_model.load_state_dict(torch.load('/data2/tzc/ocspoof/raw/3/contact/train_aug1/anti-spoofing_lfcc_model.pt'))

    # model = torch.load(feat_model_path, map_location="cuda")
    # print(model)

    lfcc_model = lfcc_model.to(device)
    raw_model = raw_model.to(device)

    test_set = ASVspoof2021Two(path_to_features='/data2/tzc/ocspoof/pre/lfcc/eval2021/', feature='LFCC', feat_len=750,
                 path_to_database="/data2/tzc/LA2021eval/flac/",pad_chop=True, padding='repeat')
    # test_set = ASVspoofDFTwo()
    # test_set = IWDTwo()
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    lfcc_model.eval()
    raw_model.eval()

    # model.eval()

    # with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
    with open(os.path.join('/data2/tzc/ocspoof/test_two/raw3/fle_2021LAquan/','checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (lfcc,waveform, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            waveform = waveform.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = lfcc_model(lfcc)
            raw_outputs = raw_model(waveform)
            lfcc_score = F.softmax(lfcc_outputs)[:, 0]
            raw_score = F.softmax(raw_outputs)[:, 0]
            score = torch.add(lfcc_score, raw_score)
            score = torch.div(score, 2)

            # outs = model(lfcc,waveform)
            # score = F.softmax(outs,dim=1)[:,0]


            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

                #DF,IWD
                # cm_score_file.write(
                #     '%s %s %s %s\n' % (audio_fn[j], tags[j],
                #                           "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                #                           score[j].item()))


    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join('/data2/tzc/ocspoof/test_two/raw3/fle_2021/','checkpoint_cm_score.txt'),
    #                                         "/data2/tzc/")
    # return eer_cm, min_tDCF

def test(model_dir, add_loss, device):
    # lfcc_model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    # raw_model_path = os.path.join(model_dir, "anti-spoofing_raw_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    lfcc_model_path = os.path.join('/data2/tzc/ocspoof/train_cfp/contact/so/', "anti-spoofing_lfcc_model.pt")
    raw_model_path = os.path.join('/data2/tzc/ocspoof/raw/fca_4/', "anti-spoofing_lfcc_model.pt")
    test_model(lfcc_model_path, raw_model_path,loss_model_path, "eval", add_loss, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="/data2/tzc/ocspoof/train_two/cfp_raw2/")
    parser.add_argument('-l', '--loss', type=str, default="softmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.loss, args.device)
    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # print(eer_cm_lst)
    # print(min_tDCF_lst)

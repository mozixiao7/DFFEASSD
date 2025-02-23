import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_two import ASVspoof2021Two
from rawnet2.model3d import Raw3D
from tqdm import tqdm
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf

def test_model(device):
    dir_yaml = os.path.splitext('./rawnet2/model_config_RawNet2')[0] + '.yaml'
    print(dir_yaml)

    lfcc_model = torch.load('/data2/tzc/ocspoof/train_fca/contact/train_aug2/anti-spoofing_lfcc_model.pt', map_location="cuda")

    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    raw_model = Raw3D(parser1['model'],device)
    raw_model.load_state_dict(torch.load('/data2/tzc/ocspoof/raw/3/contact/train_aug1/anti-spoofing_lfcc_model.pt'))

    lfcc_model = lfcc_model.to(device)
    raw_model = raw_model.to(device)

    test_set = ASVspoof2021Two(path_to_features='/data2/tzc/ocspoof/pre/lfcc/eval2021/', feature='LFCC', feat_len=750,
                 path_to_database="/data2/tzc/LA2021eval/flac/",pad_chop=True, padding='repeat')

    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    lfcc_model.eval()
    raw_model.eval()

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

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))



    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join('/data2/tzc/ocspoof/test_two/raw3/fle_2021/','checkpoint_cm_score.txt'),
                                            "/data2/tzc/")
    return eer_cm, min_tDCF

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    test_model(args.device)

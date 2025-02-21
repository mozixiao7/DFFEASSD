DFFEASSD: Dual-feature Speech Spoofing Detection Based on Feature Enhanced Attention

## Requirements
python==3.6

pytorch==1.1.0

## Data Preparation
The LFCC features are extracted with the MATLAB implementation provided by the ASVspoof 2019 organizers. Please first run the `process_LA_data.m` with MATLAB, and then run `python3 reload_data.py` with python.
Make sure you change the directory path to the path on your machine.
## Run the training code
For individual lfcc branch, during training, it is necessary to modify the code in the `train_fca.py`. Before training, you need to set some parameter paths: `--path_to_features`;`--path_to_protocol`;`--out_fold`;`--num_epochs`;`--gpu`

For individual raw branch `train_raw.py`, the training is the same as lfcc.

For dual-feature model, you can run the `train_two.py`. Before training, you need to set some parameter paths: `--path_to_features`;`--path_to_protocol`;`--out_fold`;`--num_epochs`;`--gpu`. Please note that if you cannot achieve the experimental results by directly training the entire network, you can try training individual branches separately and then fine-tuning the entire network.
## Run the test code with trained model
You can run the `test_two.py` to test model.


[//]: # (## Citation)

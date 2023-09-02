The codes are based on [PSI-Trajectory-Prediction](https://github.com/PSI-Intention2022/PSI-Trajectory-Prediction) and [SGNet](https://github.com/ChuhuaW/SGNet.pytorch).


## 1. GETTING STARTED
### 1.0. Clone the repository
```buildoutcfg
git clone https://github.com/KaAI-KMU/KaAI-PSI-Trajectory-Prediction.git
```
### 1.1. Install dependencies
Create conda environment.
```buildoutcfg
conda create -n {env_name} python=3.8
conda activate {env_name}
```
Install pytorch. Please refer to [pytorch](https://pytorch.org/get-started/locally/) for the details of installation.
```buildoutcfg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Install other dependencies.
```buildoutcfg
cd KaAI-PSI-Trajectory-Prediction
pip install -r requirements.txt
```
### 1.2. Download data
Please refer to [PSI dataset](https://github.com/PSI-Intention2022/PSI-Dataset) for the details of PSI dataset and data structure.


You can download the pre-processed and structured data(including center optical flow) from [here](https://drive.google.com/file/d/1-htPL4PB6MXztpO4kLcZXI2wiLU1N6GO/view?usp=drive_link) and extract it into KaAI-PSI-Trajectory-Prediction/psi_dataset.


The optical flows are generated by using [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) model which is pretrained with KITTI.
And we use the center of the bounding box as the pedestrian's position.


You can download the data from [here](https://drive.google.com/file/d/1rT6fjBrY0k6iXKwV1UBCNd9u0w_aAw86/view?usp=drive_link) and extract it into KaAI-PSI-Trajectory-Prediction/psi_dataset.


The data structure should be as follows:
```buildoutcfg
- KaAI-PSI-Trajectory
    - psi_dataset
        - PSI2.0_TrainVal
            - annotations
            - splits
        - center_of
            - video_0001
            - video_0002
            - ...
```
For reducing the time for loading the data, you can download the preprocessed data from [here](https://drive.google.com/file/d/10yKDUYWkCIzy8QKwZwePsVGbjVaWKtgt/view?usp=drive_link) and extract it into KaAI-PSI-Trajectory-Prediction/database. If you want to process the data by yourself, just skip this step. 


However, you should download [traj_database_test.pkl](https://drive.google.com/file/d/1fbjr6sXOy1opcOoY_c3lod0WrznWKrRG/view?usp=drive_link) because it won't be generated by running main.py.


The data structure should be as follows:
```buildoutcfg
- KaAI-PSI-Trajectory
    - database
        - traj_database_train.pkl
        - traj_database_val.pkl
        - traj_database_trainval.pkl
        - traj_database_test.pkl
```
### 1.3. Download pre-trained models
We used the pre-trained model of SGNet which is trained with **JAAD dataset**.


You can download the pre-trained models from [here](https://drive.google.com/file/d/1ubY39x34jYQtW9BIprPEgenb4ktn7juP/view?usp=drive_link) and extract it into root directory(KaAI-PSI-Trajectory).

## 2. TRAINING
### 2.1. Train the model
```buildoutcfg
python main.py --config_file configs/psi2.0/sgnet_cvae_flow.yaml --train --pretrained_path SGNet_pretrained_with_JAAD.pth --extra_tag {extra_tag} --epochs 1
```
The model converges after 1 epoch if you use the pre-trained model.


The results will be saved in KaAI-PSI-Trajectory-Prediction/ckpts/ped_traj/PSI2.0/SGNetCVAETrajBbox/{extra_tag}.


If you want to train the model with full training set(validation set is used as training set), please use --trainval option instead of --train option. (**We used full training set for training the submitted model.**)
```buildoutcfg
python main.py --config_file configs/psi2.0/sgnet_cvae_flow.yaml --train --pretrained_path SGNet_pretrained_with_JAAD.pth --extra_tag {extra_tag} --epochs 1
```
## 3. EVALUATION
### 3.1. Evaluate the model with validation set
As default, the model will be evaluated when the training is finished.


However, if you want to evaluate the model with the specific checkpoint with validation set, please use --pretrained_path option.
```buildoutcfg
python main.py --config_file configs/psi2.0/sgnet_cvae_flow.yaml --val --pretrained_path {pretrained_path} --extra_tag {extra_tag}
```
The results will be saved in KaAI-PSI-Trajectory-Prediction/ckpts/ped_traj/PSI2.0/SGNetCVAETrajBbox/{extra_tag}.
### 3.2. Test the model
If you want to test the model with test set, please use --test option.
```buildoutcfg
python main.py --config_file configs/psi2.0/sgnet_cvae_flow.yaml --test --pretrained_path {pretrained_path} --extra_tag {extra_tag}
```
The results will be saved in KaAI-PSI-Trajectory-Prediction/ckpts/ped_traj/PSI2.0/SGNetCVAETrajBbox/{extra_tag}.

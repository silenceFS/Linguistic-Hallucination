# Textual Hallucination for Partially Relevant Video Retrieval

source code


* 

## Environments 
* **python 3.8**
* **pytorch 1.9.0**
* **torchvision 0.10.0**
* **tensorboard 2.6.0**
* **tqdm 4.62.0**
* **easydict 1.9**
* **h5py 2.10.0**
* **cuda 11.1**
* **transformers** 

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.
```
conda create --name THPRVR python=3.8
conda activate THPRVR
cd THPRVR
pip install -r requirements.txt
```

## Experiment

### Required Data

Run the following script to download the video feature and text feature of the Charades-STA dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). 

```
wget http://8.210.46.84:8787/prvr/data/charades.tar
tar -xvf charades.tar
```

Run the following script to download the video feature and text feature of the TVR dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4).

```
# download the data of TVR
wget http://8.210.46.84:8787/prvr/data/tvr.tar
tar -xvf tvr.tar
```


Run the following script to download the video feature and text feature of the Activitynet dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). 

```
wget http://8.210.46.84:8787/prvr/data/activitynet.tar
tar -xvf activitynet.tar
```

### Training
Run the following script to train this network on TVR. Pay attention that you add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
```
source setup.sh
conda activate THPRVR
device_ids=0
exp_id=runs_0
ssn=3
consistency_weight=0.2
hallucination_weight=0.01
root_path=Your root path
support_ckpt_filepath=your support ckpt filepath
sh do_tvr.sh $device_ids $exp_id $ssn $consistency_weight $hallucination_weight $root_path $support_ckpt_filepath
```

`$device_ids` is the index of the GPU that is used for training.

`$exp_id` is the name of the experiment and also the name of floder where the model and results are saved in.

`ssn` is the number of context information

`consistency_weight` and `hallucination_weight` are used to balance consistency loss and hallucination loss

To train another two datasets, just run `do_activitynet.sh` and `do_charades.sh` in the same way as `do_tvr.sh`  

If you want to train in  **Curriculum Learning** method, you should modify this shell and set `--curriculum` in the end of the command

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=tvr
FEATURE=i3d_resnet
ROOTPATH=your root path
MODELDIR=the name of folder where the model is saved

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

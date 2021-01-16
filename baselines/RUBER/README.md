# RUBER

This page contains the code to train and evaluate the RUBER and RUBER-large models on dailydialog++. This code was built by referring [this repository](https://github.com/gmftbyGMFTBY/RUBER-and-Bert-RUBER)

## Installation 

This code requires Python 3.6 and the following requirements. 

```
nltk==3.5
numpy==1.19.1
prettytable==0.7.2
scikit-learn==0.23.2
scipy==1.5.2
sklearn==0.0
torch==1.1.0
tqdm==4.48.2
```
Install the above dependencies using:
 ```
 pip install -r requirements.txt 
 ```
 
 ## Preprocessing 
 
 Download the vocabulary and word embedding files from [here](https://drive.google.com/file/d/1KY_ZEtPYWPr8TfLPQrjG3EGfkGlvKsS2/view?usp=sharing) and place them in ```./data```
 
 Then use the following command to preprocess data with random negatives: 
 
 ```
 python utils.py --train-data ../../dataset/train.json 
                 --dev-data ../../dataset/dev.json \
                 --test-data ../../dataset/test.json \
                 --mode random
 ```
 Change ```mode``` to ```adversarial``` or ```both``` to preprocess data with adversarial negatives or both adversarial and random negatives. 

## Pre-trained Models

Download the pretrained checkpoints from below:


| Model       | Architecture                                                | No. of Parameters | Dataset                      | Download |
|-------------|-------------------------------------------------------------|-------------------|------------------------------|----------|
| RUBER       | 1-layer Bi-directional GRUs <br /> with hidden size of 1024 | 34M               | Reddit <br /> (20M contexts) | [here](https://drive.google.com/file/d/1fymMcrhr0K7y5xCJOa5p6dgG0F7temcE/view?usp=sharing)     |
| RUBER-Large | 2-layer Bi-directional GRUs <br /> with hidden size of 2048 | 236M              | Reddit <br /> (20M contexts) | [here](https://drive.google.com/file/d/1iCr9oIPqGn_Iaxz0ws5ttwi6ge8wpUrX/view?usp=sharing)     |


## Training & Evaluation

To finetune and evaluate RUBER on dailydailog++ random negatives, use the following command (provide appropriate paths to the ```EXPDIR``` and ```CKPT``` variables): 

```
EXPDIR='./experiments/random'
CKPT='./pretrained_checkpoints/RUBER_reddit_pretrained.pt'
python train_unreference.py --hidden-size 1024 --num-layers 1 \
              --exp-dir ${EXPDIR} --lr 1e-5 \
              --weight-decay 1e-6 --seed 100 --mode random \
              --batch-size 256 --save-steps 3000 \
              --init-checkpoint ${CKPT} 
```

To finetune on ```RUBER-Large```, change ```hidden-size``` to ```2048```, ```num-layers``` to ```2```, and change the ```CKPT``` variable to the appropriate path. 

To finetune and evaluate on adversarial or random+adversarial negative data, change the ```mode``` to ```adversarial``` or ```both``` respectively.

Add the ```--test-only``` flag to only perform evaluation (no training). 

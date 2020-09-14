# BERT+DNN

This page contains the code to train and evaluate the BERT+DNN model on dailydialog++. 


## Installation 

This code requires Python 3.6 and the following requirements. 

```
numpy==1.19.1
prettytable==0.7.2
scikit-learn==0.23.2
scipy==1.5.2
sklearn==0.0
torch==1.1.0
tqdm==4.48.2
transformers==3.1.0
```
Install the above dependencies using:
 ```
 pip install -r requirements.txt 
 ```
 
 ## Preprocessing 
  
 Run the following command to store the BERT embeddings for the data with random negatives: 
 
 ```
 python utils.py --train-data ../../dataset/train.json \
                 --dev-data ../../dataset/dev.json \
                 --test-data ../../dataset/test.json \
                 --mode random --batch-size 128
 ```
 Change ```mode``` to ```adversarial``` or ```both``` to preprocess data with adversarial negatives or both adversarial and random negatives. 
 

## Training & Evaluation

To train and evaluate a DNN on top of the BERT embeddings, use the following command: 

```
EXPDIR='./experiments/random'
python train_unreference.py --exp-dir ${EXPDIR} --lr 1e-5 \
                            --weight-decay 1e-4 --seed 100 \
                            --mode random --batch-size 512 \
                            --init-checkpoint ${CKPT} 
```

To train and evaluate on adversarial or random+adversarial negative data, change the ```mode``` to ```adversarial``` or ```both``` respectively.

Add the ```--test-only``` flag to only perform evaluation (no training). 

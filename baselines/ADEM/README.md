# ADEM

This page contains the code to train and evaluate the ADEM model on dailydialog++. This code is based on the [official implementation](https://github.com/mike-n-7/ADEM) of ADEM in Theano released by the authors. 

## Installation 

The code required Python 2.7 and the following requirements. 

```
Lasagne @ https://github.com/Lasagne/Lasagne/archive/master.zip
numpy==1.16.6
scikit-learn==0.20.4
scipy==1.2.3
six==1.15.0
sklearn==0.0
Theano==1.0.4
tqdm==4.48.2
pygpu
```

```pygpu``` is unfortunately not available in pip, install it using ```conda``` . The other dependencies can be installed using ```pip```:
 ```
 conda install pygpu
 pip install -r requirements.txt 
 ```
 
## Training & Evaluation

Download the weights for the pretrained VHRED model released by the authors from [here](https://drive.google.com/file/d/0B-nb1w_dNuMLY0Fad3N1YU9ZOU0/view?usp=sharing), and place all the files in the ```./vhred``` folder. 

The following command can be used to train and evaluate the ADEM model on the dailydialog++ dataset:

```
THEANO_FLAGS='device=${device},floatX=float32' python train.py --train-data ${train_data}\
    --test-data ${test_data} --dev-data ${dev_data}\
    --mode ${mode} --expt-dir ${expdir} --vhrd-data ${expdir}
```
where ```device``` can be either ```cpu```  or ```cuda``` (for GPU). ```train_data```, ```test_data```, and ```dev_data``` are paths to the respective dailydialog++ dataset splits. ```mode``` can be either ```random``` or ```adversarial``` indicating the type of negative response to use for training and evaluation. ```expdir``` refers to path to the experiment directory.  

For example, the following command can be used to train and evaluate on random negatives on a GPU

```
THEANO_FLAGS='device=cuda, floatX=float32' python train.py --train-data ../../dataset/train.json\
    --test-data ../../dataset/test.json --dev-data ../../dataset/dev.json\
    --mode random --expt-dir experiments/random/ --vhrd-data experiments/random/ 
```

To only perform evaluation (from a trained model), add the ```--test-only``` flag. 

 

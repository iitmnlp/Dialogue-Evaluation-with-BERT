# Evaluation using DEB

The script for running evaluation is provided in ```run_deb.sh``` The input arguments and the environment can be appropriately changed.
Since we were running on TPUs on the Google Cloud Platform (GCP), the dataset is uploaded into a storage bucket on GCP as a tfrecord.

The tfrecords corresponding to the test set used in the paper are available in the ```test_tfrecords``` folder. The train tfrecords (only if required) are available [here](https://drive.google.com/drive/folders/1cxsCjGtOJUVDEOkwHR3DCXrbRE0lVRZJ). These tfrecords can be uploaded to the GCP storage bucket and the paths need to be specified appropriately.

Also the model checkpoints can be added to the appropriate locations from [here](https://drive.google.com/drive/folders/1N-_oFl26eGQM413zSQZ36ZFLTXzcpXqj). This folder contains the checkpoints for both the models: DEB-trained-on-random-negatives and DEB-trained-on-both-random-and-advevrsarial-negatives.

## For a new dataset
To create a json from the tfrecord file, use ```run_create_tfrecord_data_from_json.sh```.

## Pretraining on Reddit

To perform Reddit pretraining, we have followed [the code repo by Henderson et. al](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit) for getting reddit dataset from the years 2005-2019.


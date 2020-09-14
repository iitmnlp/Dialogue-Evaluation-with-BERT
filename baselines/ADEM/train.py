'''
This file is the training pipeline for the ADEM model.
'''
import preprocess
import argparse
from experiments import configurations
import os
import cPickle
from models import ADEM
import sys

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--train-data", type=str, help="Path to the train file", default=None)
	parser.add_argument("--test-data", type=str, help="Path to the test file", default=None)
	parser.add_argument("--dev-data", type=str, help="Path to the dev file", default=None)
	parser.add_argument("--mode", type=str, help="Type of negative response to use", choices=['random', 'adversarial'])
	parser.add_argument("--expt-dir", type=str, help="expt directory (must be specified)", default='./experiments/default')
	parser.add_argument("--test-only", action='store_true', help='whether to only run test or both train and test')
	parser.add_argument("--vhrd-data", type=str, help="path to the saved vhrd embeddings", default='./experiments/default')
	parser.add_argument("--load-checkpoint", type=str, help="Path to the checkpoint file to load", default=None)

	args = parser.parse_args()
	return args

def create_experiment(config):
	if not os.path.exists(config['exp_folder']):
		os.makedirs(config['exp_folder'])

if __name__ == "__main__":
	args = parse_args()
	config = configurations(args)

	print 'Beginning...'
	sys.stdout.flush()
	create_experiment(config)

	adem = ADEM(config, load_from=args.load_checkpoint)
	sys.stdout.flush()
	if args.test_only:
		adem.test(use_saved_embeddings=True)
	else:
		adem.train_eval(use_saved_embeddings=True)
	
	print 'Finished!'
	sys.stdout.flush()

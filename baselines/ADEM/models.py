from pretrain import *
import os
import theano
import theano.tensor as T
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import lasagne
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from preprocess import DataLoader, Preprocessor
from scipy.stats import pointbiserialr
from sklearn.metrics import accuracy_score

class ADEM(object):
	
	def __init__(self, config, load_from=None):

		if load_from is not None:
			print ('loading model from ',load_from)
			self.load(load_from)
			self.config = config
		else:
			self.config = config

		self.pretrainer = None
		if self.config['pretraining'].lower() == 'vhred':
			self.pretrainer = VHRED(self.config)

	def _compute_pca(self, train_x):
		# Reduce the input vectors to a lower dimensional space.
		self.pca = PCA(n_components = self.config['pca_components'])

		# Count the number of examples in each set.
		n_train = train_x.shape[0]

		# Flatten the first two dimensions. The first dimension now includes all the contexts, then responses.
		x_flat = np.zeros((n_train * 3, train_x.shape[2]), dtype='float32')
		for i in range(3):
			x_flat[n_train*i: n_train*(i+1),:] = train_x[:,i,:]
		pca_train = self.pca.fit_transform(x_flat)

		print 'PCA Variance'
		print self.pca.explained_variance_ratio_
		print np.sum(self.pca.explained_variance_ratio_)

		# Expand the result back to three dimensions.
		train_pca_x = np.zeros((n_train, 3, self.config['pca_components']), dtype='float32')
		for i in range(3):
			train_pca_x[:,i,:] = pca_train[n_train*i: n_train*(i+1),:]
		
		return train_pca_x

	def _apply_pca(self, x):
		pca_x = np.zeros((x.shape[0], 3, self.config['pca_components']), dtype='float32')
		# Perform PCA, only fitting on the training set.
		pca_x[:,0,:] = self.pca.transform(x[:,0,:])
		pca_x[:,1,:] = self.pca.transform(x[:,1,:])
		pca_x[:,2,:] = self.pca.transform(x[:,2,:])
		return pca_x

	def _build_data(self, data):
		
		n_models = len(data[0]['r_models'])
		n = len(data)*(n_models)		
		emb_dim = len(data[0]['c_emb'])

		# Create arrays to store the data. The middle dimension represents:
		# 0: context, 1: gt_response, 2: model_response
		x = np.zeros((n, 3, emb_dim), dtype=theano.config.floatX)
		y = np.zeros((n,), dtype=theano.config.floatX)
		
		# Load in the embeddings from the dataset.
		for ix, entry in enumerate(data):
			for jx, m_name in enumerate(data[ix]['r_models'].keys()):
				kx = ix*n_models + jx
				x[kx, 0, :] = data[ix]['c_emb']
				x[kx, 1, :] = data[ix]['r_gt_emb']
				x[kx, 2, :] = data[ix]['r_model_embs'][m_name]
				y[kx] = data[ix]['r_models'][m_name][1]
				
		return x, y

	def _build_model(self, emb_dim, init_mean, init_range, training_mode = False):
		index = T.lscalar()
		# Theano variables for computation graph.
		x = T.tensor3('x')
		y = T.ivector('y')

		# Matrices for predicting score
		self.M = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
		self.N = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
		
		# Set embeddings by slicing tensor
		self.emb_context = x[:,0,:]
		self.emb_true_response = x[:,1,:]
		self.emb_response = x[:,2,:]

		# Compute score predictions
		self.pred1 = T.sum(self.emb_context * T.dot(self.emb_response, self.M), axis=1)
		self.pred2 = T.sum(self.emb_true_response * T.dot(self.emb_response, self.N), axis=1)

		self.pred = 0
		if self.config['use_c']: self.pred += self.pred1
		if self.config['use_r']: self.pred += self.pred2

		# To re-scale dot product values to [1,5] range.
		output = 3 + 4 * (self.pred - init_mean) / init_range  

		loss = T.mean((output - y)**2)
		l2_reg = self.M.norm(2) + self.N.norm(2)
		l1_reg = self.M.norm(1) + self.N.norm(1) 

		score_cost = loss + self.config['l2_reg'] * l2_reg + self.config['l1_reg'] * l1_reg

		# Get the test predictions.
		self._get_outputs = theano.function(
			inputs=[x,],
			outputs=output,
			on_unused_input='warn'
		)

		params = []
		if self.config['use_c']: params.append(self.M)
		if self.config['use_r']: params.append(self.N)
		updates = lasagne.updates.adam(score_cost, params)

		if training_mode == True:
			bs = self.config['bs']
			self._train_model = theano.function(
				inputs=[index],
				outputs=score_cost,
				updates=updates,
				givens={
					x: self.train_x[index * bs: (index + 1) * bs],
					y: self.train_y[index * bs: (index + 1) * bs],
				},
				on_unused_input = 'warn'
			)

	def _compute_init_values(self, emb):
		prod_list = []
		for i in xrange(len(emb[0][0])):
			term = 0
			if self.config['use_c']: term += np.dot(emb[i, 0], emb[i, 2])
			if self.config['use_r']: term += np.dot(emb[i, 1], emb[i, 2])
			prod_list.append(term)
		alpha = np.mean(prod_list)
		beta = max(prod_list) - min(prod_list)
		return alpha, beta

	def _correlation(self, output, score):
		return  [spearmanr(output, score), pearsonr(output, score)]
	
	def _set_shared_variable(self, x):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)	


	def get_vhrd_embeddings(self, data_fname, mode, save_name, use_saved_embeddings):

		if (use_saved_embeddings) and (os.path.exists(save_name)):
			with open(save_name, 'rb') as handle:
				data = cPickle.load(handle)
		else:
			data_loader = DataLoader(data_fname, mode)
			data = data_loader.load_data()
			assert not self.pretrainer is None
			data = self.pretrainer.get_embeddings(data)

			with open(save_name, 'wb') as handle:
				cPickle.dump(data, handle)

		x,y = self._build_data(data)
		return x, y

	def test(self, use_saved_embeddings=True):

		self.load('%s/adem_model.pkl' % self.config['exp_folder'])
		test_fname_embeddings = '%s/test_%s_embeddings.pkl' % (self.config['vhrd_data'], self.config['mode'])
		test_x, test_y = self.get_vhrd_embeddings(self.config['test_data'], self.config['mode'], test_fname_embeddings, use_saved_embeddings)
		if self.config['use_pca']:
			test_x = self._apply_pca(test_x)
		
		predictions = (np.array(self._get_outputs(test_x))-1)/4.0
		test_y = (np.array(test_y)-1)/4

		predictions_positive = predictions[test_y==1]
		predictions_negative = predictions[test_y==0]
		
		np.savetxt(os.path.join(self.config['exp_folder'], 'positive_probs.npy'), predictions_positive)
		np.savetxt(os.path.join(self.config['exp_folder'], '{}_negative_probs.npy'.format(self.config['mode'])), predictions_negative)

		acc = accuracy_score(test_y, predictions>0.5)
		print ('Accuracy: ',acc)
		matrix = confusion_matrix(test_y, predictions>0.5)
		print ('confusion_matrix: ', matrix)
		pbc, pval = pointbiserialr(test_y, predictions)
		print ('PBC: ', pbc, 'p-value: ', pval)
		sys.stdout.flush()

		return

	def train_eval(self, use_saved_embeddings=True):

		# Each dictionary looks like { 'c': context, 'r_gt': true response, 'r_models': {'hred': (model_response, score), ... }}
		train_fname_embeddings = '%s/train_%s_embeddings.pkl' % (self.config['vhrd_data'], self.config['mode'])
		dev_fname_embeddings = '%s/dev_%s_embeddings.pkl' % (self.config['vhrd_data'], self.config['mode'])

		train_x, train_y = self.get_vhrd_embeddings(self.config['train_data'], self.config['mode'], train_fname_embeddings, use_saved_embeddings)
		val_x, val_y = self.get_vhrd_embeddings(self.config['dev_data'], self.config['mode'], dev_fname_embeddings, use_saved_embeddings)

		if self.config['use_pca']:
			print ('Computing PCA ...')
			train_x = self._compute_pca(train_x)
			val_x = self._apply_pca(val_x)
		
		init_mean, init_range = self._compute_init_values(train_x)
		self.init_mean, self.init_range = init_mean, init_range

		self.train_x = self._set_shared_variable(train_x)
		self.val_x = self._set_shared_variable(val_x)

		self.train_y = theano.shared(np.asarray(train_y, dtype='int32'), borrow=True)

		n_train_batches = train_x.shape[0] / self.config['bs']

		# Build the Theano model.
		self._build_model(train_x.shape[2], init_mean, init_range, training_mode=True)

		# Train the model.
		print ('Starting training...')
		epoch = 0
		# Vairables to keep track of the best achieved so far.
		best_output_val = np.zeros((50,)) 
		best_val_cor, best_test_cor = [0,0], [0,0]
		# Keep track of loss/epoch.
		loss_list = []
		# Keep track of best parameters so far.
		best_val_loss, best_epoch = np.inf, -1

		indices = range(n_train_batches)


		for epoch in (range(self.config['max_epochs'])):
			np.random.shuffle(indices)

			# Train for an epoch.
			cost_list = []
			for minibatch_index in indices:
				minibatch_cost = self._train_model(minibatch_index)
				cost_list.append(minibatch_cost)
			loss_list.append(np.mean(cost_list))

			# Get the predictions for each dataset.
			model_train_out = self._get_outputs(train_x)
			model_val_out = self._get_outputs(val_x)
			# Get the training and validation MSE.
			train_loss = np.sqrt(np.mean(np.square(model_train_out - train_y)))
			val_loss = np.sqrt(np.mean(np.square(model_val_out - val_y)))
			# Keep track of the correlations.
			train_correlation = self._correlation(model_train_out, train_y)
			val_correlation = self._correlation(model_val_out, val_y)

			print ('Epoch : {} Train Loss: {} Train Correlation: {}'.format(epoch, train_loss, train_correlation))
			print ('Epoch : {} Val Loss: {} Val Correlation: {}'.format(epoch, val_loss, val_correlation))
			sys.stdout.flush()

			# Only save the model when we best the best MSE on the validation set.
			if val_loss < best_val_loss:
				best_val_cor = val_correlation
				best_val_loss = val_loss
				print ('Saving at epoch', epoch)
				best_epoch = epoch
				self.best_params = [self.M.get_value(), self.N.get_value()]
				self.save()

		print ('Done training!')
		print ('Last updated on epoch %d' % best_epoch)

		self.test(use_saved_embeddings)		


	def load(self, f_name):
		with open(f_name, 'rb') as handle:
			saved_model = cPickle.load(handle)
		config = saved_model['config']
		init_mean, init_range = saved_model['init_mean'], saved_model['init_range']
		self._build_model(config['pca_components'], init_mean, init_range)
		self.pca = saved_model['pca']
		self.M.set_value(saved_model['params'][0])
		self.N.set_value(saved_model['params'][1])	

	def save(self):
		# Save the PCA model.
		saved_model = {'pca': self.pca, 
					'params': self.best_params,
					'config': self.config,
					'init_mean': self.init_mean,
					'init_range': self.init_range }
		with open('%s/adem_model.pkl' % self.config['exp_folder'], 'wb') as handle:
			cPickle.dump(saved_model, handle)
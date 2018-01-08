'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import json
import ipdb
import numpy as np
import copy

import os
import warnings
import sys
import time

from subprocess import Popen

from collections import OrderedDict

profile = False
#theano.config.compute_test_value = 'warn'

from data_iterator import TextIterator
from util import *
from theano_util import *
from alignment_util import *

from layers import *
from initializers import *
from optimizers import *

from domain_interpolation_data_iterator import DomainInterpolatorTextIterator


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
	# x: a list of sentences
	lengths_x = [len(s) for s in seqs_x]
	lengths_y = [len(s) for s in seqs_y]

	if maxlen is not None:
		new_seqs_x = []
		new_seqs_y = []
		new_lengths_x = []
		new_lengths_y = []
		for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
			"""
				discard the setence pair, as long as 'length'(src or tgt) > 'maxlen'
		   """
			if l_x < maxlen and l_y < maxlen:
				new_seqs_x.append(s_x)
				new_lengths_x.append(l_x)
				new_seqs_y.append(s_y)
				new_lengths_y.append(l_y)
		lengths_x = new_lengths_x
		seqs_x = new_seqs_x
		lengths_y = new_lengths_y
		seqs_y = new_seqs_y
		'''
				lengths_x  format: [23, 23, 30, 28, 37, 32, 39, 19, 34, 26, 31, 14, 21, 19]  items in list means source sentence length in this batch
				lengths_y  format: is the same as lengths_y, target sentence length
				seqs_x     format: [ [74], [11], [16], [92], [401], [521], [56], [27], [5], [521], [7], [144], [5060], [521], [35], [149], [14], [62] ]
					 a  two dimensional array   item means the source word index which is in source vocabulay
				seqs_y       format: is the same as seqs_x, target word index
		'''
		if len(lengths_x) < 1 or len(lengths_y) < 1:
			return None, None, None, None

	n_samples = len(seqs_x)
	n_factors = len(seqs_x[0][0])
	maxlen_x = numpy.max(lengths_x) + 1
	maxlen_y = numpy.max(lengths_y) + 1

	x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
	y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
	x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
	y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
	'''
		x is a three dimensional matrix
		y is a two dimmensional matrix  which maxlen_y rows  *  n_samples columns
	'''
	for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
		x[:, :lengths_x[idx], idx] = zip(*s_x)
		x_mask[:lengths_x[idx] + 1, idx] = 1.
		y[:lengths_y[idx], idx] = s_y
		y_mask[:lengths_y[idx] + 1, idx] = 1.
	return x, x_mask, y, y_mask


# initialize all parameters
def init_params(options):
	params = OrderedDict()

	# embedding
	"""
		comment: source embeding, different factor has different size of embeding
	"""
	for factor in range(options['factors']):
		params[embedding_name(factor)] = norm_weight(options['n_words_src'], options['dim_per_factor'][factor])

	"""
		commnet: target embedining, fixed
	"""
	params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

	# encoder: bidirectional RNN
	params = get_layer_param(options['encoder'])(options, params,
	                                             prefix='encoder',
	                                             nin=options['dim_word'],
	                                             dim=options['dim'])
	params = get_layer_param(options['encoder'])(options, params,
	                                             prefix='encoder_r',
	                                             nin=options['dim_word'],
	                                             dim=options['dim'])
	ctxdim = 2 * options['dim']

	# init_state, init_cell
	"""
		comment: this ffnn is used to learn to transfer 'encoder average hidden state' to 'decoder initial hidden state'
	"""
	params = get_layer_param('ff')(options, params, prefix='ff_state',
	                               nin=ctxdim, nout=options['dim'])
	# decoder
	params = get_layer_param(options['decoder'])(options, params,
	                                             prefix='decoder',
	                                             nin=options['dim_word'],
	                                             dim=options['dim'],
	                                             dimctx=ctxdim)
	# readout
	params = get_layer_param('ff')(options, params, prefix='ff_logit_lstm',
	                               nin=options['dim'], nout=options['dim_word'],
	                               ortho=False)
	params = get_layer_param('ff')(options, params, prefix='ff_logit_prev',
	                               nin=options['dim_word'],
	                               nout=options['dim_word'], ortho=False)
	params = get_layer_param('ff')(options, params, prefix='ff_logit_ctx',
	                               nin=ctxdim, nout=options['dim_word'],
	                               ortho=False)
	params = get_layer_param('ff')(options, params, prefix='ff_logit',
	                               nin=options['dim_word'],
	                               nout=options['n_words'])

	return params


"""
    return:
    trng: random stream
    use_noise: 0.
    x: encoder input batch, shape is [factor_num, encoder_step_num, sample_num]
    x_mask: source padding mask, shape is [encoder_step_num, sample_num], because every 'factor' has same mask
    y: decoder input batch, shape is [decoder_step_num, sample_num]
    y_mask: target padding mask, shape is same with 'y'
    opt_ret: dict, key=dec_alphas, value is attention weight, shape is [decoder_step_num, sample_num, encoder_step_num]
    cost: accumulate '-log(Pi)' of every valid sample (not include padding, here sample is sentence-level), shape is [sample_num]
"""


# build a training model
def build_model(tparams, options):
	opt_ret = dict()

	trng = RandomStreams(1234)
	use_noise = theano.shared(numpy.float32(0.))

	# description string: #words x #samples
	x = tensor.tensor3('x', dtype='int64')
	x.tag.test_value = (numpy.random.rand(1, 5, 10) * 100).astype('int64')
	x_mask = tensor.matrix('x_mask', dtype='float32')
	x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
	y = tensor.matrix('y', dtype='int64')
	y.tag.test_value = (numpy.random.rand(8, 10) * 100).astype('int64')
	y_mask = tensor.matrix('y_mask', dtype='float32')
	y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')
	#phrase_mask = tensor.matrix('phrase_mask',dtype='float32')

	# for the backward rnn, we just need to invert x and x_mask
	"""
		comment: revert input wid
		x shape is [factor_num, step_num, sample_num]
		x_mask shape is [step_num, sample_num]
		e.g. a = [[1,2,3], [4,5,6], [7,8,9]], a[::-1] = [[7,8,9], [4,5,6], [1,2,3]]
		here raw samples are (1) 1-4-7 (2) 2-5-8 (3) 3-6-9
		changed samples are (1) 7-4-1 (2) 8-5-2 (3) 9-6-3, so this operation inverted 'x'
	"""
	xr = x[:, ::-1]
	xr_mask = x_mask[::-1]

	n_timesteps = x.shape[1]
	n_timesteps_trg = y.shape[0]
	n_samples = x.shape[2]

	if options['use_dropout']:
		retain_probability_emb = 1 - options['dropout_embedding']
		retain_probability_hidden = 1 - options['dropout_hidden']
		retain_probability_source = 1 - options['dropout_source']
		retain_probability_target = 1 - options['dropout_target']
		rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
		rec_dropout_r = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
		rec_dropout_d = shared_dropout_layer((5, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
		emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
		emb_dropout_r = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng,
		                                     retain_probability_emb)
		emb_dropout_d = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng,
		                                     retain_probability_emb)
		ctx_dropout_d = shared_dropout_layer((4, n_samples, 2 * options['dim']), use_noise, trng,
		                                     retain_probability_hidden)
		source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source)
		target_dropout = shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng,
		                                      retain_probability_target)
		source_dropout = tensor.tile(source_dropout, (1, 1, options['dim_word']))
		target_dropout = tensor.tile(target_dropout, (1, 1, options['dim_word']))
	else:
		rec_dropout = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		rec_dropout_r = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		rec_dropout_d = theano.shared(numpy.array([1.] * 5, dtype='float32'))
		emb_dropout = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		emb_dropout_r = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		emb_dropout_d = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		ctx_dropout_d = theano.shared(numpy.array([1.] * 4, dtype='float32'))

	# word embedding for forward rnn (source)
	"""
		comment: each factor has some-dim embedding, concatenate all factor's embedding, then we can get the final embedding
		if 'use_dropout', we can apply here
		emb shape is [step_num, sample_num, embedding_size], here embedding_size = hidden_size
	"""
	emb = []
	for factor in range(options['factors']):
		emb.append(tparams[embedding_name(factor)][x[factor].flatten()])
	emb = concatenate(emb, axis=1)
	emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
	if options['use_dropout']:
		emb *= source_dropout

	"""
		comment: proj shape is [step_num, sample_num, hidden_size]
		regard as +h_1, +h_2, ...,+h_m, '+' denote ordinal
	"""
	proj = get_layer_constr(options['encoder'])(tparams, emb, options,
	                                            prefix='encoder',
	                                            mask=x_mask,
	                                            emb_dropout=emb_dropout,
	                                            rec_dropout=rec_dropout,
	                                            profile=profile)

	# word embedding for backward rnn (source)
	embr = []
	for factor in range(options['factors']):
		embr.append(tparams[embedding_name(factor)][xr[factor].flatten()])
	embr = concatenate(embr, axis=1)
	embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
	if options['use_dropout']:
		embr *= source_dropout[::-1]

	"""
		comment: proj shape is [step_num, sample_num, hidden_size]
		regard as -h_m, -h_m-1, ...,-h_1, '-' denote reversed
	"""
	projr = get_layer_constr(options['encoder'])(tparams, embr, options,
	                                             prefix='encoder_r',
	                                             mask=xr_mask,
	                                             emb_dropout=emb_dropout_r,
	                                             rec_dropout=rec_dropout_r,
	                                             profile=profile)


	# context will be the concatenation of forward and backward rnns
	word_ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

	if options.has_key('use_phrase') and options['use_phrase'] == True:
		"""
			what I do:
			span(i,j) = wi,wi+1,...,wj-1
			R(span(i,j)) = [hj-hi, h'i-h'j]

			we change ctx, x_mask, x_mask_r
		"""

		wq_src_len = x.shape[1]
		wq_batch_size = x.shape[2]
		wq_hidden_size = proj[0].shape[2]
		wq_N = options['ngram']
		wq_zero_state = tensor.zeros((1, wq_batch_size, wq_hidden_size), dtype='float32')

		# add h0 in head of forward-rnn and tail of backward-rnn
		# proj = np.vstack((zero_state, proj))
		wq_proj = proj[0]
		wq_projr = projr[0]
		wq_proj = tensor.concatenate((wq_zero_state, wq_proj), axis=0)
		wq_projr = tensor.concatenate((wq_projr[::-1], wq_zero_state), axis=0)

		# get span index sequence
		wq_idx = tensor.arange((wq_N * wq_src_len))

		def _wq_slice_1_(i, l):
			beg_id = i % l
			n_id = i / l
			end_id = beg_id + n_id + 1
			return (beg_id, end_id)

		(beg, end), _ = theano.scan(
			_wq_slice_1_,
			sequences=[wq_idx],
			non_sequences=[wq_src_len],
			n_steps=wq_N * wq_src_len
		)
		beg_tmp = beg
		end_tmp = end

		wq_indice = (end <= wq_src_len).nonzero()
		beg = beg[wq_indice]
		end = end[wq_indice]

		# phrase-based context
		# phrase_num = (2 * src_len - N + 1) * N / 2
		phrase_num = beg.shape[0]
		ctx = tensor.zeros((phrase_num, wq_batch_size, 2 * wq_hidden_size), dtype='float32')
		ctx = tensor.set_subtensor(ctx[:, :, :wq_hidden_size], wq_proj[end, :, :] - wq_proj[beg, :, :])
		ctx = tensor.set_subtensor(ctx[:, :, wq_hidden_size:], wq_projr[beg, :, :] - wq_projr[end, :, :])

		new_mask = tensor.zeros((phrase_num, wq_batch_size), dtype='float32')
		new_mask = tensor.set_subtensor(new_mask[:, :], x_mask[beg, :] * x_mask[end - 1, :])

	"""
		comment: ctx_mean shape is [sample_num, 2*hidden_size]
	"""
	# mean of the context (across time) will be used to initialize decoder rnn
	#ctx_mean = (ctx * new_mask[:, :, None]).sum(0) / new_mask.sum(0)[:, None]
	ctx_mean = (word_ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

	# or you can use the last state of forward + backward encoder rnns
	# ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

	if options['use_dropout']:
		ctx_mean *= shared_dropout_layer((n_samples, 2 * options['dim']), use_noise, trng, retain_probability_hidden)


	# initial decoder state
	init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
	                                    prefix='ff_state', activ='tanh')

	# word embedding (target), we will shift the target sequence one time step
	# to the right. This is done because of the bi-gram connections in the
	# readout and decoder rnn. The first target will be all zeros and we will
	# not condition on the last output.

	emb = tparams['Wemb_dec'][y.flatten()]
	emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])

	emb_shifted = tensor.zeros_like(emb)
	emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
	emb = emb_shifted

	if options['use_dropout']:
		emb *= target_dropout

	# decoder - pass through the decoder conditional gru with attention
	proj = get_layer_constr(options['decoder'])(tparams, emb, options,
	                                            prefix='decoder',
	                                            mask=y_mask, context=ctx,
	                                            context_mask=new_mask,
	                                            one_step=False,
	                                            init_state=init_state,
	                                            emb_dropout=emb_dropout_d,
	                                            ctx_dropout=ctx_dropout_d,
	                                            rec_dropout=rec_dropout_d,
	                                            profile=profile)

	# hidden states of the decoder gru

	#proj_h shape is [decoder_step_num, sample_num, hidden_size]

	proj_h = proj[0]

	# weighted averages of context, generated by attention module
	ctxs = proj[1]

	if options['use_dropout']:
		proj_h *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
		emb *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
		ctxs *= shared_dropout_layer((n_samples, 2 * options['dim']), use_noise, trng, retain_probability_hidden)

	# weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
	# shape is [decoder_step_num, sample_num, encoder_step_num]
	opt_ret['dec_alphas'] = proj[2]

	# compute word probabilities
	logit_lstm = get_layer_constr('ff')(tparams, proj_h, options,
	                                    prefix='ff_logit_lstm', activ='linear')
	logit_prev = get_layer_constr('ff')(tparams, emb, options,
	                                    prefix='ff_logit_prev', activ='linear')
	logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
	                                   prefix='ff_logit_ctx', activ='linear')
	logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)

	if options['use_dropout']:
		logit *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden)


	#logit shape is [decoder_step_num, sample_num, target_vocab_size]

	logit = get_layer_constr('ff')(tparams, logit, options,
	                               prefix='ff_logit', activ='linear')
	logit_shp = logit.shape
	probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
	                                           logit_shp[2]]))

	# cost
	y_flat = y.flatten()
	y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
	cost = -tensor.log(probs.flatten()[y_flat_idx])
	cost = cost.reshape([y.shape[0], y.shape[1]])
	cost = (cost * y_mask).sum(0)

	# print "Print out in build_model()"
	# print opt_ret

	cost = init_state
	return trng, use_noise, x, x_mask, y, y_mask, ctx, ctx_mean, new_mask, word_ctx, init_state, cost, phrase_num, wq_src_len, beg_tmp, end_tmp


# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):
	"""
		commnet: x shape is [factor_num, step_num, sample_num]
	"""
	x = tensor.tensor3('x', dtype='int64')
	xr = x[:, ::-1]
	n_timesteps = x.shape[1]
	n_samples = x.shape[2]

	"""
		comment: emb shape is [step_num, sample_num, embedding_size]
				  embr shape is same
	"""
	# word embedding (source), forward and backward
	emb = []
	embr = []
	for factor in range(options['factors']):
		emb.append(tparams[embedding_name(factor)][x[factor].flatten()])
		embr.append(tparams[embedding_name(factor)][xr[factor].flatten()])
	emb = concatenate(emb, axis=1)
	embr = concatenate(embr, axis=1)
	emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
	embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

	if options['use_dropout']:
		retain_probability_emb = 1 - options['dropout_embedding']
		retain_probability_hidden = 1 - options['dropout_hidden']
		retain_probability_source = 1 - options['dropout_source']
		retain_probability_target = 1 - options['dropout_target']
		rec_dropout = theano.shared(numpy.array([retain_probability_hidden] * 2, dtype='float32'))
		rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden] * 2, dtype='float32'))
		rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden] * 5, dtype='float32'))
		emb_dropout = theano.shared(numpy.array([retain_probability_emb] * 2, dtype='float32'))
		emb_dropout_r = theano.shared(numpy.array([retain_probability_emb] * 2, dtype='float32'))
		emb_dropout_d = theano.shared(numpy.array([retain_probability_emb] * 2, dtype='float32'))
		ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden] * 4, dtype='float32'))
		source_dropout = theano.shared(numpy.float32(retain_probability_source))
		target_dropout = theano.shared(numpy.float32(retain_probability_target))
		emb *= source_dropout
		embr *= source_dropout
	else:
		rec_dropout = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		rec_dropout_r = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		rec_dropout_d = theano.shared(numpy.array([1.] * 5, dtype='float32'))
		emb_dropout = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		emb_dropout_r = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		emb_dropout_d = theano.shared(numpy.array([1.] * 2, dtype='float32'))
		ctx_dropout_d = theano.shared(numpy.array([1.] * 4, dtype='float32'))

	"""
		commnet: proj shape is [step_num, sample_num, hidden_size]
		proj sequence: h_1,h_2,...,h_m
		projr sequence: h_m, h_m-1...,h_1
	"""
	# encoder
	proj = get_layer_constr(options['encoder'])(tparams, emb, options,
	                                            prefix='encoder', emb_dropout=emb_dropout, rec_dropout=rec_dropout,
	                                            profile=profile)

	projr = get_layer_constr(options['encoder'])(tparams, embr, options,
	                                             prefix='encoder_r', emb_dropout=emb_dropout_r,
	                                             rec_dropout=rec_dropout_r, profile=profile)

	"""
		commnet: ctx shape is [step_num, sample_num, 2*hidden_size]
	"""
	# concatenate forward and backward rnn hidden states
	ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

	"""
		commnet: ctx_mean shape is [sample_num, 2*hidden_size], this 'encoder average hidden state' is used
		to transfer to 'decoder initial hidden state'
	"""
	# get the input for decoder rnn initializer mlp
	ctx_mean = ctx.mean(0)
	# ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

	if options['use_dropout']:
		ctx_mean *= retain_probability_hidden

	"""
		comment: 'decoder initial hidden state' is learned by ffnn according to 'encoder average hidden state'
	"""
	init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
	                                    prefix='ff_state', activ='tanh')

	print >> sys.stderr, 'Building f_init...',
	"""
		f_init function:
			input:
				input sequence 'x', shape [factor_num, step_num, sample_num]
			output:
				decoder initial hidden state 'init_state', shape [sample_num, hidden_size]
				encoder each step hidden state 'ctx', shape [step_num, sample_num, 2*hidden_size]
	"""
	start_time = time.time()
	outs = [init_state, ctx]
	f_init = theano.function([x], outs, name='f_init', profile=profile)
	print >> sys.stderr, 'Done [%.2fs]' % (time.time() - start_time)

	# x: 1 x 1
	"""
		commnet:
			y shape is [sample_num]
			init_state shape is [sample_num, hidden_size]
	"""
	y = tensor.vector('y_sampler', dtype='int64')
	init_state = tensor.matrix('init_state', dtype='float32')

	"""
		comment: emb shape is [sample_num, target_word_embedding]
	"""
	# if it's the first word, emb should be all zero and it is indicated by -1
	emb = tensor.switch(y[:, None] < 0,
	                    tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
	                    tparams['Wemb_dec'][y])

	if options['use_dropout']:
		emb *= target_dropout

	# apply one step of conditional gru with attention
	proj = get_layer_constr(options['decoder'])(tparams, emb, options,
	                                            prefix='decoder',
	                                            mask=None, context=ctx,
	                                            one_step=True,
	                                            init_state=init_state,
	                                            emb_dropout=emb_dropout_d,
	                                            ctx_dropout=ctx_dropout_d,
	                                            rec_dropout=rec_dropout_d,
	                                            profile=profile)
	# get the next hidden state
	next_state = proj[0]

	# get the weighted averages of context for this target word y
	ctxs = proj[1]

	# alignment matrix (attention model)
	dec_alphas = proj[2]

	if options['use_dropout']:
		next_state_up = next_state * retain_probability_hidden
		emb *= retain_probability_emb
		ctxs *= retain_probability_hidden
	else:
		next_state_up = next_state

	logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
	                                    prefix='ff_logit_lstm', activ='linear')
	logit_prev = get_layer_constr('ff')(tparams, emb, options,
	                                    prefix='ff_logit_prev', activ='linear')
	logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
	                                   prefix='ff_logit_ctx', activ='linear')
	logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)

	if options['use_dropout']:
		logit *= retain_probability_hidden

	logit = get_layer_constr('ff')(tparams, logit, options,
	                               prefix='ff_logit', activ='linear')
	"""
		next_probs shape is [sample_num, target_vocab_size]
	"""
	# compute the softmax probability
	next_probs = tensor.nnet.softmax(logit)

	# sample from softmax distribution to get the sample
	next_sample = trng.multinomial(pvals=next_probs).argmax(1)

	# compile a function to do the whole thing above, next word probability,
	# sampled word for the next target, next hidden state to be used
	print >> sys.stderr, 'Building f_next..',
	start_time = time.time()
	inps = [y, ctx, init_state]
	outs = [next_probs, next_sample, next_state]

	if return_alignment:
		outs.append(dec_alphas)
	"""
		f_next function:
			input:
				previous target word 'y_i-1', shape [sample_num]
				context vector 'h', shape [encoder_step_num ,sample, context_size]
				previous decoder hidden state 's_i-1', shape [sample_num, hidden_size]
			output:
				probability distribution of next target word 'p(y_i)', shape [sample_num, target_vocab_size]
				next target word 'y_i', shape [sample_num]
				next decoder hidden state 's_i', shape [sample_num, hidden_size]
				alignment(optional), shape [sample, sample_num, encoder_step_num]
	"""
	f_next = theano.function(inps, outs, name='f_next', profile=profile)
	print >> sys.stderr, 'Done [%.2fs]' % (time.time() - start_time)

	return f_init, f_next


"""
    generate some translation samples:
    support ensemble decoding(multi-model), beam-search

    parameters:
        f_init: a list of f_init function, c,init_state = f(x)
        f_next: a list of f_next function, p(y_i), y_i, s_i, align(optional) = f(y_i-1, s_i-1, c)
        x: input(wid), shape is [factor_num, encoder_step_num, sample_num]
            in training, x shape is [factor_num, encoder_step_num, 1], each time only one sample
        trng: random stream
        k: beam size
        maxlen: max translation length, more than 'maxlen' will not translate
        stochastic:

    return:
        sample: k-best target sequence(wid), shape is [k, sequence_len], sequence_len is not a fix number
            e.g. first target sequence length is 3, second one is 5
        score: a list of score of every target sequence, shape is [k]
        sample_word_prob: every element is probability of word, shape is [k, sequence_len]
        alignment: attention weight between every source word and target word of k-best target sequence
            shape is [k, sequence_len, source_word_count]
"""


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False):
	# k is the beam size we have
	if k > 1:
		assert not stochastic, \
			'Beam search does not support stochastic sampling'

	sample = []
	sample_score = []
	sample_word_probs = []
	alignment = []
	if stochastic:
		sample_score = 0

	live_k = 1
	dead_k = 0

	hyp_samples = [[]] * live_k
	word_probs = [[]] * live_k
	hyp_scores = numpy.zeros(live_k).astype('float32')
	hyp_states = []
	if return_alignment:
		hyp_alignment = [[] for _ in xrange(live_k)]

	# for ensemble decoding, we keep track of states and probability distribution
	# for each model in the ensemble
	num_models = len(f_init)
	next_state = [None] * num_models
	ctx0 = [None] * num_models
	next_p = [None] * num_models
	dec_alphas = [None] * num_models
	# get initial state of decoder rnn and encoder context
	"""
		f_init()
		input:
			1. id, shape [factor_num, encoder_step_num, sample_num]
		returns:
			1. init_state of decoder, shape is [sample_num, hidden_size]
			2. context (every step hidden state of encoder ), shape is [encoder_step_num, sample_num, context_size]
	"""
	for i in xrange(num_models):
		ret = f_init[i](x)
		next_state[i] = ret[0]
		ctx0[i] = ret[1]
	next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

	# x is a sequence of word ids followed by 0, eos id
	"""
		f_next function:
			input:
				previous target word 'y_i-1', shape [sample_num]
				context vector 'h', shape [encoder_step_num ,sample, context_size]
				previous decoder hidden state 's_i-1', shape [sample_num, hidden_size]
			output:
				probability distribution of next target word 'p(y_i)', shape [sample_num, target_vocab_size]
				next target word 'y_i', shape [sample_num]
				next decoder hidden state 's_i', shape [sample_num, hidden_size]
				alignment(optional), shape [decoder_step_num, sample_num, encoder_step_num]

	"""
	for ii in xrange(maxlen):
		for i in xrange(num_models):
			ctx = numpy.tile(ctx0[i], [live_k, 1])
			inps = [next_w, ctx, next_state[i]]
			# print 'step_%d | model_%d:' %(ii,i)
			# print 'next_w shape:', str(next_w.shape)
			# print 'next_w: ',str(next_w)
			# print 'ctx shape:', str(ctx.shape)
			# print 'next_state shape:',str(next_state[i].shape)
			ret = f_next[i](*inps)
			# dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
			"""
				next_p shape is [model_num, sample_num, target_vocab_size]
				next_state shape is [model_num, sample_num, hidden_size]
				dec_alphas shape is [model_num, decoder_step_num, sample_num, encoder_step_num]
			"""
			next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
			if return_alignment:
				dec_alphas[i] = ret[3]

			# if we don't allow to output any <unk>, we can set p(<unk>)=-inf
			if suppress_unk:
				next_p[i][:, 1] = -numpy.inf
		if stochastic:
			"""
				find max sigma(Pw)
			"""
			if argmax:
				nw = sum(next_p)[0].argmax()
			else:
				nw = next_w_tmp[0]
			sample.append(nw)
			sample_score += numpy.log(next_p[0][0, nw])
			if nw == 0:
				break
		else:
			"""
				cand_scores -simga (log(P_i(w))), shape is [sample_num, target_vocab_size]

				probs is algorithm average of all model probability, shape is [sample_num, target_vocab_size]
				probs = (sigma P_i(w))/N

				mean_alignment is algorithm average of all model attention weight
				shape is [decoder_step_num, sample_num, encoder_step_num]

				cand_flat shape is [sample_num * target_vocab_size]
				probs_flat shape is [sample_num * target_vocab_size]

			"""
			cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
			# print 'cand_score shape:',str(cand_scores.shape)
			probs = sum(next_p) / num_models
			cand_flat = cand_scores.flatten()
			probs_flat = probs.flatten()
			ranks_flat = cand_flat.argpartition(k - dead_k - 1)[:(k - dead_k)]
			# print 'ranks_flat shape:',str(ranks_flat.shape)

			# averaging the attention weights accross models
			if return_alignment:
				mean_alignment = sum(dec_alphas) / num_models

			"""
				voc_size is vocabulary size of target
			"""
			voc_size = next_p[0].shape[1]
			# index of each k-best hypothesis
			trans_indices = ranks_flat / voc_size
			word_indices = ranks_flat % voc_size
			costs = cand_flat[ranks_flat]

			new_hyp_samples = []
			new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
			new_word_probs = []
			new_hyp_states = []
			if return_alignment:
				# holds the history of attention weights for each time step for each of the surviving hypothesis
				# dimensions (live_k * target_words * source_hidden_units]
				# at each time step we append the attention weights corresponding to the current target word
				new_hyp_alignment = [[] for _ in xrange(k - dead_k)]

			# ti -> index of k-best hypothesis
			for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
				new_hyp_samples.append(hyp_samples[ti] + [wi])
				new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
				new_hyp_scores[idx] = copy.copy(costs[idx])
				new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
				if return_alignment:
					# get history of attention weights for the current hypothesis
					new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
					# extend the history with current attention weights
					new_hyp_alignment[idx].append(mean_alignment[ti])

			# check the finished samples
			new_live_k = 0
			hyp_samples = []
			hyp_scores = []
			hyp_states = []
			word_probs = []
			if return_alignment:
				hyp_alignment = []

			# sample and sample_score hold the k-best translations and their scores
			for idx in xrange(len(new_hyp_samples)):
				if new_hyp_samples[idx][-1] == 0:
					sample.append(new_hyp_samples[idx])
					sample_score.append(new_hyp_scores[idx])
					sample_word_probs.append(new_word_probs[idx])
					if return_alignment:
						alignment.append(new_hyp_alignment[idx])
					dead_k += 1
				# print 'find a seq: ', str(new_hyp_samples[idx])
				else:
					new_live_k += 1
					hyp_samples.append(new_hyp_samples[idx])
					hyp_scores.append(new_hyp_scores[idx])
					hyp_states.append(new_hyp_states[idx])
					word_probs.append(new_word_probs[idx])
					if return_alignment:
						hyp_alignment.append(new_hyp_alignment[idx])
					# print 'hyp_sample:',str(hyp_samples)
			hyp_scores = numpy.array(hyp_scores)

			live_k = new_live_k

			if new_live_k < 1:
				break
			if dead_k >= k:
				break

			next_w = numpy.array([w[-1] for w in hyp_samples])
			next_state = [numpy.array(state) for state in zip(*hyp_states)]

	if not stochastic:
		# dump every remaining one
		if live_k > 0:
			for idx in xrange(live_k):
				sample.append(hyp_samples[idx])
				sample_score.append(hyp_scores[idx])
				sample_word_probs.append(word_probs[idx])
				if return_alignment:
					alignment.append(hyp_alignment[idx])

	if not return_alignment:
		alignment = [None for i in range(len(sample))]

	return sample, sample_score, sample_word_probs, alignment


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False, alignweights=False):
	probs = []
	n_done = 0

	alignments_json = []

	for x, y in iterator:
		# ensure consistency in number of factors
		if len(x[0][0]) != options['factors']:
			sys.stderr.write(
				'Error: mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(
					options['factors'], len(x[0][0])))
			sys.exit(1)

		n_done += len(x)

		x, x_mask, y, y_mask = prepare_data(x, y,
		                                    n_words_src=options['n_words_src'],
		                                    n_words=options['n_words'])

		### in optional save weights mode.
		if alignweights:
			pprobs, attention = f_log_probs(x, x_mask, y, y_mask)
			for jdata in get_alignments(attention, x_mask, y_mask):
				alignments_json.append(jdata)
		else:
			pprobs = f_log_probs(x, x_mask, y, y_mask)

		# normalize scores according to output length
		if normalize:
			lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask.T])
			pprobs /= lengths

		for pp in pprobs:
			probs.append(pp)

		if numpy.isnan(numpy.mean(probs)):
			ipdb.set_trace()

		if verbose:
			print >> sys.stderr, '%d samples computed' % (n_done)

	return numpy.array(probs), alignments_json


"""
    Attention Neural Network:
    for encoder, use bi-directional RNN(gru):
        +encoder_h = gru(x_i, +h_i-1), +h_0 is zero-vector, get forward hidden state of encoder
        -encoder_h = gru(xr_i, -h_i-1), -h_0 is zero-vector, get backward hidden state of encoder
        encoder_h = [+encoder_h, -encoder_h], concatenate forward & backward hidden state
    for decoder:
        recurrent(conditional gru):
            1. fake_new_state:
                s_i~ = gru(s_i-1, y_i-1)
            note s_0 is learned by s_o = tanh(w*(avg_h) + b)
            2. attention:
                as raw paper description, c_i = f(s_i-1, encoder_h), but in code, I found c_i = f(s_i~, encoder_h)
                c_i = sigma a_ij*h_j
                    c_i is linear weighted hidden state of encoder, as a context to predict y_i
                a_ij = k(s_i~, h_j)
                    k() is a ffnn, specifically, e_ij = w_ho *(a(w_ih1*s_i~, w_ih2*h_j + b_ih)) + b_ho, a() is a activation function,
                a_ij = softmax([e_i1,e_i2,,,e_im])
            3. new_state
                s_i = gru(s_i~, c_i)

        prediction:
            y_i = g(s_i, y_i-1, c_i)
                g() is a ffnn actually, but due to shape of 's_i', 'y_i-1', 'c_i' is different, so we can't use a ffnn directly.
                specifically, we convert 's_i', 'y_i-1', 'c_i' into a hidden layer with size = target_embedding_size
                o_s = Ws*s_i + bs
                o_y = Wy*y_i-1 + by
                o_c = Wc*c_i + bc
                then, a = tanh(o_s + o_y + o_c), not concatenate, just add
                y_i = softmax(wa + b)

            so, Ws shape is [hidden_size, embedding_size], bs shape is [embedding_size]
                Wy shape is [embedding_size, embedding_size], by shape is [embedding_size]
                Wc shape is [context_size, embedding_size], bc shape is [embedding_size], here context_size = 2*hidden_size(bi-rnn)
                w shape is [embedding_size, target_vocab_size], b shape is [target_vocab_size]


        note: we only init recurrent unit weight with ortho, for other parameters, we use normal_init

"""


def get_bleu_history(file_bleu_history):
	_file = open(file_bleu_history)
	bleu_history = []
	for line in _file:
		bleu_history.append(float(line.strip()))
	_file.close()
	current_bleu = bleu_history[-1]

	return current_bleu, bleu_history


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          factors=1,  # input factors
          dim_per_factor=None,
          # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0.,  # L2 regularization penalty towards original weights
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=None,  # source vocabulary size
          n_words=None,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          datasets=[
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          valid_source_set=None,  # if validation-set includes multi-references, specific the unique source
          niutrans_dev=None,  # used only for validation by bleu, if use, set the niutrans-format validation-set
          ref_num=1,
          dictionaries=[
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          dropout_embedding=0.2,  # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.5,  # dropout for hidden layers (0: no dropout)
          dropout_source=0,  # dropout source words (0: no dropout)
          dropout_target=0,  # dropout target words (0: no dropout)
          reload_=False,  # default False
          overwrite=False,
          external_validation_script=None,
          early_stop_flag='COST',  # value is in ['COST', 'BLEU']
          shuffle_each_epoch=True,
          finetune=False,
          finetune_only_last=False,
          sort_by_length=True,
          use_domain_interpolation=False,
          domain_interpolation_min=0.1,
          domain_interpolation_inc=0.1,
          domain_interpolation_indomain_datasets=['indomain.en', 'indomain.fr'],
          maxibatch_size=20,
          validation_after_updates=0,
          use_phrase=False,
          ngram=2):  # How many minibatches to load at one time

	# Model options
	model_options = locals().copy()

	if model_options['dim_per_factor'] == None:
		if factors == 1:
			model_options['dim_per_factor'] = [model_options['dim_word']]
		else:
			sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
			sys.exit(1)

	assert (len(dictionaries) == factors + 1)  # one dictionary per source factor + 1 for target factor
	assert (len(model_options['dim_per_factor']) == factors)  # each factor embedding has its own dimensionality
	assert (sum(model_options['dim_per_factor']) == model_options[
		'dim_word'])  # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
	assert (early_stop_flag in ['COST', 'BLEU'])
	if early_stop_flag == 'BLEU':
		assert (external_validation_script != None and niutrans_dev != None)
	if ref_num > 1:
		assert (valid_source_set != None)

	"""
		comment: a word can include many factors, such as lexical, pos-of-tagging, synmatic label..
		any factor can project with some-dim embeding
		all factor-embeding composed together, we can get a whole embeding of a word
	"""

	# load dictionaries and invert them
	worddicts = [None] * len(dictionaries)
	worddicts_r = [None] * len(dictionaries)
	for ii, dd in enumerate(dictionaries):
		worddicts[ii] = load_dict(dd)
		worddicts_r[ii] = dict()
		for kk, vv in worddicts[ii].iteritems():
			worddicts_r[ii][vv] = kk
	"""
		comment: worddicts maps 'label' -> 'id', 'label' can be lexical or something else
				 worddicts_r maps 'id' -> 'label'
	"""

	if n_words_src is None:
		n_words_src = len(worddicts[0])
		model_options['n_words_src'] = n_words_src
	if n_words is None:
		n_words = len(worddicts[1])
		model_options['n_words'] = n_words

	# reload options
	if reload_ and os.path.exists(saveto):
		print 'Reloading model options'
		try:
			with open('%s.json' % saveto, 'rb') as f:
				loaded_model_options = json.load(f)
		except:
			with open('%s.pkl' % saveto, 'rb') as f:
				loaded_model_options = pkl.load(f)
		model_options.update(loaded_model_options)

	print 'Loading data'
	domain_interpolation_cur = None
	if use_domain_interpolation:
		print 'Using domain interpolation with initial ratio %s, increase rate %s' % (
		domain_interpolation_min, domain_interpolation_inc)
		domain_interpolation_cur = domain_interpolation_min
		train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
		                                       dictionaries[:-1], dictionaries[1],
		                                       n_words_source=n_words_src, n_words_target=n_words,
		                                       batch_size=batch_size,
		                                       maxlen=maxlen,
		                                       shuffle_each_epoch=shuffle_each_epoch,
		                                       sort_by_length=sort_by_length,
		                                       indomain_source=domain_interpolation_indomain_datasets[0],
		                                       indomain_target=domain_interpolation_indomain_datasets[1],
		                                       interpolation_rate=domain_interpolation_cur,
		                                       maxibatch_size=maxibatch_size)
	else:
		"""
			dictionaries[:-1]: put N-1 dict, each dict is used in a factor of source
			dictionaries[-1]: put the last dict, this dict is used for target

			'train' and 'valid' is iterable object, function 'next()' can return a group of mini-batch
			a mini-batch includes 'batch_size' samples
			a group of mini-batch includes 'maxibatch_size' mini-batch
		"""
		train = TextIterator(datasets[0], datasets[1],
		                     dictionaries[:-1], dictionaries[-1],
		                     n_words_source=n_words_src, n_words_target=n_words,
		                     batch_size=batch_size,
		                     maxlen=maxlen,
		                     shuffle_each_epoch=False,
		                     sort_by_length=False,
		                     maxibatch_size=maxibatch_size)
	valid = TextIterator(valid_datasets[0], valid_datasets[1],
	                     dictionaries[:-1], dictionaries[-1],
	                     n_words_source=n_words_src, n_words_target=n_words,
	                     batch_size=valid_batch_size,
	                     maxlen=maxlen)

	print 'Building model'
	"""
		init or reload all params
		This create the initial parameters as numpy ndarrays.
		Dict name (string) -> numpy ndarray
	"""
	params = init_params(model_options)
	# reload parameters
	if reload_ and os.path.exists(saveto):
		print 'Reloading model parameters'
		params = load_params(saveto, params)

	"""
		comment: convert all params into theano shared variable
		This create Theano Shared Variable from the parameters.
		Dict name (string) -> Theano Tensor Shared Variable
		params and tparams have different copy of the weights.

	"""
	tparams = init_theano_params(params)

	"""
		commnet:     use_noise is for dropout
	"""
	trng, use_noise, \
	x, x_mask, y, y_mask, \
	opt_ret, \
	cost = \
		build_model(tparams, model_options)

	inps = [x, x_mask, y, y_mask]

	print 'Building sampler'
	f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

	# before any regularizer
	"""
		commnet: 'f_log_probs' is a function to get log_prob in mini-batch, not include any regularization
		such as 'L2'
		f_log_probs:
		input:
			x: shape is [factor_num, encoder_step_num, sample_num]
			x_mask: shape is [encoder_step_num, sample_num]
			y: shape is [decoder_step_num, sample_num]
			y_mask: shape is [decoder_step_num, sample_num]
		output:
			cost: sentence-level likelyhood, '-sigma_log(Pi)', shape is [sample_num]
	"""
	print 'Building f_log_probs...',
	f_log_probs = theano.function(inps, cost, profile=profile)
	print 'Done'

	cost = cost.mean()

	# apply L2 regularization on weights
	"""
		comment: L2 regularization for all weights
		'decay_c' is a theano shared variable
	"""
	if decay_c > 0.:
		decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
		weight_decay = 0.
		for kk, vv in tparams.iteritems():
			weight_decay += (vv ** 2).sum()
		weight_decay *= decay_c
		cost += weight_decay

	# regularize the alpha weights
	if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
		alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
		alpha_reg = alpha_c * (
			(tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
			 opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
		cost += alpha_reg

	# apply L2 regularisation to loaded model (map training)
	if map_decay_c > 0:
		map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
		weight_map_decay = 0.
		for kk, vv in tparams.iteritems():
			init_value = theano.shared(vv.get_value(), name=kk + "_init")
			weight_map_decay += ((vv - init_value) ** 2).sum()
		weight_map_decay *= map_decay_c
		cost += weight_map_decay

	"""
		f_cost is a function like 'f_log_probs', but the cost include all kinds of regularization
		input:
			same with 'f_log_probs'
		output:
			likelyhood + L2 + attention weight + something else.., just a scalar
	"""
	# after all regularizers - compile the computational graph for cost
	print 'Building f_cost...',
	f_cost = theano.function(inps, cost, profile=profile)
	print 'Done'

	# allow finetuning with fixed embeddings
	"""
		comment: fine tuning is used for optimize parameters except embedding(every factor in source and target)
	"""
	if finetune:
		updated_params = OrderedDict(
			[(key, value) for (key, value) in tparams.iteritems() if not key.startswith('Wemb')])
	else:
		updated_params = tparams

	# allow finetuning of only last layer (becomes a linear model training problem)
	if finetune_only_last:
		updated_params = OrderedDict(
			[(key, value) for (key, value) in tparams.iteritems() if key in ['ff_logit_W', 'ff_logit_b']])
	else:
		updated_params = tparams

	print 'Computing gradient...',
	grads = tensor.grad(cost, wrt=itemlist(updated_params))
	print 'Done'

	# apply gradient clipping here
	"""
		comment: gradient clipping is used for preventing the updated gradient is too big, so that cause 'gradient explosiion'
	"""
	if clip_c > 0.:
		g2 = 0.
		for g in grads:
			g2 += (g ** 2).sum()
		new_grads = []
		for g in grads:
			new_grads.append(tensor.switch(g2 > (clip_c ** 2),
			                               g / tensor.sqrt(g2) * clip_c,
			                               g))
		grads = new_grads

	# compile the optimizer, the actual computational graph is compiled here
	lr = tensor.scalar(name='lr')

	"""
		comment: optimizer, update parameters
	"""
	print 'Building optimizers...',
	f_grad_shared, f_update = eval(optimizer)(lr, updated_params, grads, inps, cost, profile=profile)
	print 'Done'

	"""
		comment: training step
	"""
	print 'Optimization'

	best_p = None
	bad_counter = 0
	uidx = 0  # the number of update done
	estop = False  # early stop
	history_errs = []  # history error, cost or bleu on validation
	# reload history
	if reload_ and os.path.exists(saveto):
		rmodel = numpy.load(saveto)
		history_errs = list(rmodel['history_errs'])
		if 'uidx' in rmodel:
			uidx = rmodel['uidx']

	if validFreq == -1:
		validFreq = len(train[0]) / batch_size
	if saveFreq == -1:
		saveFreq = len(train[0]) / batch_size
	if sampleFreq == -1:
		sampleFreq = len(train[0]) / batch_size

	for eidx in xrange(max_epochs):
		epoch_start = time.time()  # add by lifuxue
		n_samples = 0
		for x, y in train:
			n_samples += len(x)
			uidx += 1
			use_noise.set_value(1.)

			# ensure consistency in number of factors
			if len(x) and len(x[0]) and len(x[0][0]) != factors:
				sys.stderr.write(
					'Error: mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(
						factors, len(x[0][0])))
				sys.exit(1)

			# Get the data in numpy.ndarray format
			# This swap the axis!
			# Return something of shape (minibatch maxlen, n samples)
			"""
				x: shape is [factor_num, encoder_step_num, sample_num]
				x_mask: padding mask of x, shape is [encoder_step_num, sample_num]
				y: shape is [decoder_step_num, sample_num]
				y_mask: padding mask of y, shape is [decoder_step_num, sample_num]
			"""
			lengths_y = [len(s) for s in y]
			target_words_count = sum(lengths_y)

			x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
			                                    n_words_src=n_words_src,
			                                    n_words=n_words)
			print 'src_len:%d | batch:%d ' % (x.shape[1], x.shape[2])

			if x is None:
				print 'Minibatch with zero sample under length ', maxlen
				uidx -= 1
				continue

			ud_start = time.time()

			# compute cost, grads and copy grads to shared variables
			cost = f_grad_shared(x, x_mask, y, y_mask)

			# do the update on parameters
			f_update(lrate)

			ud = time.time() - ud_start

			# check for bad numbers, usually we remove non-finite elements
			# and continue training - but not done here
			if numpy.isnan(cost) or numpy.isinf(cost):
				print 'NaN detected'
				return 1., 1., 1.

			# verbose
			if numpy.mod(uidx, dispFreq) == 0:
				# print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
				# print 'Epoch ', eidx, 'Update ', uidx , 'Cost ', cost, 'time consume %.2f seconds' % ud ,'generate target words ', target_words_count, 'speed %.2f ws/sec' % (target_words_count*1.0 / ud) # add by lifuxue
				print 'Epoch [%d] Update [%d] Cost [%.2f] Time(a batch) [%.2f]s Target Word(a batch) [%d] Speed [%.2f]w/s' % (
				eidx, uidx, cost, ud, target_words_count, target_words_count * 1.0 / ud)

			"""
				comment: save model
			"""
			# save the best model so far, in addition, save the latest model
			# into a separate file with the iteration number for external eval
			if numpy.mod(uidx, saveFreq) == 0:
				save_model_start = time.time()  # add by lifuxue
				print 'Saving the best model...',
				if best_p is not None:
					params = best_p
				else:
					params = unzip_from_theano(tparams)
				numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
				json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)
				print 'Done'

				# save with uidx
				if not overwrite:
					print 'Saving the model at iteration {}...'.format(uidx),
					saveto_uidx = '{}.iter{}.npz'.format(
						os.path.splitext(saveto)[0], uidx)
					numpy.savez(saveto_uidx, history_errs=history_errs,
					            uidx=uidx, **unzip_from_theano(tparams))
					print 'Done'
					save_model_cost = time.time() - save_model_start
					print "Saving model time consume %.2f seconds" % save_model_cost  # add by lifuxue
			"""
				comment: dump some samples
			"""
			gen_time_start = time.time()
			generate_word_count = 0
			# generate some samples with the model and display them
			if numpy.mod(uidx, sampleFreq) == 0:
				# FIXME: random selection?
				"""
					dump N=min(5, sample_num in this batch) samples
					x shape is [factor_num, encoder_step_num, sampe_num]
				"""
				sample_sent_count = numpy.minimum(5, x.shape[2])
				for jj in xrange(sample_sent_count):
					stochastic = True
					sample, score, sample_word_probs, alignment = gen_sample([f_init], [f_next],
					                                                         x[:, :, jj][:, :, None],
					                                                         trng=trng, k=1,
					                                                         maxlen=30,
					                                                         stochastic=stochastic,
					                                                         argmax=False,
					                                                         suppress_unk=False)
					print 'Source ', jj, ': ',
					for pos in range(x.shape[1]):
						if x[0, pos, jj] == 0:
							break
						for factor in range(factors):
							vv = x[factor, pos, jj]
							if vv in worddicts_r[factor]:
								sys.stdout.write(worddicts_r[factor][vv])
							else:
								sys.stdout.write('<UNK>')
							if factor + 1 < factors:
								sys.stdout.write('||')
							else:
								sys.stdout.write(' ')
					print
					print 'Truth ', jj, ' : ',
					for vv in y[:, jj]:
						if vv == 0:
							break
						if vv in worddicts_r[-1]:
							print worddicts_r[-1][vv],
						else:
							print '<UNK>',
					print
					print 'Sample ', jj, ': ',
					if stochastic:
						ss = sample
					else:
						score = score / numpy.array([len(s) for s in sample])
						ss = sample[score.argmin()]
					for vv in ss:
						if vv == 0:
							break
						if vv in worddicts_r[-1]:
							print worddicts_r[-1][vv],
						else:
							print '<UNK>',
					print
					generate_word_count += len(ss)
				gen_time = time.time() - gen_time_start
				print "Generate [%d] sent | [%d] target words | using [%.2f]s | speed [%.2f]w/s" % (
				sample_sent_count, generate_word_count, gen_time, generate_word_count * 1.0 / gen_time)
			"""
				commnet: test on validation set
			"""
			# validate model on validation set and early stop if necessary
			if uidx >= validation_after_updates and numpy.mod(uidx - validation_after_updates, validFreq) == 0:
				valid_start = time.time()  # add by lifuxue
				if early_stop_flag == "COST":
					use_noise.set_value(0.)
					valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
					                                   model_options, valid)
					valid_err = valid_errs.mean()
				# use '-bleu' as validation error, so it's compatible with 'cost', the less value is, the better result is
				elif early_stop_flag == 'BLEU':
					if external_validation_script:
						valid_external_start = time.time()
						print "Calling external validation script: %s" % (external_validation_script)
						# usage: prefix_of_model | source_of_dev | niutrans_form_dev | ref_num
						prefix = os.path.splitext(saveto)[0]
						p = Popen([external_validation_script, prefix, valid_source_set, niutrans_dev, str(ref_num)])
						p.wait()
						current_bleu, bleu_history = get_bleu_history("%s_bleu_history" % prefix)
						valid_err = -current_bleu

				history_errs.append(valid_err)
				if uidx - validation_after_updates == 0 or valid_err <= numpy.array(history_errs).min():
					best_p = unzip_from_theano(tparams)
					bad_counter = 0
				if len(history_errs) > patience and valid_err >= \
						numpy.array(history_errs)[:-patience].min():
					bad_counter += 1
					if bad_counter > patience:
						if use_domain_interpolation and (domain_interpolation_cur < 1.0):
							domain_interpolation_cur = min(domain_interpolation_cur + domain_interpolation_inc, 1.0)
							print 'No progress on the validation set, increasing domain interpolation rate to %s and resuming from best params' % domain_interpolation_cur
							train.adjust_domain_interpolation_rate(domain_interpolation_cur)
							if best_p is not None:
								zip_to_theano(best_p, tparams)
							bad_counter = 0
						else:
							print 'Early Stop!'
							estop = True
							break
				if numpy.isnan(valid_err):
					ipdb.set_trace()

				valid_cost = time.time() - valid_start
				# turn '-bleu' into 'bleu'
				if early_stop_flag == 'BLEU': valid_err = -valid_err
				print 'Valid_Type:%s | Value:%.4f | Time:%.2f' % (early_stop_flag, valid_err, valid_cost)

			# finish after this many updates
			if uidx >= finish_after:
				print 'Finishing after %d iterations!' % uidx
				estop = True
				break

			# print 'Seen %d samples' % n_samples

		if estop:
			break
		epochCostTime = time.time() - epoch_start  # add by lifuxue
		print "Epoch %s time consume %.2f seconds" % (eidx, epochCostTime)  # add by lifuxue

	if best_p is not None:
		zip_to_theano(best_p, tparams)

	if early_stop_flag == "COST":
		use_noise.set_value(0.)
		valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
		                                   model_options, valid)
		valid_err = valid_errs.mean()
		print 'Valid ', valid_err

	if best_p is not None:
		params = copy.copy(best_p)
	else:
		params = unzip_from_theano(tparams)
	numpy.savez(saveto, zipped_params=best_p,
	            history_errs=history_errs,
	            uidx=uidx,
	            **params)

	return valid_err


def my_train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          factors=1,  # input factors
          dim_per_factor=None,
          # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0.,  # L2 regularization penalty towards original weights
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=None,  # source vocabulary size
          n_words=None,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          datasets=[
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          valid_source_set=None,  # if validation-set includes multi-references, specific the unique source
          niutrans_dev=None,  # used only for validation by bleu, if use, set the niutrans-format validation-set
          ref_num=1,
          dictionaries=[
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
	          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          dropout_embedding=0.2,  # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.5,  # dropout for hidden layers (0: no dropout)
          dropout_source=0,  # dropout source words (0: no dropout)
          dropout_target=0,  # dropout target words (0: no dropout)
          reload_=False,  # default False
          overwrite=False,
          external_validation_script=None,
          early_stop_flag='COST',  # value is in ['COST', 'BLEU']
          shuffle_each_epoch=True,
          finetune=False,
          finetune_only_last=False,
          sort_by_length=True,
          use_domain_interpolation=False,
          domain_interpolation_min=0.1,
          domain_interpolation_inc=0.1,
          domain_interpolation_indomain_datasets=['indomain.en', 'indomain.fr'],
          maxibatch_size=20,
          validation_after_updates=0,
          use_phrase=False,
          ngram=2):  # How many minibatches to load at one time

	# Model options
	model_options = locals().copy()

	if model_options['dim_per_factor'] == None:
		if factors == 1:
			model_options['dim_per_factor'] = [model_options['dim_word']]
		else:
			sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
			sys.exit(1)

	assert (len(dictionaries) == factors + 1)  # one dictionary per source factor + 1 for target factor
	assert (len(model_options['dim_per_factor']) == factors)  # each factor embedding has its own dimensionality
	assert (sum(model_options['dim_per_factor']) == model_options[
		'dim_word'])  # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
	assert (early_stop_flag in ['COST', 'BLEU'])
	if early_stop_flag == 'BLEU':
		assert (external_validation_script != None and niutrans_dev != None)
	if ref_num > 1:
		assert (valid_source_set != None)

	"""
		comment: a word can include many factors, such as lexical, pos-of-tagging, synmatic label..
		any factor can project with some-dim embeding
		all factor-embeding composed together, we can get a whole embeding of a word
	"""

	# load dictionaries and invert them
	worddicts = [None] * len(dictionaries)
	worddicts_r = [None] * len(dictionaries)
	for ii, dd in enumerate(dictionaries):
		worddicts[ii] = load_dict(dd)
		worddicts_r[ii] = dict()
		for kk, vv in worddicts[ii].iteritems():
			worddicts_r[ii][vv] = kk
	"""
		comment: worddicts maps 'label' -> 'id', 'label' can be lexical or something else
				 worddicts_r maps 'id' -> 'label'
	"""

	if n_words_src is None:
		n_words_src = len(worddicts[0])
		model_options['n_words_src'] = n_words_src
	if n_words is None:
		n_words = len(worddicts[1])
		model_options['n_words'] = n_words

	# reload options
	if reload_ and os.path.exists(saveto):
		print 'Reloading model options'
		try:
			with open('%s.json' % saveto, 'rb') as f:
				loaded_model_options = json.load(f)
		except:
			with open('%s.pkl' % saveto, 'rb') as f:
				loaded_model_options = pkl.load(f)
		model_options.update(loaded_model_options)


	shuffle_each_epoch = False
	print 'Loading data'
	domain_interpolation_cur = None
	if use_domain_interpolation:
		print 'Using domain interpolation with initial ratio %s, increase rate %s' % (
		domain_interpolation_min, domain_interpolation_inc)
		domain_interpolation_cur = domain_interpolation_min
		train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
		                                       dictionaries[:-1], dictionaries[1],
		                                       n_words_source=n_words_src, n_words_target=n_words,
		                                       batch_size=1,
		                                       maxlen=maxlen,
		                                       shuffle_each_epoch=False,
		                                       sort_by_length=False,
		                                       indomain_source=domain_interpolation_indomain_datasets[0],
		                                       indomain_target=domain_interpolation_indomain_datasets[1],
		                                       interpolation_rate=domain_interpolation_cur,
		                                       maxibatch_size=maxibatch_size)
	else:
		"""
			dictionaries[:-1]: put N-1 dict, each dict is used in a factor of source
			dictionaries[-1]: put the last dict, this dict is used for target

			'train' and 'valid' is iterable object, function 'next()' can return a group of mini-batch
			a mini-batch includes 'batch_size' samples
			a group of mini-batch includes 'maxibatch_size' mini-batch
		"""
		train = TextIterator(datasets[0], datasets[1],
		                     dictionaries[:-1], dictionaries[-1],
		                     n_words_source=n_words_src, n_words_target=n_words,
		                     batch_size=1,
		                     maxlen=maxlen,
		                     shuffle_each_epoch=False,
		                     sort_by_length=False,
		                     maxibatch_size=1)
	valid = TextIterator(valid_datasets[0], valid_datasets[1],
	                     dictionaries[:-1], dictionaries[-1],
	                     n_words_source=n_words_src, n_words_target=n_words,
	                     batch_size=valid_batch_size,
	                     maxlen=maxlen)

	print 'Building model'
	"""
		init or reload all params
		This create the initial parameters as numpy ndarrays.
		Dict name (string) -> numpy ndarray
	"""
	params = init_params(model_options)
	# reload parameters
	if reload_ and os.path.exists(saveto):
		print 'Reloading model parameters'
		params = load_params(saveto, params)

	"""
		comment: convert all params into theano shared variable
		This create Theano Shared Variable from the parameters.
		Dict name (string) -> Theano Tensor Shared Variable
		params and tparams have different copy of the weights.

	"""
	tparams = init_theano_params(params)

	"""
		commnet:     use_noise is for dropout
	"""
	trng, use_noise, \
	x, x_mask, y, y_mask,  \
	ctx, ctx_mean, new_mask, word_ctx, init_state, cost, phrase_num, wq_src_len, beg_tmp, end_tmp= \
		build_model(tparams, model_options)

	inps = [x, x_mask, y, y_mask]

	f_wq = theano.function(inps, [ctx,ctx_mean, new_mask, word_ctx, init_state, cost, phrase_num, wq_src_len, beg_tmp, end_tmp],on_unused_input='warn')

	"""
		comment: training step
	"""
	print 'Optimization'

	best_p = None
	bad_counter = 0
	uidx = 0  # the number of update done
	estop = False  # early stop
	history_errs = []  # history error, cost or bleu on validation
	# reload history
	if reload_ and os.path.exists(saveto):
		rmodel = numpy.load(saveto)
		history_errs = list(rmodel['history_errs'])
		if 'uidx' in rmodel:
			uidx = rmodel['uidx']

	if validFreq == -1:
		validFreq = len(train[0]) / batch_size
	if saveFreq == -1:
		saveFreq = len(train[0]) / batch_size
	if sampleFreq == -1:
		sampleFreq = len(train[0]) / batch_size

	for eidx in xrange(max_epochs):
		epoch_start = time.time()  # add by lifuxue
		n_samples = 0
		for x, y in train:
			n_samples += len(x)
			uidx += 1
			use_noise.set_value(1.)

			# ensure consistency in number of factors
			if len(x) and len(x[0]) and len(x[0][0]) != factors:
				sys.stderr.write(
					'Error: mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(
						factors, len(x[0][0])))
				sys.exit(1)

			# Get the data in numpy.ndarray format
			# This swap the axis!
			# Return something of shape (minibatch maxlen, n samples)
			"""
				x: shape is [factor_num, encoder_step_num, sample_num]
				x_mask: padding mask of x, shape is [encoder_step_num, sample_num]
				y: shape is [decoder_step_num, sample_num]
				y_mask: padding mask of y, shape is [decoder_step_num, sample_num]
			"""
			lengths_y = [len(s) for s in y]
			target_words_count = sum(lengths_y)
			x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
			                                    n_words_src=n_words_src,
			                                    n_words=n_words)

			print 'src_len:%d | batch:%d ' % (x.shape[1], x.shape[2])

			if x is None:
				print 'Minibatch with zero sample under length ', maxlen
				uidx -= 1
				continue

			my_ctx, my_ctx_mean, new_mask, my_word_ctx, my_init_state, cost, phrase_num, wq_src_len, beg_tmp, end_tmp = f_wq(x,x_mask,y,y_mask)
			print 'ctx:'
			print my_ctx
			print 'word ctx'
			print my_word_ctx
			print 'ctx mean'
			print my_ctx_mean
			print 'ctx shape:',my_ctx.shape
			print 'ctx mean shape:', my_ctx_mean.shape
			print 'word ctx shape:', my_word_ctx.shape
			print x_mask.shape
			print x_mask
			print new_mask.shape
			print new_mask
			print x.shape
			print x
			print my_init_state
			print cost
			print phrase_num
			print wq_src_len
			print beg_tmp
			print beg_tmp.shape
			print end_tmp
			print end_tmp.shape

			import pdb
			pdb.set_trace()



if __name__ == '__main__':
	pass

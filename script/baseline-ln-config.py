import numpy
import os
import sys
sys.path.append('../core/nematus')
from nmt import train
import theano

# directory for storing data
DATA_DIR = "../data"

# vocabulary size for source, default value is 30000
VOCAB_SIZE_SRC = 30000

# vocabulary size for target, default value is 30000
VOCAB_SIZE_TARGET = 30000

#theano.config.exception_verbosity='high'
#theano.config.scan.allow_gc=False

if __name__ == '__main__':
    validerr = train(saveto='../model/model.npz',
                    reload_=True,
                    reload_training_progress=True, 
                    dim_word=256, # word embedding  before:500
                    dim=256, # hidden state size    before:1000
                    n_words=VOCAB_SIZE_TARGET,
                    n_words_src=VOCAB_SIZE_SRC,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adam',
                    maxlen=80,
                    batch_size=80,#80
                    valid_batch_size=80,
                    dictionaries=[DATA_DIR + '/40w/train.ch.json', DATA_DIR + '/40w/train.en.json'],
                    datasets=[DATA_DIR + '/40w/train.ch', DATA_DIR + '/40w/train.en'],
                    #valid_datasets=[DATA_DIR + '/valid/', DATA_DIR + '/valid/newstest2013.tok.bpe.de'],
                    valid_datasets=None,
                    early_stop_flag = 'BLEU',
                    ref_num=4,
                    valid_source_set=DATA_DIR+'/valid/c.utf8.token.nogen',
                    niutrans_dev=DATA_DIR+'/valid/dev.txt',
                    validFreq=200, #1W
                    dispFreq=100,
                    saveFreq=5000,
                    sampleFreq=50000000,
                    use_dropout=True,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.0, # dropout source words (0: no dropout)
                    dropout_target=0.0, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate_by_bleu.sh',
                    validation_after_updates=0,#8W
                    patience=10,
                    objective="CE",
                    mrt_alpha=0.005,
                    mrt_samples=100,
                    mrt_samples_meanloss=10,
                    mrt_reference=False,
                    mrt_loss="SENTENCEBLEU n=4",  # loss function for minimum risk training
                    mrt_ml_mix=0,  # interpolate mrt loss with ML loss
                    #finetune=False,
                    #use_phrase=False,
                    #ngram=1,
                    #use_identifier=False,
                    #wq_prefix='wq_identifier',
                    #use_cat=True,
                    #use_boost=False,
                    #use_rnn_identifier=True,
                    #identifier_dim=100,
                    model_version=0.1,  # store version used for training for compatibility,
                    tie_encoder_decoder_embeddings=False, # Tie the input embeddings of the encoder and the decoder (first factor only)
                    tie_decoder_embeddings=False, # Tie the input embeddings of the decoder with the softmax output embeddings
                    encoder_truncate_gradient=-1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
                    decoder_truncate_gradient=-1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
                    #show_gradient_norm=False
                    layer_normalisation=True,
                    enc_depth=1,
                    dec_depth=1
                    )
    print validerr

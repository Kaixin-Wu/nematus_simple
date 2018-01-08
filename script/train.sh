# theano device
device=gpu0
script=phrase_nmt_train.py

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $script

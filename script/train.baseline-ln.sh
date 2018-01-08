# theano device
device=gpu3
script=baseline-ln-config.py

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,gpuarray.preallocate=6000 \
python -u $script

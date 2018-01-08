#coding:utf-8
__author__ = 'wang qiang'

"""
    This script can get a average model from multi-models.
    Instead of using multi-models in decoding, which computation cost is huge,
    by a average model, we can improve performance like multi-models ensembling,
    while just same computation cost as a single model
"""

#usage: model_1 model_2 .... model_n model_avg
import sys
import numpy
import time
if len(sys.argv) < 4:
    print '3 parameters at least! model_1 model_2 model_avg'
    sys.exit(-1)

input_model = sys.argv[1:-1]
avg_model = sys.argv[-1]

print 'total %d models' %(len(input_model))

start_time = time.time()
model_params = [numpy.load(path) for path in input_model]
print 'load parameters done ...'

key_set = set()
for param in model_params:
    for k in param.keys():
        key_set.add(k)
if 'uidx' in key_set:
        key_set.remove('uidx')
if 'history_errs' in key_set:
        key_set.remove('history_errs')
print 'total find %d parameter name' %len(key_set)
print key_set


param_num = 0
avg_params = dict()
for key in key_set:
    value_list = []
    for id,param in enumerate(model_params):
        # parameter name miss
        if key not in param:
            print 'model %s not include parameter %s' %(input_model[id], key)
            sys.exit(-1)
        value_list.append(param[key])
        #print 'key:%s param_%d:%s' %(key, id, str(param[key]))
    avg_params[key] = numpy.array(value_list).mean(axis=0)

print 'calculate average model done ...'

numpy.savez(avg_model, **avg_params)
print 'save average model done ...'

print 'total use %f seconds' %(time.time() - start_time)


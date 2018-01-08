# -*- coding: utf8 -*-
'''
This script is to replace the unknown words in target sentences with their aligned words in source sentences.
Args: 
	- input: a text file (json format), each line 
			including a full alignment matrix, a pair of source and target sentences
	- output (optional): updated text file (json format)
	- unknown word token (optional): a string, default="UNK"
To use:
	python replace_target_unknown_words.py -i translation.txt -o updated_translation.txt -u 'UNK'
'''

import json
import numpy
import argparse
import sys
from util import load_dict

reload(sys)
sys.setdefaultencoding( "utf-8" )
''' 
Example input file:
{"id": 0, "prob": 0, "target_sent": "Obama empfängt Netanjahu", "matrix": [[0.9239920377731323, 0.04680762067437172, 0.003626488381996751, 0.02343202754855156, 0.0021418146789073944], [0.009942686185240746, 0.4995519518852234, 0.44341862201690674, 0.02077348716557026, 0.026313267648220062], [0.01032756082713604, 0.6475557088851929, 0.029476342722773552, 0.27724361419677734, 0.035396818071603775], [0.0010026689851656556, 0.35200807452201843, 0.06362949311733246, 0.4778701961040497, 0.1054895892739296]], "source_sent": "Obama kindly receives Netanjahu"}
'''
def load_phrase_table(phrase_table):
    print "Loading PhraseTable"
    dic = {}
    num = 0
    for line in phrase_table:
        num += 1
        if num % 10000000 == 0:
            print "%s\tdone" % num
        tArr = line.strip().split(" ||| ")
        if tArr[1] in ["<NULL>", ""]:
            continue
        if tArr[0] not in dic:
            dic[tArr[0]] = tArr[1]
    return dic

def load_dict_from_model_config(models):
    import re
    re.sub(' +', ' ', models)
    model = models.split(" ")[0]
    options = []
    try:
        with open('%s.json' % model, 'rb') as f:
            options.append(json.load(f))
    except:
        with open('%s.pkl' % model, 'rb') as f:
            options.append(pkl.load(f))
    dictionaries = options[0]['dictionaries']
    dictionaries_source = dictionaries[:-1]
    #dictionary_target = dictionaries[-1] 
    word_dict = load_dict(dictionaries_source[0])
    if options[0]['n_words_src']:
        for key, idx in word_dict.items():
            if idx >= options[0]['n_words_src']:
                del word_dict[key]
    del word_dict['<EOS>']
    del word_dict['<UNK>']
    return word_dict

def filter_source_by_dic(source_words, dic):
    filter = []
    for it in source_words:
        if it.encode('utf-8') in dic:
            filter.append(it)
        else:
            filter.append("<UNK>")
    return filter

def copy_unknown_words(filename, out_filename, unk_token, phrase_table, output_trans, model):
        source_dic = load_dict_from_model_config(args.model)
        dic = load_phrase_table(phrase_table)
        file = open("%s.source_sent.debug" % output_trans.name ,"w")
        number = 0
	for line in filename:
                number += 1
		sent_pair = json.loads(line)
# 		print "Translation:"
# 		print sent_pair
		source_sent = sent_pair["source_sent"]
		target_sent = sent_pair["target_sent"]
		# matrix dimension: (len(target_sent) + 1) * (len(source_sent) + 1)
		# sum of values in a row = 1
		full_alignment = sent_pair["matrix"]
		source_words = source_sent.split()
		target_words = target_sent.split()

                # output source filter by source dict
                f_source_words = filter_source_by_dic(source_words, source_dic)
                f_source_sent = " ".join(f_source_words[:-1])
		# get the indices of maximum values in each row 
		# (best alignment for each target word)
		hard_alignment = numpy.argmax(full_alignment, axis=1)
# 		print hard_alignment
		
		updated_target_words = []
                align_pairs = [] # save debug info
		for j in xrange(len(target_words)):
                        #print "%s\t%s-->\t%s" % (j, hard_alignment[j], full_alignment[j][hard_alignment[j]])
			if target_words[j] == unk_token:
				unk_source = source_words[hard_alignment[j]]
                                f_unk_source = f_source_words[hard_alignment[j]]
                                if unk_source.encode('utf-8') in dic: # find correspond translation in external data[e.g. phrase table in smt]
				    #updated_target_words.append(unk_source)
                                    replace_word = dic.get(unk_source.encode('utf-8'))
				    updated_target_words.append(replace_word)
                                    pair = (unk_source, f_unk_source, target_words[j], replace_word)
                                else:
                                    if unk_source != "<EOS>": # deal conditions: target word aligns to <EOS>, not print 
                                        updated_target_words.append(unk_source)
                                    pair = ( unk_source, "%s_not_in_source_dic" % f_unk_source, target_words[j],"%s_not_exist_in_phrase_table" % target_words[j] )
                                align_pairs.append(pair)
			else:
				updated_target_words.append(target_words[j])
                                #pair = (unk_source, f_unk_source, target_words[j], target_words[j])
                                
                        #align_pairs.append(pair)
                print >>output_trans," ".join(updated_target_words).encode('utf-8')
		sent_pair["target_sent"] = " ".join(updated_target_words)
# 		print "Updated translation:"
# 		print sent_pair
		sent_pair = json.dumps(sent_pair).decode('unicode-escape').encode('utf-8')
		print >>out_filename, sent_pair
                file.write("%s*********************************************************************\n" % str(number))
                if len(align_pairs) > 0:
                    file.write("src:[%s]\n" % " ".join(source_words[:-1]).encode('utf-8'))
                    file.write("nmt:[%s]\n" % f_source_sent.encode('utf-8'))
                    file.write("tgt:[%s]\n" % " ".join(target_words).encode('utf-8'))
                    file.write("rep:[%s]\n" % " ".join(updated_target_words).encode('utf-8'))
                    for pair in align_pairs:
                        file.write("source:[%s]\tnmt:[%s]\t=>\ttarget:[%s]\tafter replacement:[%s]\n" % (pair[0], pair[1], pair[2], pair[3] ))
        file.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', type=argparse.FileType('r'),
						metavar='PATH', required=True,
						help='''Input text file in json format including alignment matrix, 
								source sentences, target sentences''')
	parser.add_argument('--output', '-o', type=argparse.FileType('w'),
						default=sys.stdout, metavar='PATH',
						help="Output file (default: standard output)")
	parser.add_argument('--output_trans', '-otrans', type=argparse.FileType('w'),
						default=sys.stdout, metavar='PATH',
						help="Output file (default: standard output)")
	parser.add_argument('--unknown', '-u', type=str, nargs = '?', default="<UNK>",
						help='Unknown token to be replaced (default: "<UNK>")')

        parser.add_argument('--phrasetable', '-p', type=argparse.FileType('r'), metavar='PATH', required=True, help='''filename of phrase table ''')
        parser.add_argument('--model', '-m', type=str,nargs='?', default="model.npz", help='model file')
	args = parser.parse_args()
		
	copy_unknown_words(args.input, args.output, args.unknown, args.phrasetable, args.output_trans, args.model)
		
	

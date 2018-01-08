import sys
from util import load_dict

VOCAB_SIZE = 10000

if 2 > len(sys.argv):
    print "Usage:\n      python statisticCoverPercent.py dictfile  filename"
    sys.exit(1)



dictionaries = sys.argv[1]
filename = sys.argv[2]


worddicts = load_dict(dictionaries)
print "In all %s words in training sentences" %  len(worddicts)

dict1 = {}
for k , idx in worddicts.items():
    if idx < VOCAB_SIZE:
        dict1.update({k: idx})

print "*****************************"
#res = sorted(dict1.items(), key=lambda d: d[1])
#################################################

num = 0
countUNK = 0
all_word_countUNK = 0
all_word_count = 0
file = open(filename)
for line in file:
    num += 1
    flag = True
    words_in = line.strip().split(' ')
    all_word_count += len(words_in)
    for w in words_in:
        if w not in dict1:
            flag = False
            all_word_countUNK += 1
    if not flag:
        countUNK += 1
file.close()

print "%s \tsentences,\t %s\t sentences contain UNK\tpercent: %s" % (num, countUNK, countUNK *1.0 / num)
print "%s \ttokens,\t%s \ttokens is UNK\tpercent: %s" %  (all_word_count, all_word_countUNK, all_word_countUNK *1.0 / all_word_count)

#!/bin/sh

#usage: prefix_of_model | source_of_dev | niutrans_form_dev | ref_num

# theano device
device=gpu3
# core path
core_path=../core/nematus/

#model prefix
prefix=$1
#prefix=../data/iwslt/debug/model

# source sentences
dev=$2
# conventional dev-format in NiuTrans, both source sentence and multi-reference
ref=$3
# reference count
ref_num=$4

echo "prefix:$prefix  dev:$dev  ref:$ref  ref_num:$ref_num"

# beam size
beam_size=12


# model
model=${prefix}.npz
# translation result of dev
dev_trans=${dev}.trans.`date +%Y-%m-%d-%H:%M:%S`
# best bleu
best_bleu_path=${prefix}_bleu_best
# bleu history
bleu_history_path=${prefix}_bleu_history


# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python ${core_path}translate.py \
     -m $model \
     -i $dev \
     -o $dev_trans \
     -k $beam_size -n -p 2 -v


# create best_bleu file if not exist
if [ ! -f ${best_bleu_path} ]; then
touch ${best_bleu_path}
fi

# if not get best_bleu, set 'best_bleu'=0
BEST=`cat $best_bleu_path`
if [ -z "$BEST" ]; then
BEST=0
fi
 
# remove subword sysbol '@@'
sed -i 's/@@ //g' $dev_trans

# get best bleu
#BEST=`cat ${best_bleu_path} || echo 0` 
echo  "perl NiuTrans-generate-xml-for-mteval.pl -1f $dev_trans -tf $ref -rnum ${ref_num}" 
order=`perl NiuTrans-generate-xml-for-mteval.pl -1f $dev_trans -tf $ref -rnum ${ref_num}` 
echo "perl  mteval-v13a.pl  -r ref.xml -s src.xml -t tst.xml |grep 'NIST score'|cut -f 9 -d ' '" 
BLEU=`perl  mteval-v13a.pl  -r ref.xml -s src.xml -t tst.xml| grep 'NIST score'|cut -f 9 -d " "` 
`echo  $BLEU >> ${bleu_history_path}`
`rm ref.xml src.xml tst.xml $dev_trans $dev_trans.temp`
echo "current_bleu:$BLEU best_bleu:$BEST"
# save model with highest BLEU 
if [ $(echo "$BLEU > $BEST"|bc) -eq 1 ]; then 
  echo "current BLEU=$BLEU BEST=$BEST" 
  echo "new best; saving" 
  echo $BLEU > ${best_bleu_path} 
  cp ${model} ${model}.best_bleu 
fi 



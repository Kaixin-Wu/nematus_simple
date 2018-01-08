#!/bin/sh

# theano device
device=gpu0

# core path
core_path=../core/nematus/

#phrase table for replace UNK in target words 
phrase_table=./phrase.translation.table

if [ $# = 0 ] ; then 
    ./translate.sh -h
    exit 1; 
fi 

model=""
file=""
flag=""
devset=""
ref_number=""
outfile=""
beam_size=12
while getopts :a:m:t:r:d:n:o:k: opt  
do  
 case $opt in  
 a) echo ;;  
 m) model=$OPTARG ;; 
 t) file=$OPTARG ;;
 r) flag=$OPTARG;;
 d) devset=$OPTARG;;
 n) ref_number=$OPTARG;;
 o) outfile=$OPTARG;;
 k) beam_size=$OPTARG;;
 
 *) echo "Usage "
    echo "***********************************************************************************************************"
    echo "*  ./translate.sh -m model -t testfile -r 1 -o output [1 means replace UNK] -k 12 [default value is 12] [ -d devset -n refnum ]       *"
    echo "*  OR                                                                                                     *"
    echo "*  ./translate.sh -m \"model1 model2 ...\" -t testfile -r 1 -o output[ -d devset -n refnum ]              *"
    echo "***********************************************************************************************************"
    echo "For example                                                                                               *"
    echo "Only translate(don'ot replace UNK): sh translate.sh -m model -t c.test -r 0 -o output                     *" 
    echo "Only translate(replace UNK):        sh translate.sh -m model -t c.test -r 1 -o output                     *"
    echo "Cacluate BLEU:                      sh translate.sh -m model -t c.test -r 1 -o output -d devset -n refnum *"
    echo "***********************************************************************************************************"
    exit;;
 esac  
done
  
echo "parametres:"
echo $model
echo $file
echo $flag
echo $devset
echo $ref_number
echo $outfile
# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python ${core_path}translate.py \
     -m $model \
     -i $file \
     -o $outfile \
     -k $beam_size -n -p 5  -a $outfile.tmp -ma $outfile.alignment -v #-pi
echo "translate fininshed"

# remove subword symbol '@@'
sed -i 's/@@ //g' $outfile

if [ "1" = "$flag" ];then
    echo "replacing UNK"
    echo "python ${core_path}replace_target_unknown_words.py  -i $outfile.alignment -o $outfile.res -otrans $outfile.trans.del.unk  --phrasetable $phrase_table -m \"$model\""
    python ${core_path}replace_target_unknown_words.py  -i $outfile.alignment -o $outfile.res -otrans $outfile.trans.del.unk  --phrasetable $phrase_table -m "$model"
    rm $outfile.tmp $outfile.alignment $outfile.res
    echo "replace UNK done"
else
    echo "not replace"
    rm $outfile.tmp $outfile.alignment
fi

if [ "" != "$devset"  -a  "" != "$ref_number" ];then
    echo "Caculating BLEU"
    if [ "1" = "$flag" ];then
        echo "perl NiuTrans-generate-xml-for-mteval.pl -1f $outfile.trans.del.unk -tf $devset -rnum $ref_number"
        perl NiuTrans-generate-xml-for-mteval.pl -1f $outfile.trans.del.unk -tf $devset -rnum $ref_number
    else
        echo "perl NiuTrans-generate-xml-for-mteval.pl -1f $outfile.trans -tf $devset -rnum $ref_number"
        perl NiuTrans-generate-xml-for-mteval.pl -1f $outfile -tf $devset -rnum $ref_number
    fi
    BLEU=`perl  mteval-v13a.pl  -r ref.xml -s src.xml -t tst.xml| grep 'NIST score'|cut -f 9 -d " "`
    echo "testfile: $file, BLEU=$BLEU"
    rm ref.xml src.xml tst.xml
    if [ "1" = "$flag" ];then
        rm $outfile.trans.del.unk.temp
    else
       rm $outfile.temp
    fi
fi

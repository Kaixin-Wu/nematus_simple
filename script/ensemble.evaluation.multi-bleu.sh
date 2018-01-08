set -e
DIR=../model/

MODEL=("model.iter200000.npz" "model.iter210000.npz" "model.iter220000.npz" "model.iter230000.npz" "model.iter240000.npz" "model.iter250000.npz")

MODEL_STR=""
for m in ${MODEL[@]}
do
	MODEL_STR="${MODEL_STR}${DIR}${m} "
done
#MODEL_STR=${MODEL_STR}"\""
#echo $MODEL_STR

#MODEL=ensemble4.avg.npz
test1=mt04
test2=mt05
test3=mt08

eval "./translate.sh -m "\""$MODEL_STR"\"" -t ../data/valid/c.utf8.token.nogen -r 0 -o ../data/valid/trans.out" 
perl multi-bleu.perl ../data/valid/e.ref < ../data/valid/trans.out

eval "./translate.sh -m \"$MODEL_STR\" -t ../data/test/$test1/c.utf8.token.nogen -r 0 -o ../data/test/$test1/trans.out"
perl multi-bleu.perl ../data/test/$test1/e.ref < ../data/test/$test1/trans.out

eval "./translate.sh -m \"$MODEL_STR\"  -t ../data/test/$test2/c.utf8.token.nogen -r 0 -o ../data/test/$test2/trans.out"
perl multi-bleu.perl ../data/test/$test2/e.ref < ../data/test/$test2/trans.out

eval "./translate.sh -m  \"$MODEL_STR\" -t ../data/test/$test3/c.utf8.token.nogen -r 0 -o ../data/test/$test3/trans.out"
perl multi-bleu.perl ../data/test/$test3/e.ref < ../data/test/$test3/trans.out

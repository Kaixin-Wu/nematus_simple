model=../data/c2e.180W.gen/model/finetune/model.npz.best_bleu

input=../data/c2e.180W.gen/test/mt08/c.utf8.token

beam_size=(1 4 8 12 16 20)

replace=1

ref=../data/c2e.180W.gen/test/mt08/test.txt

refnum=4

len=${#beam_size[*]}
for k in $( seq 0 $(expr $len - 1))
do
	k=${beam_size[$k]}
	output=../data/c2e.180W.gen/test/mt08/c.utf8.token.trans.k$k
	echo "./translate.sh -m $model -t $input -o $output -r $replace -d $ref -n $refnum -k $k"
	./translate.sh -m $model -t $input -o $output -r $replace -d $ref -n $refnum -k $k
done

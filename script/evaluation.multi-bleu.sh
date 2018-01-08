set -e

#MODEL=model.npz.best_bleu
MODEL=ensemble6.avg.npz

unk_rep=0

model_dir=../model
data_dir=../data

test1=mt04
test2=mt05
test3=mt08

model_path=$model_dir/$MODEL
test_dir=$data_dir/test
valid_dir=$data_dir/valid

./translate.sh -m $model_path -t $valid_dir/c.utf8.token.nogen -r $unk_rep -o $valid_dir/trans.out
# multi-references file name should be <ref-name>0, <ref-name>1 ...
perl multi-bleu.perl $valid_dir/e.ref < $valid_dir/trans.out

./translate.sh -m $model_path -t $test_dir/$test1/c.utf8.token.nogen -r $unk_rep -o $test_dir/$test1/trans.out
perl multi-bleu.perl $test_dir/$test1/e.ref < $test_dir/$test1/trans.out

./translate.sh -m $model_path -t $test_dir/$test2/c.utf8.token.nogen -r $unk_rep -o $test_dir/$test2/trans.out
perl multi-bleu.perl $test_dir/$test2/e.ref < $test_dir/$test2/trans.out

./translate.sh -m $model_path -t $test_dir/$test3/c.utf8.token.nogen -r $unk_rep -o $test_dir/$test3/trans.out
perl multi-bleu.perl $test_dir/$test3/e.ref < $test_dir/$test3/trans.out

#MODEL=model.npz
#MODEL=model.npz.best_bleu
MODEL=ensemble8.avg.npz
test1=mt04
test2=mt05
test3=mt08
./translate.sh -m ../model/$MODEL -t c.token.nogen200 -r 0 -o c.token.nogen200.out -d ../data/test/$test1/test.txt -n 5
#./translate.sh -m ../model/$MODEL -t ../data/test/$test2/$test2.tok.bpe.30000.de -r 0 -o ../data/test/$test2/trans.out -d ../data/test/$test2/test.txt -n 1
#./translate.sh -m ../model/$MODEL -t ../data/test/$test3/$test3.tok.bpe.30000.de -r 0 -o ../data/test/$test3/trans.out -d ../data/test/$test3/test.txt -n 1

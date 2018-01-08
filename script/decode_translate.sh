#!/bin/sh

core_path=../core/nematus/
nohup python ${core_path}translate.py \
	-m ../model/ensemble8.avg.npz \
        -i ../data/valid/c.utf8.token.nogen \
        -o ./output.txt \
        -k 12 -n -p 3 \
        > ./log.decode &  

#! /bin/sh

if [ ! -e data/$1 ]; then
	mkdir data/$1
fi

touch data/$1/ans.txt

bin/detect test/img_ml/$1.jpeg 1> data/$1/ref.txt 2> data/$1/data.txt

if [ $?  = 0 ]; then
	echo "done"
else
	echo "oh... something wrong"
fi

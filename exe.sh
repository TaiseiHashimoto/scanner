#! /bin/sh

if [ ! -e data/$1 ]; then
	mkdir data/$1
fi

touch data/$1/ans.txt

bin/detect data/$1

if [ $?  = 0 ]; then
	echo "done"
else
	echo "oh... something wrong"
fi

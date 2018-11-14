#!/bin/bash

bigimgs=`find $1`
regexp="(.*)_big.jpeg"

for bigimg in $bigimgs; do
	if [[ $bigimg =~ $regexp ]]; then
		basename=${BASH_REMATCH[1]}
		# echo $basename
		convert -resize 600x -unsharp 0.125x1.0+1+0.05 -quality 90 -verbose $bigimg $basename.jpeg
	fi
done
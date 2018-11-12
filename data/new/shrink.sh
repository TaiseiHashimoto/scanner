#!/bin/bash

bigimgs=`find $1`
regexp="(.*)_big.jpeg"

for bigimg in $bigimgs; do
	if [[ $bigimg =~ $regexp ]]; then
		basename=${BASH_REMATCH[1]}
		# echo $basename
		convert -resize 600x -unsharp 2x1.4+0.5+0 -colors 65 -quality 100 -verbose $bigimg $basename.jpeg
	fi
done
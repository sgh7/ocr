#!/bin/bash

picklefile=IMG_20150924_205053.pickle
#trainingfile=p1.train
#simfile=sim.out
#cl_img_file=single03.png
trainingfile=p1b.train
simfile=simb.out
cl_img_file=single03b.png

if [ "$1" = "f" ] || [ "$1" = "cst" ] || [ "$1" = "cinv" ] || [ "$1" = "ci" ] ; then
	tests=$1
else
	tests="f cst cinv ci"
fi


# do feature selection
if echo $tests | grep -q f ; then
	./feature_sel.py -c B -o $picklefile IMG_20150924_205053.jpg
fi

# output similarity matrix and training data
if echo $tests | grep -q cst ; then
	./cluster.py -S $simfile -r 0.3 -T $trainingfile $picklefile
fi

# invalid flag combo should be caught
if echo $tests | grep -q cinv ; then
	if ./cluster.py -s $simfile -r 0.3 -t $trainingfile -w $cl_img_file $picklefile ; then
		echo "invalid flag combo was not caught!"
	fi
fi

# write out image file showing all glyphs clustered and training glyphs marked
if echo $tests | grep -q ci ; then
	./cluster.py -r 0.3 -t $trainingfile -w $cl_img_file $picklefile
fi

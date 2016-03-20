#!/bin/bash

imgfile=IMG_20150924_205053.jpg
picklefile=IMG_20150924_205053.pickle
#trainingfile=p1.train
#simfile=sim.out
#cl_img_file=single03.png
trainingfile=p1b.train
simfile=simb.out
cl_img_file=single03b.png
verbose=

if [ "$1" = "-v" ] ; then
	verbose=-v
	shift
fi

if [ "$1" = "f" ] || [ "$1" = "cst" ] || [ "$1" = "cinv" ] || [ "$1" = "ci" ] || [ "$1" = "class" ] ; then
	tests=$1
	shift
else
	tests="f cst cinv ci class"
fi


# do feature selection
if echo $tests | grep -q f ; then
	./feature_sel.py $verbose -c B -g 15.0 -o $picklefile $imgfile
fi

# output similarity matrix and training data
if echo $tests | grep -q cst ; then
	./cluster.py $verbose -S $simfile -r 0.3 -T $trainingfile $picklefile
fi

# invalid flag combo should be caught
if echo $tests | grep -q cinv ; then
	if ./cluster.py -s $simfile -r 0.3 -t $trainingfile -w $cl_img_file $picklefile ; then
		echo "invalid flag combo was not caught!"
	fi
fi

# write out image file showing all glyphs clustered and training glyphs marked
if echo $tests | grep -q ci ; then
	./cluster.py $verbose -r 0.3 -t $trainingfile -w $cl_img_file $picklefile
fi

# do final classification
if echo $tests | grep -q class ; then
	./classify.py $verbose -f $picklefile -t $trainingfile $imgfile
fi

#!/bin/bash

picklefile=IMG_20150924_205053.pickle
#trainingfile=p1.train
#simfile=sim.out
#cl_img_file=single03.png
trainingfile=p1b.train
simfile=simb.out
cl_img_file=single03b.png

# do feature selection
./feature_sel.py -c B -o $picklefile IMG_20150924_205053.jpg

# output similarity matrix and training data
./cluster.py -S $simfile -r 0.3 -T $trainingfile $picklefile

# invalid flag combo should be caught
if ./cluster.py -s $simfile -r 0.3 -t $trainingfile -w $cl_img_file $picklefile ; then
	echo "invalid flag combo was not caught!"
fi

# write out image file showing all glyphs clustered and training glyphs marked
./cluster.py -r 0.3 -t $trainingfile -w $cl_img_file $picklefile

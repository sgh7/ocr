#!/bin/bash

tests="f cst class"
echo=
verbose=
previoustrainfiles=


while [ "${1:0:1}" == "-" ] ; do
	opt=$1
	if [ "$opt" == "-f" ] || [ "$1" = "-cst" ] || [ "$1" = "-ci" ] || [ "$1" = "-class" ] ; then
		tests=$opt
	elif [ "$opt" == "-v" ] ; then
		verbose=-v
	elif [ "$opt" == "-n" ] ; then
		echo=echo
	elif [ "$opt" == "-t" ] ; then
		shift
		previoustrainfiles="$previoustrainfiles -t $1"
	else
		echo "unknown option $1"
		exit 4
	fi
	shift
done

if [ "x$1" = "x" ] ; then
	echo "please specify an image file"
	exit 1
fi

imgfile=$1
suffix=`echo $imgfile | sed -e 's/^.*\(\.[^.]*\)$/\1/'`
featurefile=`basename $imgfile $suffix`.feat
trainingfile=`basename $imgfile $suffix`.train
simfile=`basename $imgfile $suffix`.sim

#trainingfile=p1b.train
cl_img_file=single03b.png


# do feature selection
if echo $tests | grep -q f ; then
	$echo ./feature_sel.py $verbose -c B -g 15.0 -o $featurefile $imgfile
fi

# output similarity matrix and training data
if echo $tests | grep -q cst ; then
	$echo ./cluster.py $verbose -S $simfile -r 0.3 $previoustrainfiles -T $trainingfile $featurefile
fi

# write out image file showing all glyphs clustered and training glyphs marked
if echo $tests | grep -q ci ; then
	$echo ./cluster.py $verbose -r 0.3 -t $trainingfile -w $cl_img_file $featurefile
fi

# do final classification
if echo $tests | grep -q class ; then
	$echo ./classify.py $verbose -f $featurefile -t $trainingfile $imgfile
fi

=======================================
Optical Character Recognition in Python
=======================================

:Version: 0.0.1
:Source: https://github.com/sgh7/ocr/
:Keywords: command line, OCR, python

--

Optical character engine
========================
This project was a challenge to myself to
write a functioning OCR program using
contemporary technology.  It is a testament
to the programmer productivity afforded
by the python language and available
libraries and documentation that it took
only eight person-days of concerted effort
to achieve this.  Decades ago, the language
of choice might have been C. There was no
stack exchange or even google, let alone
the world-wide web, just photocopy machines,
microfiche readers, and card-catalogs for
research, magnetic tapes or dial-up modems
for file transfer, and most computing
chassis weighed as much or more than an
adult human.  I might estimate that
duplicating the functionality of that 8 day 
effort a few decades ago would have taken
the order of a year, even if all of the
algorithms were known.

It was also a beginner project in aspects
of machine learning and various python
libraries.


Intended Use Case
=================

A cellphone camera is used to capture images
of printed data.  The original document was
a numerical table printed in a journal.  The
set of characters of interest is small (i.e.
the digits 0-9 and numerical signs and punctuation).
The image is warped due to page curvature and
suffers from moderate blurring.

It is strongly desirable that some indication
is given to the user of the certainty of
classification of each character glyph, so that
the error rate of the entire process (imaging,
training the classifier, and post-editing the
output) can be made arbitrarily small, given 
an image that is not too blurry or warped.  The
amount of required post-editing should be
minimized.

The possible categories to be predicted (classes)
can be arbitrary strings if it desirable to
recognize characters in different fonts.



Processing Stages
=================

- **feature_sel.py** extracts features (actually
  instances - character glyphs) from an image.

- **cluster.py** performs hierarchical clustering
  of the character glyphs and interactively prompts
  the user for training data for the specific fonts
  inherent in the image.

- **classify.py** classifies the remaining character
  glyphs, estimates character spacing and layout,
  performs left-to-right text flowing and output.
  (actually the classifier currently just uses the
  results of the earlier clustering phase)





Road Map
========

- Satisfactory deblurring the image is desirable.
  Some form of Bayesian analysis could be used to
  estimate a usable Point Spread Function for use
  as a deconvolution kernel.  Some attempt has
  been made using the **pymc** probabilistic
  programming suite.  The model specification
  needs help.  **STAN** might be preferable to **pymc**.
  
- Review use of terminology throughout the project.   

- Provide GUI for entering training data.

- Implement some sort of neural network with saved
  training set for common generic feature sets as an
  alternative to manual training for specific fonts

- Explore what, if anything, this project might
  provide to other projects such as ocropus/kraken.

- Process a montage of images of the same source
  page, handling overlaps, different warps, skews
  and blurrings.  Maybe Bayesian modelling of it
  all.

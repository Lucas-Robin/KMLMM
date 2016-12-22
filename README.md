# KMLMM

------------
What we have
------------

scripts to load the datasets ZIP and CIFAR10
kernel-multiclass SVM on both datasets
neural networks for both
interactive digit-drawing and classification



----
TODO
----

use MNIST (replaces ZIP, or we use both)
parameter optimization
write report



--------------------
Ideas for extensions
--------------------

implement our own multiclass-svm-strategies
. e.g. using two-class svm in a tree-structure:
.. first build a model that distinguishes between classes "round digits" (0, 3, 6, 8, 9) vs (1, 2, 4, 5, 7)
.. then split these subsets etc.
.. in the end have svms that are specialised on the most difficult problems, for example distinguishes between 8 and 9 which look similar
.. -> in CIFAR10, distinguishing between dogs and cats might be hard
. or build one regression svm for each class (vs. the 9 others)
.. that will give us a confidence between 0 and 1 for every class

image feature extraction
. e.g. interactions between pixels and their neighbors in order to detect edges (might improve svm results but probably not neural networks)

transpose images
-> evaluate if the accuracy is different

data augmentation (transpose, zoom, change contrast, change color,...)

downsampling: average or median of 4 pixels

grayscale the coloured images

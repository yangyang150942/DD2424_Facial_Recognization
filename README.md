# DD2424_Facial_Recognization

In this project, the theory of our network is mainly based on FaceNet, which learns a Euclidean 
embedding per image using a deep convolutional network. This network directly trains the output
to be a compact 128-D embedding using a triplet-based loss function based on LMNN. The
triplets we use consist of two matching face thumbnails and non-matching face thumbnails. The
thumbnails are the right crops of the face area, no 2D or 3D alignment, other than scale and translation
is performed. We train this network to make the squared L2 distances in the embedding space
directly correspond to face similarity: the same person faces have small distances and the different
person faces have large distances.

# How to build
1. put this git in ./tensorflow/contrib folder
2. cd to tensorflow root patch
3. $ git am ./tensorflow/contrib/mkldnn_rnn/0001-tensorflow-patch-to-enable-mkldnn_rnn.patch -3
4. $ ./configure, pls enable mkl while configuration
5. $ cp ./tensorflow/contrib/mkldnn_rnn/build.sh ./
6. $ cp ./tensorflow/contrib/mkldnn_rn`n/mkl-dnn/lib/libmkldnn.so ./third_party/mkl/
7. $ . ./build.sh

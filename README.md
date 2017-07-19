# How to build
1. put this git in ./tensorflow/contrib folder
2. cd to tensorflow root patch
3. $ git am -3 ./tensorflow/contrib/mkldnn_rnn/0001-tensorflow-patch-to-enable-mkldnn_rnn.patch
4. $ cp ./tensorflow/contrib/mkldnn_rnn/build.sh ./
5. $ cp ./tensorflow/contrib/mkldnn_rnn/mkl-dnn/lib/libmkldnn.so ./third_party/mkl/
5. $ . ./build.sh

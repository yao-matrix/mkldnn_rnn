# How to build
```shell
1. put this git in ./tensorflow/contrib folder
2. cd <tensorflow-root-path>
3. $ git am ./tensorflow/contrib/mkldnn_rnn/0001-enable-mkldnn_rnn-in-tensorflow-codebase.patch -3
4. $ cp ./tensorflow/contrib/mkldnn_rnn/build.sh ./
5. $ ./configure, pls enable mkl while configuration
6. $ . ./build.sh
```

# How to run
1. functionality test
```shell
  $ python ./python/kernel_tests/mkldnn_rnn_ops_test.py
```
2. benchmark test
```shell
  $ python ./python/kernel_tests/mkldnn_rnn_ops_benchmark.py
```

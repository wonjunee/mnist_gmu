#!/bin/bash

# python 20190315-MNIST.py 100 30 20190316-without-batch
# python 20190315-MNIST.py 200 30 20190316-without-batch
# python 20190315-MNIST.py 400 30 20190316-without-batch
# python 20190315-MNIST.py 800 30 20190316-without-batch
# python 20190315-MNIST.py 1600 30 20190316-without-batch
# python 20190315-MNIST.py 3200 30 20190316-without-batch

echo python 20190315-MNIST-batch.py 100 30 20190316-with-batch
python 20190315-MNIST-batch.py 100 30 20190316-with-batch
python 20190315-MNIST-batch.py 200 30 20190316-with-batch
python 20190315-MNIST-batch.py 400 30 20190316-with-batch
python 20190315-MNIST-batch.py 800 30 20190316-with-batch
python 20190315-MNIST-batch.py 1600 30 20190316-with-batch
python 20190315-MNIST-batch.py 3200 30 20190316-with-batch
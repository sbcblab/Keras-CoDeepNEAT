# Keras-CoDeepNEAT
[CoDeepNEAT](https://arxiv.org/abs/1703.00548) inspired implementation using Keras and Tensorflow as backend.

Experiment discussion and description: [publication](https://www.lume.ufrgs.br/)

## General instructions

Download the repository and import the ``base/kerascodeepneat.py`` file into your Python Script.
This will give you access to the Population and Dataset classes, which are the only necessary classes to run the entire process.

Configuration parameters must be set, as in the examples ``run_cifar10.py`` and ``run_mnist.py`` in [Example Scripts](https://github.com/sbcblab/Keras-CoDeepNEAT/tree/master/example_scripts)

## Outputs

The framework generates a series of files logging the evolution of populations (into .log, .csv and .json files), including informations related to:
- Model assembling, evaluation, speciation, reproduction and generation summary details (test.log)
- Scores, features and species for individuals over generations (iterations.csv)
- Training history for final model (training.csv and training.json)
- Images representing modules, blueprints, assembled graphs, keras models for every individual (\images directory in .png format)
- Keras models related to the best models for every generation (\models directory in .h5 format).

## Example Scripts

- ``run_mnist.py`` describes a sample run using the MNIST dataset.


- ``run_cifar10.py`` describes a sample run using the CIFAR-10 dataset.

## Requirements
- Keras 2.2.5
- Tensorflow 1.13.1.
- Networkx 2.3.
- PyDot 1.4.1
- GraphViz 0.11.1
- SkLearn 0.21.3

Compartibility with other version has not been tested.

## Dev infos
Code developed and tested by [Jonas Bohrer](https://github.com/jonasbohrer) (jsbohrer@inf.ufrgs.br)

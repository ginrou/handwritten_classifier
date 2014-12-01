# Handwritten Classifier
Handwrittern digit classification commandline tool and simple web app.

this sample is hosted on [http://hc.tkd.uni.me](http://hc.tkd.uni.me)

## Abstract
#### NeuralNetwork Classifier
* `NeuralNetwork.py` : Implementation of 3-layer neural network.
* `handwritten_classifier.py` : command line tool to train and test NeuralNetwork.py

Since `handwritten_classifier.py` depends on [MNIST handwritten digits database](http://yann.lecun.com/exdb/mnist/), `TrainData.py` supports to use MNIST datasets.

#### Flask Application
Simple web application based on flask to use this classifier. User writes on HTML canvas, drawn handwritten data is send with ajax requests, and get classification result.

* `app.py` : Description of flask application.
* `static/drawer.js` : Canvas drawer and post ajax request.

#### Docker support
The most simple way to try this app is to use from Docker. See [https://registry.hub.docker.com/u/ginrou/handwritten-classifier/](https://registry.hub.docker.com/u/ginrou/handwritten-classifier/) for container.

## Requirements
* Python3.x
* Python libraries installed with pip
  * flask
  * numpy

## How to use
### handwritten_classifier.py

```
### check python version
$ python --version
Python 3.3.0

### download repo
$ git clone https://github.com/ginrou/handwritten_classifier.git
$ cd handwritten_classifier

### install numpy and flask
$ pip install numpy flask

### download MNIST dataset
$ wget http://deeplearning.net/data/mnist/mnist.pkl.gz

### train with MNIST dataset
### traind parameters are saved in mat.npz at this sample
$ python handwritten_classifier.py train --data-set mnist.pkl.gz --nn mat.npz

### evaluate
$ python handwritten_classifier.py evaluate --data-set mnist.pkl.gz --nn mat.npz

```

NOTE:
* Dataset is good to use pickled dataset from pickled data set ready at [deeplearning.net](http://deeplearning.net/tutorial/gettingstarted.html)
* With this dataset, precision is 92.95%
* Trained parameters data is included in this repo (mat.npz)

### Web App
Assuming repo is downloaded, libraries are installed, and trained parameters are ready.
File name of trained parameters is hard-coded as `mat.npz`

```
### run app
$ python app.py
```

Then, access to `http://localhost:5000` from your browser.

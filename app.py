#!/usr/bin/env python

from flask import Flask, render_template, request, jsonify
from NeuralNetwork import *
import numpy

import pprint
pp = pprint.PrettyPrinter(indent=4)

app = Flask(__name__)
nn = NeuralNetwork(in_size = 784, hidden_size = 300, out_size = 10)
nn.load("mat.npz");

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/estimate", methods = ["POST"])
def estimate():
    try:
        x = numpy.array(request.json["input"]) / 255.0
        y = int(nn.predicate(x))
        return jsonify({"estimated":y})
    except Exception as e:
        print(e)
        return jsonify({"error":e})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

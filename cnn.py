# -*- coding: utf-8 -*-

"""
Version: 2.7.15
Author: Ünver Can Ünlü
"""

from __future__ import print_function
import os
import numpy
import pickle
import lasagne
import theano

########## DATASET ##########

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3
IMAGE_SHAPE = (IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH)

########## PARAMETERS ##########

EPOCHS = 10
BATCH_SIZE = 500
LEARNING_RATE = 1e-03
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
LEAKNESS = 0.01
DROPOUT = 0.25
CONSTANT_VALUE = 0

########## METHODS ##########

NORMAL = lasagne.init.Normal()
UNIFORM = lasagne.init.Uniform()
HE_NORMAL = lasagne.init.HeNormal(gain='relu')
HE_UNIFORM = lasagne.init.HeUniform(gain='relu')
CONSTANT = lasagne.init.Constant(val=CONSTANT_VALUE)
LEAKY_RECTIFY = lasagne.nonlinearities.LeakyRectify(leakiness=LEAKNESS)
RELU = lasagne.nonlinearities.rectify
SIGMOID = lasagne.nonlinearities.sigmoid
TANH = lasagne.nonlinearities.tanh
SOFTMAX = lasagne.nonlinearities.softmax
ADAM = lasagne.updates.adam
CROSS_ENTROPY = lasagne.objectives.categorical_crossentropy

########## MODEL ##########

def create_network(variable=None, droput=DROPOUT, activation=LEAKY_RECTIFY, classifier=SOFTMAX, weight=HE_NORMAL, bias=CONSTANT, image_shape=IMAGE_SHAPE):
    channel, height, width = image_shape
    input_layer = lasagne.layers.InputLayer(shape=(None, channel, height, width), input_var=variable)
    conv1 = lasagne.layers.Conv2DLayer(incoming=input_layer, num_filters=8, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    conv2 = lasagne.layers.Conv2DLayer(incoming=conv1, num_filters=8, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    pool1 = lasagne.layers.Pool2DLayer(incoming=conv2, pool_size=(2, 2), stride=(2, 2), pad=0)
    drop1 = lasagne.layers.DropoutLayer(incoming=pool1, p=droput)
    conv3 = lasagne.layers.Conv2DLayer(incoming=drop1, num_filters=16, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    conv4 = lasagne.layers.Conv2DLayer(incoming=conv3, num_filters=16, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    pool2 = lasagne.layers.Pool2DLayer(incoming=conv4, pool_size=(2, 2), stride=(2, 2), pad=0)
    drop2 = lasagne.layers.DropoutLayer(incoming=pool2, p=droput)
    fc = lasagne.layers.DenseLayer(incoming=drop2, num_units=len(LABELS), nonlinearity=classifier, W=weight, b=bias)
    return fc

def load_network_from_model(network, model):
    with open(model, 'r') as model_file:
        parameters = pickle.load(model_file)
    lasagne.layers.set_all_param_values(layer=network, values=parameters)

def save_network_as_model(network, model):
    parent_directory = os.path.abspath(model + "/../")
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    parameters = lasagne.layers.get_all_param_values(layer=network)
    with open(model, 'w') as model_file:
        pickle.dump(parameters, model_file)

########## DATASET ##########

def preprocess(data):
    return data / numpy.float32(256)

def load_batch(batch_file):
    with open(batch_file, mode='rb') as opened_file:
        batch = pickle.load(opened_file)
        labels = batch[b'labels']
        datas = batch[b'data']
        names = batch[b'filenames']
    return names, datas, labels

def load_train_samples(dataset=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cifar10'), labels=LABELS, image_shape=IMAGE_SHAPE):
    number_of_labels = len(labels)
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    train_batch_files = [os.path.join(dataset, train_batch) for train_batch in train_batches]
    x_train = []; y_train = []
    for train_batch_file in train_batch_files:
        _, datas, labels = load_batch(train_batch_file)
        number_of_batch_samples = len(datas)
        for index in range(number_of_batch_samples):
            data = preprocess(data=numpy.reshape(datas[index], image_shape))
            label = [1 if labels[index] == j else 0 for j in range(number_of_labels)]
            x_train.append(data); y_train.append(label)
    datas = numpy.array(x_train, dtype=numpy.float32)
    labels = numpy.array(y_train, dtype=numpy.int8)
    return datas, labels

def load_test_samples(dataset_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cifar10'), labels=LABELS, image_shape=IMAGE_SHAPE):
    number_of_labels = len(labels)
    test_batch = 'test_batch'
    test_batch_file = os.path.join(dataset_path, test_batch)
    x_test = []; y_test = []
    _, datas, labels = load_batch(test_batch_file)
    number_of_samples = len(datas)
    for index in range(number_of_samples):
        data = preprocess(data=numpy.reshape(datas[index], image_shape))
        label = [1 if labels[index] == j else 0 for j in range(number_of_labels)]
        x_test.append(data); y_test.append(label)
    datas = numpy.array(x_test, dtype=numpy.float32)
    labels = numpy.array(y_test, dtype=numpy.int8)
    return datas, labels

########## TRAIN ##########

def generate_batches(datas, labels, batch_size=BATCH_SIZE):
    number_of_samples = len(datas)
    number_of_batch = number_of_samples / batch_size
    data_batches = numpy.split(datas, number_of_batch)
    label_batches = numpy.split(labels, number_of_batch)
    batches = [dict(data=data_batches[index], label=label_batches[index]) for index in range(number_of_batch)]
    return batches

def train(datas, labels, updater=ADAM, loss=CROSS_ENTROPY, epochs=EPOCHS, rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2, epsilon=EPSILON, model='model.params', model_path=os.path.dirname(os.path.realpath(__file__))):
    epoch_path = os.path.join(model_path, 'epochs')
    tensors = dict(input=theano.tensor.tensor4(dtype='float32'), output=theano.tensor.matrix(dtype='int8'))
    network = create_network(variable=tensors['input'])
    predictions = lasagne.layers.get_output(layer_or_layers=network)
    losses = loss(predictions=predictions, targets=tensors['output']).mean()
    parameters = lasagne.layers.get_all_params(layer=network, trainable=True)
    updates = updater(loss_or_grads=losses, params=parameters, learning_rate=rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    trainer = theano.function(inputs=[tensors['input'], tensors['output']], outputs=losses, updates=updates)
    batches = generate_batches(datas=datas, labels=labels)
    for epoch in range(epochs):
        print('Epoch {e}:'.format(e=(epoch+1)))
        number_of_batch = len(batches)
        for batch_index in range(number_of_batch):
            batch = batches[batch_index]
            batch_loss = trainer(batch['data'], batch['label'])
            print('Batch {b}: Loss = {l:.5f}'.format(b=(batch_index+1), l=batch_loss))
        epoch_file = 'epoch_{e}.params'.format(e=(epoch+1))
        epoch_model = os.path.join(epoch_path, epoch_file)
        save_network_as_model(network, epoch_model)
    trained_model_file = os.path.join(model_path, model)
    save_network_as_model(network, trained_model_file)

########## TEST ##########

def predict(data_or_datas, model, image_shape=IMAGE_SHAPE):
    input_tensor = theano.tensor.tensor4(dtype='float32')
    network = create_network(variable=input_tensor)
    load_network_from_model(network=network, model=model)
    prediction = lasagne.layers.get_output(layer_or_layers=network, deterministic=True)
    result = theano.tensor.argmax(prediction, axis=1)
    predictor = theano.function(inputs=[input_tensor], outputs=result)
    if data_or_datas.shape != image_shape:
        datas = data_or_datas
        predictions = predictor(datas)
        return predictions
    else:
        channel, height, width = image_shape
        data = numpy.reshape(data_or_datas, newshape=(1, channel, height, width))
        prediction = predictor(data)
        return prediction

def test(datas, labels, model=os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.params")):
    number_of_samples = len(datas)
    predictions = predict(data_or_datas=datas, model=model)
    accurancy = 0
    for index in range(number_of_samples):
        prediction = predictions[index]
        target = numpy.argmax(labels[index])
        if target == prediction:
            accurancy += 1
    accurancy = (numpy.float32(accurancy) / number_of_samples) * 100
    print('Accurancy: {a:.3f}'.format(a=accurancy))

########## MAIN ##########

def main():
    print('Train samples are loading.')
    train_datas, train_labels = load_train_samples()
    print('Train samples are loaded.')
    print('Training:')
    train(datas=train_datas, labels=train_labels)
    print('Trained:')
    print('Test samples are loading.')
    test_datas, test_labels = load_test_samples()
    print('Testing:')
    test(datas=test_datas, labels=test_labels)
    print('Tested:')

if __name__ == '__main__':
    main()

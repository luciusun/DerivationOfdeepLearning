#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'document note: '

__author__ = 'slucius'

import numpy as np
from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''

        :param input_num: set the number of input parameters
        :param activator: the acticator Function, double to double

        '''

        self.activator = activator
        # self.weights = [0.0 for _ in range(input_num)]
        self.weights = np.zeros(input_num)
        self.bias=0.0


    def __str__(self):
        '''

        :return: print the learned weights and bias;

        '''

        return 'weights\t:%s\nbias\t:%s\n' %(self.weights, self.bias)


    def predict(self, input_vec):
        '''

        :param input_vec: input the vector
        :return: output the Perceptron anwser

        '''

        # z = zip(input_vec, self.weights)
        # m = map(lambda x, w: x*w, z)
        # r = reduce(lambda a,b: a+b, m)
        # return self.activator(r+self.bias)

        # v1 = np.array(input_vec, dtype=float)
        # v2 = np.array(self.weights, dtype=float)
        return self.activator(input_vec.dot(self.weights)+ self.bias)


    def train(self, input_vecs, labels, iteration, rate):
        '''

        :param input_vecs:
        :param labels:
        :param interation:
        :param rate:
        :return:

        '''

        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)


    def _one_iteration(self, input_vecs, labels, rate):
        '''

        :param input_vecs:
        :param labels:
        :param rate:
        :return:
        '''


        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)


    def _update_weights(self, input_vec, output, label, rate):
        '''

        :param input_vec:
        :param output:
        :param label:
        :param rate:
        :return:
        '''

        delta = label - output
        z = zip(input_vec, self.weights)
        self.weights = map(lambda (x, w): w+rate*delta*x, z)
        self.bias += rate*delta



def activator_function(x):
    '''

    :param x: define the activator
    :return:

    '''
    return 1 if x>0 else 0


def get_training_dataset():
    '''

    :return:
    '''

    input_vecs = np.array([[1,1],[0,0],[1,0],[0,1]], dtype=float)
    labels = np.array([1,0,0,0], dtype=float)
    return input_vecs, labels


def train_and_perceptron():
    '''

    :return:
    '''

    p=Perceptron(2, activator_function)
    input_vecs, labels= get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)
    print '1 and 1 = %s' % and_perception.predict(np.array([1, 1]))
    print '0 and 0 = %s' % and_perception.predict(np.array([0, 0]))
    print '1 and 0 = %s' % and_perception.predict(np.array([1, 0]))
    print '0 and 1 = %s' % and_perception.predict(np.array([0, 1]))


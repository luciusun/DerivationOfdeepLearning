#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'document note: '

__author__ = 'slucius'

import numpy as np

class Perceptron(object):
    '''
    build a class named: Perceptron
    '''
    def __init__(self, input_number, activator):
        '''
        :param input_number:
        :param activation:
        '''
        # self.input_number = input_number
        self.activator = activator
        self.weights = np.zeros(input_number, dtype=float)
        self.bias = 0.0


    def __str__(self):
        '''

        :return:
        '''
        return 'weights\t:%s\nbias\t:%s' %(self.weights, self.bias)


    def predict(self, input_vec):
        '''

        :param input_vec:
        :return: output the anwser of Perceptron
        '''
        anwser =sum(input_vec*self.weights) + self.bias
        return self.activator(anwser)


    def _one_iteration(self, input_vecs, labels, rate):
        '''

        :param input_vec:
        :param label:
        :param rate:
        :return:
        '''
        # deltw=rate*(label-self.predict(input_vec))
        sample = zip(input_vecs, labels)
        for (input_vec, label) in sample:
            self._update_weights(input_vec, label, rate)

    def _update_weights(self, input_vec, label, rate):
        '''

        :param input_vecs:
        :param labels:
        :param rate:
        :return:
        '''
        delta = label-self.predict(input_vec)
        z = zip(input_vec, self.weights)
        self.weights = map(lambda (x,w): w+rate*delta*x, z)
        self.bias += rate*delta


    def train(self, input_vecs, labels, iteration, rate):
        '''

        :param input_vecs:
        :param labels:
        :param iteration:
        :param rate:
        :return:
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)



def activator_function(x):
    '''

    :param x:
    :return:
    '''
    return 1 if x>0 else 0


def get_training_examples():
    '''

    :return:
    '''
    input_vectors=np.array([[1,1],[1,0],[0,1],[0,0]], dtype=float)
    labels = np.array([1, 0, 0, 0], dtype=float)
    return input_vectors, labels


def train_and_percptron():
    p=Perceptron(2, activator_function)
    input_vecs, labels=get_training_examples()
    p.train(input_vecs, labels, 10, 0.1)
    return p



if __name__ == '__main__':
    and_perception = train_and_percptron()
    print(and_perception)

    #test
    print("1 and 1 = %s" % and_perception.predict(np.array([1, 1])))
    print('0 and 0 = %s' % and_perception.predict(np.array([0, 0])))
    print('1 and 0 = %s' % and_perception.predict(np.array([1, 0])))
    print('0 and 1 = %s' % and_perception.predict(np.array([0, 1])))






#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'document note: build a liear unit, it is suitable for Regression problem'

__author__ = 'slucius'


from perceptron_sl import Perceptron
import numpy as np
import matplotlib.pyplot as plt

#define linear activator
f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''
        :param input_num: initializate linear unit, and the input numbers
        '''
        Perceptron.__init__(self, input_num, f)



def get_training_dataset():
    '''
    select some taining data, must be numpy array.
    :return:
    '''
    input_vecs = np.array([[5],[3],[8],[1.4],[10.1]])
    labels = np.array([5500, 2300, 7600, 1800, 11400])
    return input_vecs, labels


def train_linear_unit():
    '''
    use dataset to train linear ubit.
    :return:
    '''
    lu = LinearUnit(2)
    input_vecs, labels = get_training_dataset()
    plt.plot(input_vecs, labels, 'ro')
    lu.train(input_vecs, labels, 10, 0.01)
    return lu



if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)

    #test
    print('Work 3.4 years , monthly salary = %.2f' %linear_unit.predict(np.array([3.4])))

    #drawing
    x = [3.4, 15, 1.5, 6.3]
    y=[]
    for i in x:
        y.append(linear_unit.predict(np.array(i)))
    print(y)

    plt.plot(x, y)
    plt.legend(['Raw data', 'Trained line'])
    plt.xlabel('Work Year axis')
    plt.ylabel('Salary axis')
    plt.show()





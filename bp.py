#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'document note: Deducte BP algorithm '

__author__ = 'slucius'

from functools import reduce
import random



class Node(object):
    '''
    Node class: record and maintain self informatain, and relate upstream and downstream liks,
    to calculate outputs and errors.
    '''
    def __init__(self, layer_index, node_index):
        '''
        Constract Node object.
        :param layer_index:
        :param node_index:
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0


    def set_output(self, output):
        '''
        Set the Node output, used for INPUT layer.
        :param output:
        :return:
        '''
        self.output = output


    def append_downstream_connection(self, conn):
        '''
        Append Node to downstream connection.
        :param conn:
        :return:
        '''
        self.downstream.append(conn)


    def append_upstream_connection(self, conn):
        '''
        Append Node to upstream connection.
        :param conn:
        :return:
        '''
        self.upstream.append(conn)


    def cacl_output(self):
        '''
        Acoording to neural network feedforward formula to caclculate output.
        :return:
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        #need to define sigmoid function before.
        self.output = sigmoid(output)


    def cacl_hidden_layer_delta(self):
        '''
        caclculate by deducting formulation.
        :return:
        '''
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0)
        self.delta = self.output * (1 - self.output) * downstream_delta


    def cacl_output_layer_delta(self):
        '''
        cacl Output delta.
        :return:
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)


    def __str__(self):
        node_str = '%u-%u: output: %f, delta: %f' %(self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str




class ConstNode(object):
    '''
    for caclculating bias b.
    '''
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1


    def append_downstream_connection(self, conn):
        self.downstream.append(conn)


    def cacl_hidden_layer_delta(self):
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta


    def __str__(self):

        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str




class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))


    def set_ouput(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])


    def cacl_output(self):
        for node in self.nodes[:-1]:
            node.cacl_output()


    def dump(self):
        '''
        peint node information.
        :return:
        '''
        for node in self.nodes:
            print(node)



class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0


    def cacl_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output



    def get_gradient(self):
        return self.gradient


    def update_weight(self, rate):
        self.cacl_gradient()
        self.weight += rate * self.gradient

    def __str__(self):

        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)



class Connections(object):
    def __init__(self):
        self.connections = []


    def add_connection(self, connection):
        self.connections.append(connection)


    def dump(self):
        for conn in self.connections:
            print(conn)



class Network(object):
    def __init__(self, layers):
        '''
        Init full connection network.
        :param layers: 2D arrays, descripe the layer's node numbers.
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))

        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)


    def train(self, labels, data_set, rate, iteration):
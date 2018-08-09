# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:04:18 2018

@author: Ko-Shin Chen
"""

import numpy as np
import tensorflow as tf

def get_M_list(T):
    M = []
    M.append(np.eye(T)) # M0 = id
    M.append(np.eye(T, k=1) + np.eye(T, k=-1)) # M1 = one upper/lower diagonal 1
    
    M2 = np.zeros([T,T])
    M2[0,0] = 1.0
    M2[T-1,T-1] = 1.0
    M.append(M2)
    
    M.append(np.ones([T,T]) - np.eye(T)) # M3 = all off-diagonal 1
    
    return M


def get_X_repeat(X, tau):
    '''
    @param X: [batch, T, d1]
    @return: [batch, T-tau, (tau+1)*d1]
    '''
    _, T, d1 = X.get_shape()
    T = int(T)
    d1 = int(d1)
    X_vect = tf.reshape(X, [-1, T*d1])

    slice_list = []
    for t in range(T-tau):
        slice_list.append(tf.slice(X_vect, [0, t*d1], [-1, (tau+1)*d1]))

    return tf.stack(slice_list, axis=1, name='X_repeat')


def init_weight(shape, ver_name=None):
    return tf.Variable(np.random.normal(loc=0.0, scale=0.05, size=shape), dtype=tf.float64, name=ver_name)


def full_layer(input_layer, output_size, bias=True, layer_name=None):
    if not layer_name:
        layer_name = 'full_layer'

    print(layer_name)

    input_size = int(input_layer.get_shape()[-1])
    W = init_weight([input_size, output_size], layer_name + '/weight')

    if not bias:
        return tf.matmul(input_layer, W, name=layer_name + '/output')

    b = tf.Variable(np.ones([output_size]), dtype=tf.float64, name=layer_name + '/bias')
    return tf.add(tf.matmul(input_layer, W), b, name=layer_name + '/output')


def group_reduce(input_list, layer_size_list, bias=True, act=None, layer_name=None):
    """
    @ param input_list: a list of feature groups each has dim [batch, T, d1]
    @ param layer_size_list: a list of list of integers. It should have the same length as the input_list
    @ return: tf.tensor [batch, T,  sum(output_size orver all group)]
    """
    if len(input_list) != len(layer_size_list):
        print("Invalid arguments!!")
        return

    if not layer_name:
        layer_name = 'group_reduce'

    T = int(input_list[0].get_shape()[1])
    output_list = []
    final_output_size = 0

    for g in range(len(input_list)):
        input_size = int(input_list[g].get_shape()[-1])
        output_list.append(tf.reshape(input_list[g], [-1, input_size])) # [batch* T, input_size]
        final_output_size += layer_size_list[g][-1]

        for l in range(len(layer_size_list[g])):
            output_list[g] = full_layer(output_list[g], layer_size_list[g][l], bias, layer_name + '/group_' + str(g) + '_level_' + str(l))
            if act and l != len(layer_size_list[g]) - 1:
                output_list[g] = act(output_list[g])

    return tf.reshape(tf.concat(output_list, axis=1), [-1, T, final_output_size], name=layer_name + '/output')
    
    



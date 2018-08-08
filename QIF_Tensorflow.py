# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:15:26 2018

@author: Ko-Shin Chen
"""

import tensorflow as tf
import numpy as np
import math

def QIF_Gaussian_tf(X_repeat, y, M, W, corr_restruct=False, alpha=10**-3):
    '''
    Eq(4): mu = eta = <X, W>
    Eq(6): A = I
    
    @param X_repeat: placeholder [batch, T-tau, (tau+1)*d1]
    @param y: placeholder [batch, T-tau]
    @param M: placeholder list [M1, M2, ..., Md], 
              Mi are of dim [T-tau, T-tau]; ref: Eq (8)
    @param W: [tau+1, d1];
    '''
    batch = tf.shape(X_repeat)[0]
    dims = X_repeat.get_shape() # [batch, T-tau, (tau+1)*d1]
    
    W_vect = tf.reshape(W, [int(dims[2]), 1]) # [(tau+1)*d1, 1]

    X_flat = tf.reshape(X_repeat, [batch*int(dims[1]), int(dims[2])]) # [batch*(T-tau), (tau+1)*d1]
    mu = tf.add(tf.reshape(tf.matmul(X_flat, W_vect), [batch, int(dims[1])]), tf.Variable(tf.ones(1, dtype='float64')),name='mu') # [batch, T-tau]
    s = tf.subtract(y, mu, name='s')
    
    d = len(M)
    g_list = []
    
    for j in range(d):
        Ms = tf.expand_dims(tf.matmul(s, M[j]), -1) # [batch, T-tau, 1]
        D_tr_Ms = tf.matmul(tf.transpose(X_repeat, [0,2,1]), Ms, name='D_tr_Ms_'+str(j)) # [batch, (tau+1)*d1, 1]
        g_list.append(D_tr_Ms)
        
    g_i = tf.concat(g_list, 1, name='g_i') # [batch, d*(tau+1)*d1, 1]
    g_m = tf.reduce_mean(g_i, 0, name='g_m') # [d*(tau+1)*d1, 1]
    g_m_tr = tf.transpose(g_m)

    if corr_restruct and alpha >= 1.0:
        val = tf.matmul(g_m_tr, g_m)

    else:
        C_i = tf.matmul(g_i, tf.transpose(g_i,[0,2,1]), name='C_i') # [batch, d*(tau+1)*d1, d*(tau+1)*d1]
        C_m = tf.reduce_mean(C_i, 0, name='C_m') # [d*(tau+1)*d1, d*(tau+1)*d1]
    
        if not corr_restruct:
            C_m_inv = tf.matrix_inverse(C_m, name='C_m_inv')

        else:
            C_m_restruct = tf.add((1-alpha)*C_m, tf.constant(alpha*np.eye(d*int(dims[2]),dtype='float64')), name='C_m_restruct')
            C_m_inv = tf.matrix_inverse(C_m_restruct, name='C_m_inv')

        val = tf.matmul(g_m_tr, tf.matmul(C_m_inv, g_m))
    
    return tf.multiply(tf.squeeze(val), tf.cast(batch, dtype='float64'), name='QIF')
    

def matrix_L12(lam, W_slices, name='matrix_L12'):
    lam *= math.sqrt(2.0)
    sum = tf.sqrt(tf.nn.l2_loss(W_slices[0]))
    for i in range(1, len(W_slices)):
        sum = sum + tf.sqrt(tf.nn.l2_loss(W_slices[i]))
            
    return tf.multiply(sum, lam, name=name)
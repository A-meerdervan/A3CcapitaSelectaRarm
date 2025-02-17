# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:39:41 2018

@author: Alex
"""

# Session is a pointer that points to your structure ofzo

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                        [2]])
product = tf.matmul(matrix1,matrix2) #np.dot(m1,m2)

# method 1 to use sesion
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result)
    


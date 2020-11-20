import tensorflow as tf

def cross_entropy(x, y):
    return -1.0*x*tf.log(y+1.0e-4) - (1.0-x)*tf.log(1.0-y+1.0e-4)
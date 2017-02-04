# Solution is available in the other "solution.py" tab
import numpy as np
# Solution is available in the other "solution.py" tab
import tensorflow as tf


def softmax(logits):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    exp = np.exp(logits)
    sum = np.sum(exp, axis=0)
    return exp / sum


print(softmax([3.0, 1.0, 0.2]))


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output


print(run())


def cross_entropy():
    softmax_data = [0.7, 0.2, 0.1]
    one_hot_data = [1.0, 0.0, 0.0]

    softmax = tf.placeholder(tf.float32)
    one_hot = tf.placeholder(tf.float32)

    # TODO: Print cross entropy from session
    entropy = -tf.reduce_sum(tf.mul(one_hot, tf.log(softmax)))

    with tf.Session() as sess:
        output = sess.run(entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data})

    return output


print(cross_entropy())

import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.sub(tf.div(x, y), 1)

with tf.Session() as sess:
    output = sess.run(z)
    print(output)

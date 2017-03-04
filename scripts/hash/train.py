import tensorflow as tf

from slender.blob import Blob
from slender.net import HashNet, TrainScheme


class Net(HashNet, TrainScheme):
    pass

net = Net(
    num_bits=16,
    gpu_frac=0.3,
)

blob = (
    Blob(
        images=tf.random_normal((16, 224, 224, 3), dtype=tf.float32),
        labels=tf.random_uniform((16,), maxval=8, dtype=tf.int32),
    )
    .f(net.forward)
)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
# val = blob.eval(sess)
val = sess.run([net.dists, net.hard_bits])

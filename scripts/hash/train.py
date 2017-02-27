import tensorflow as tf

from slender.blob import Blob
from slender.net import HashNet as Net


net = Net(
    is_training=True,
    num_bits=256,
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
blob_val = blob.eval(sess)

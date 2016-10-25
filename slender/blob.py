
class Blob(object):
    def __init__(self, **kwargs):
        self._dict = kwargs

        for (key, value) in kwargs.iteritems():
            setattr(self, key, value)

    def func(self, f):
        return f(self)

    def eval(self, sess, feed_dict=None):
        output = sess.run(self._dict, feed_dict=feed_dict)
        return Blob(**output)

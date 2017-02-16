from .util import composite


class Blob(dict):
    def __init__(self, **kwargs):
        super(Blob, self).__init__(**kwargs)
        for (key, value) in kwargs.iteritems():
            setattr(self, key, value)

    def f(self, functions):
        composited = composite(*functions)
        return composited(self)

    def eval(self, sess, **kwargs):
        output = sess.run(self, **kwargs)
        return Blob(**output)

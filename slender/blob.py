class Blob(object):
    def __init__(self, **kwargs):
        self._dict = kwargs
        for (key, value) in kwargs.iteritems():
            setattr(self, key, value)

    def func(self, fn):
        return fn(self)

    def funcs(self, fns):
        if fns:
            return fns[0](self).funcs(fns[1:])
        else:
            return self

    def eval(self, sess, **kwargs):
        output = sess.run(self._dict, **kwargs)
        return Blob(**output)

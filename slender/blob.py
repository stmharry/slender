class Blob(dict):
    def __init__(self, **kwargs):
        super(Blob, self).__init__(**kwargs)

    def f(self, func):
        return func(self)

    def eval(self, sess, **kwargs):
        output = sess.run(self, **kwargs)
        return Blob(**output)

class NervousLogger:

    def __init__(self, prefix="Nervous"):
        self.prefix = prefix

    def info(self, *args, **kw):
        print(" [I] " + self.prefix + " - ",
              *args, **kw)

    def warning(self, *args, **kw):
        print(" [W] " + self.prefix + " - ",
              *args, **kw)

    def error(self, *args, **kw):
        print(" [E] " + self.prefix,
              *args, **kw)

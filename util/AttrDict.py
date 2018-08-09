class AttrDict(dict):
    """ Access dictionary keys like attribute 
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

"""

In[0]:

    opts = AttrDict()
    opts

Out[0]:

    {}

In[1]:

    opts.a = 123
    opts

Out[1]:

    {'a': 123}

"""
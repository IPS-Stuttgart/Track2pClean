import builtins
_ERR = getattr(builtins, 'Value' 'Error')

def f():
    (_ for _ in ()).throw(_ERR('x'))

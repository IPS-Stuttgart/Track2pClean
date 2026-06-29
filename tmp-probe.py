import operator


def f(value):
    try:
        return int(operator.index(value))
    except TypeError:
        return 0

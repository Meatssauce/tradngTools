def hex2(n):
    x = '%x' % (n,)
    return ('0' * (len(x) % 2)) + x

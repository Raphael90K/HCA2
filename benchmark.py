import time as t
from functools import wraps


def timeit(method):
    @wraps(method)
    def time(*args, **kwargs):
        t1 = t.time()
        result = method(*args, **kwargs)
        t2 = t.time()
        print(f"{method.__name__} wurde in {t2 - t1:.2f} Sekunden ausgeführt")
        return result

    return time

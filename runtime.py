import time
import functools
#装饰器语法
def outer(func):
    @functools.wrapps(func)
    def inner(*args,**kwargs):
        ret = func(*args,**kwargs)
        return ret
        
    return  inner





#计算运行时间
def run_time(fun):
    def inner():
        start = time.time()
        fun()
        end = time.time()
        print(f"code running {(end-start):.2f}s")
    return inner


@run_time
def demo():
    time.sleep(1)


demo()    

model = {}

def register(name):
    def inner(cls):
        model[name] = cls
        return cls
    return inner

class mycls():
    def __init__(self):
        pass



@register('f1')
def func1(n):
    cl=mycls()
    return cl

func1('nn')

print(model)
print(model["f1"])



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

import torch

def load_match_dict(model, model_path):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                       k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
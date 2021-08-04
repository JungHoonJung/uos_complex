import os
confpath = os.path.join(os.path.expanduser('~'),'.ucomplex','modulealias.conf')
os.makedirs(os.path.split(confpath)[0], exist_ok=True)

def set_modules_and_alias(*noalias,**packages):
    modules = {}
    if os.path.exists(confpath):
        with open(confpath,'r') as f:
            for mod in f:
                module, alias = mod.split(':')
                modules[module] = alias
    for na in noalias:
        exec(f'import {na}')
        modules[na] = na
    for package in packages:    
        exec(f'import {packages[package]}')
        modules[packages[package]] = package
    
    with open(confpath,'w') as f:
        for module in modules:
            f.write(f'{module}:{modules[module]}\n')
        

def show_modules_and_alias():
    with open(confpath,'r') as f:
        for mod in f:
            print(mod)

if os.path.exists(confpath):
    all = []
    with open(confpath,'r+') as f:
        for mod in f:
            module, alias = mod.rstrip('\n').split(':')
            if module !=alias:
                exec(f'import {module} as {alias}')
            else:
                exec(f'import {module}')
            all.append(alias)

    __all__ = all
else:
    __all__ = []
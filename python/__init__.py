from loop_tool_py import *
from .ui import ui
from . import nn

def symbols(s):
    syms = []
    for n in s.split(" "):
        syms.append(Symbol(n))
    return syms

class Backend():
    def __init__(self, backend):
        self.old_backend = get_default_backend() 
        self.backend = backend
    def __enter__(self):
        set_default_backend(self.backend)
        return self
    def __exit__(self, type, value, traceback):
        set_default_backend(self.old_backend)


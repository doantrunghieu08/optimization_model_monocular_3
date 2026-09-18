import sys
import numpy as np
import joblib
import inspect

def patch_numpy_and_inspect():
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'object'): np.object = object
    if not hasattr(np, 'typeDict'): np.typeDict = np.sctypeDict
    if not hasattr(np, 'complex'): np.complex = complex
    if not hasattr(np, 'unicode'): np.unicode = str
    if not hasattr(np, 'str'): np.str = str
    if not hasattr(inspect, 'getargspec'): inspect.getargspec = inspect.getfullargspec

def configure_stdout_encoding():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

def load_joblib_compat(filepath):
    with open(filepath, 'rb') as f:
        return joblib.load(f)

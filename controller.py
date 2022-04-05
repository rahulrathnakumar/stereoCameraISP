import numpy as np


class Controller():
    '''
    Extend ESC... 
    '''
    def __init__(self, state_dims) -> None:
        self.wES = np.linspace(1.0, 1.75, nES) # frequencies for each param...

import numpy as np

######################################################################
class formula_gen:
    def __init__ (self, gentype. variables):
        self.__getype__ = gentype
        self.__variables__ = variables

    def fit (self, x, y):
        if self.__getype__ == "gen1":
            self.__fit_gen1__ (x, y)

    def predict (self, x):
        pred_y = []
        for i in range(x.shape[0]):
            pred_y.append (0.0)
        return np.asarray(pred_y)

######################################################################

def build_model (gentype. variables):
    
    model = formula_gen (gentype, variables)

    return model 
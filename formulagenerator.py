import numpy as np

######################################################################
class formula_gen:
    def __init__ (self):
        pass

    def predict (self, x):
        pred_y = []
        for i in range(x.shape[0]):
            pred_y.append (0.0)
        return np.asarray(pred_y)

######################################################################

def build_model (variables, train_x, train_y):
    
    model = formula_gen ()

    return model 
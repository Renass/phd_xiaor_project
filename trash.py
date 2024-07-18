import inspect
import torch 

a = torch.zeros(4)
b = torch.zeros(3,2)

def print_shape(x):
    frame = inspect.currentframe().f_back
    variable_names = {id(value): name for name, value in frame.f_locals.items()}
    var_name = variable_names.get(id(x), 'variable')
    print(var_name, 'shape:', x.shape)


print_shape(b)




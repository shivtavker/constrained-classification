import numpy as np

def line_search(func, start_point, stop_point, iterations = 10):
    param_start = 0
    param_stop = 1
    for i in range(iterations):
        param_ter_1 = 2/3*param_start + 1/3*param_stop
        param_ter_2 = 1/3*param_start + 2/3*param_stop
        val_ter_1 = func((1-param_ter_1)*start_point + param_ter_1*stop_point)
        val_ter_2 = func((1-param_ter_2)*start_point + param_ter_2*stop_point)
        val_start = func((1-param_start)*start_point + param_start*stop_point)
        val_stop = func((1-param_stop)*start_point + param_stop*stop_point)
        
        if(val_ter_1 < val_ter_2):
            param_stop = param_ter_2
        else:
            param_start = param_ter_1
    
    return 0.5*(param_start + param_stop) 
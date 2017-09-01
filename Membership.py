import numpy as np
import math

def trimf(x, parameters):
    """
    Triangular fuzzy membership function
    :param x: data point
    :param parameters: a list with 3 real values
    :return: the membership value of x given the parameters
    """
    xx = round(x, 3)
    if xx < parameters[0]:
        return 0
    elif parameters[0] <= xx < parameters[1]:
        return (x - parameters[0]) / (parameters[1] - parameters[0])
    elif parameters[1] <= xx <= parameters[2]:
        return (parameters[2] - xx) / (parameters[2] - parameters[1])
    else:
        return 0

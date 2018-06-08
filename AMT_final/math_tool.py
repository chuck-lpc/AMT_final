import numpy as np

dvar = np.float(0.0000000001)

#def deriv(func, val):
#    return (func(val+dvar)-func(val-dvar))/(dvar*2)

def deriv_func(func):
# description:  finding the derivate of the input function
# param:        func: the target function
# return:       func_out: the derivation function
    def func_out(val):
        result = (func(val+dvar)-func(val-dvar))/(dvar*2)
        return result
    return func_out

def solve_newton_method(func, initval):
# description:  find the zero point of input func, using Newton's method
# param:        func: the target function
#               initval: initial value of Newton's method
# return:       temp: the zero point
    func_deriv = deriv_func(func)
    temp = initval
    while abs(func(temp))>0.00001:
        if(func_deriv(temp)!=0):
            temp = temp - func(temp)/func_deriv(temp)
    return temp

#def func(x):
#    return x**2+2*x+1

def get_roc(func, val):
    deriv_1 = deriv(func, val)
    deriv_2 = (deriv(func, val+dvar)-deriv(func, val-dvar))/(dvar*2)
    return (1+deriv_1**2)**1.5/deriv_2

def get_min_roc(func, upper_lim, lower_lim):
    itera = np.linspace(upper_lim, lower_lim, 100)
    result = np.zeros(100)    
    i = 0
    while i<100:
#        a = get_roc(func, itera[i])
        result[i]=(get_roc(func, itera[i]))
        i = i+1
#    print(np.min(result))
    for i in range(len(result)-1,-1,-1):
      if(result[i]<=0):
        result[i] = np.max(result)#negative replacer

    return np.min(result)

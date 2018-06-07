#! python3
# coding: utf-8
# * =====================================================================================
# *
# *       Filename:  AMT_final.py
# *
# *    Description:  Python program for AMT2018 final thesis
# *
# *        Version:  1.0
# *        Created:  06/07/2018 07:09:10 PM
# *       Revision:  none
# *    Interpreter:  Python3.6
# *
# *         Author:  Pengchao LIANG, chuck_lpc@126.com
# *   Organization:  Tinjin University
# *
# * =====================================================================================
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.fftpack import fft
import math_tool 

def polyfunc(r, theta):
# description:  Definition of the Zernike polynomial. Takes in r and theta, which is
#               the position based on cylindrical coordinate (CC), returns Z value on that
#               position, just like the literal function do.
# param:        r, theta
# return:       sum,  which represents the value of Z
    ci = np.array([0 ,0, 0, 0, 1.205834, 1.209232, -0.073527,
                  0.283878, -0.047157, 0.69305, 0.0821, -0.520752,
                 -0.054379, -0.092302, 0.02262, -0.009395], dtype=np.double)
    zi = np.array([1, r*np.cos(theta), r*np.sin(theta), 2*r**2-1, 
                   r**2*np.cos(2*theta), r**2*np.sin(2*theta), 
                   (3*r**2-2)*r*np.cos(theta), (3*r**2-2)*r*np.sin(theta), 
                   6*r**4-6*r**2+1, r**3*np.cos(3*theta), 
                   r**3*np.sin(3*theta), (4*r**2-3)*r**2*np.cos(2*theta), 
                   (4*r**2-3)*r**2*np.sin(2*theta), 
                   (10*r**4-12*r**2+3)*r*np.cos(theta), 
                   (10*r**4-12*r**2+3)*r*np.sin(theta), 
                   20*r**6-30*r**4+12*r**2-1], dtype = np.double)
    sum = 0
    for i in np.arange(0, np.shape(ci)[0]):
        sum = sum+ci[i]*zi[i]
    return sum

def rot(r, theta):
# description:  Definition of the conical section. Similar to polyfunc
# param:        r, theta
# return:       result,  which represents the value of Z
    c = 1/594.51107
    k = 0
    numerator = c*r**2
    denominator = 1+(1-(1+k)*c**2*r**2)**0.5
    result = numerator/denominator
    return result

def zfunc(r, theta):
# description:  Definition of the surface shape function. This function
#               is a combination of polyfunc and rot. Meanwhile, normalization
#               for r is processed.
# param:        r, theta
# return:       result,  which represents the value of Z
    result = polyfunc(r/200, theta)+rot(r, theta)
    return result

def rtheta2xy(func, x, y):
# description:  Takes a combination of x and y (under rectangular coordinates)(RC), 
#               and gives the Z value of a function which takes cylindrical 
#               coordinate (CC) at that point.
# param:        x, y: coordinate in rectangular coordinates (RC).
# return:       result,  which represents the value of Z
    r = (x**2+y**2)**0.5
    if x > 0:
        theta = np.arctan(y/x)
    elif x < 0:
        theta = np.arctan(y/x)+np.pi
    else:
        if y>=0:
            theta = np.pi/2
        else:
            theta = -np.pi/2
    result = func(r, theta)
    return result

def get_rad_seq(rtfunc, r, N):
# description:  For a given function in cylindrical coordinate (CC), generate a 
#               sample sequence of Z value on a specific circle with radius r.
# param:        rtfunc: the target function which is based on CC
#               r: radius of the circle
#               N: number of sample points
# return:       result,  which is a generated sequence with length N
    result = np.zeros(N)
    vtheta = np.linspace(0, 2*np.pi, N)
    for i in np.arange(0, N):
        result[i] = rtfunc(r, vtheta[i])
    return result

def get_rad_fft(rtfunc, r, N, T, with_column_zero=1):
# description:  Generate a FFT sequence for a given function (CC).
# param:        rtfunc: the target function which is based on CC
#               r: radius of the circle
#               N: number of sample points
#               T: sample spacing
#               with_column_zero: controls whether the FFT sequence will
#                                 have 0th term, 1 (by default) 
#                                 represents yes, 0 represents no
# return:       xf: frequency scale
#               yf: amplitude scale
    if (with_column_zero==1):
        x = np.linspace(0.0, N*T, N)
        y = get_rad_seq(rtfunc, r, N)
        yf_complex = fft(y)
        yf = 1.0/N * np.abs(yf_complex)
        xf = np.linspace(0.0, 1.0/T-1, N)
        return xf, yf
    elif (with_column_zero==0):
        x = np.linspace(0.0, N*T, N)
        y = get_rad_seq(rtfunc, r, N)
        yf_complex = fft(y)[1:]
        yf = 1.0/N * np.abs(yf_complex)
        xf = np.linspace(0.0, 1.0/T-1, N)[1:]
        return xf, yf

def get_rot_part(func):
# description:  Get the rotation part in a given surface shape function
# param:        func: the target function which is based on CC
# return:       func_out: function that represents the rot composition,
#                         whose input and output are similar to that of zfunc
    def func_out(r, theta):
        xf, yf = get_rad_fft(func, r, 50, 1.0/50.0, with_column_zero=1)
        value = yf[0]
        return value
    return func_out

def get_nonrot_part(func):
# description:  Get the non-rotation part in a given surface shape function, 
#               similar to get_rot_part
# param:        func: the target function which is based on CC
# return:       func_out: function that represents the non-rot composition,
#                         whose input and output are similar to that of zfunc
    rot_part = get_rot_part(func)
    def func_out(r, theta):
        value = func(r, theta)-rot_part(r, theta)
        return value
    return func_out

def tool_rad_compensate(func, r_tool):
# description:  decorate func to output a compensated surface shape function
# param:        func: the target function which is based on CC
#               r_tool: radius of lathe tool
# return:       func_out: function that represents the compensated surface shape
#                         function whose input and output are similar to that of zfunc
    def func_out(r, theta):
        def f(x):
            val = func(x, theta)
            return val
        df = math_tool.deriv_func(f)
        def diff_var_g(x):
            val = x - r_tool*df(x)/np.sqrt(1+df(x)**2)-r
            return val
        x = math_tool.solve_newton_method(diff_var_g, r)
        result = f(x)-r_tool/np.sqrt(1+df(x)**2)
        return result
    return func_out

def plot_surface(func ,edge_r=15, title_sub='NO TITLE'):
    X=np.linspace(-edge_r, edge_r, 2*2*int(edge_r)+1)
    Y=np.linspace(-edge_r, edge_r, 2*2*int(edge_r)+1)
    #X, Y represent all possible values in the coordinate
    #using all possible values in linspace object to map all possible points
    X,Y=np.meshgrid(X,Y)
    Z = np.zeros((np.shape(X)[0], np.shape(Y)[0]), dtype = np.double)
    Z_in = np.zeros((np.shape(X)[0], np.shape(Y)[0]), dtype = np.double)
    for i in np.arange(0, np.shape(X)[0]):
        for j in np.arange(0, np.shape(Y)[0]):
            Z[i, j] = rtheta2xy(func, X[i, j], Y[i, j])
            if (X[i, j]**2+Y[i, j]**2>edge_r**2):
                Z_in[i, j] = rtheta2xy(func, 0, 0)
            else:
                Z_in[i, j] = rtheta2xy(func, X[i, j], Y[i, j])

    print('~~~~~~~~~~~~max/min~~~~~~~~~~~~~')
    print(title_sub)
    print('max:')
    print(Z_in.max())
    print('min:')
    print(Z_in.min())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    fig1=plt.figure()#Create a plt object  
    ax1=Axes3D(fig1, title=title_sub)#Create an Axes object (with 3D cord)  
    theta_edge = np.linspace(0, 2*np.pi, 200)
    xr = edge_r*np.cos(theta_edge)
    yr = edge_r*np.sin(theta_edge)
    zr = np.zeros(np.size(theta_edge))
    for i in np.arange(0, np.size(theta_edge)):
        zr[i] = func(edge_r, theta_edge[i])

    ax1.plot(xr, yr, zr, label='edge')
    #construct surface with (x,y,z)
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    ax1.set_xlabel('x label', color='r')  
    ax1.set_ylabel('y label', color='g')  
    ax1.set_zlabel('z label', color='b')#adding lable
#    plt.show()#show all plotting objects
    return

def plot_3d_fft(func, edge_r=15, D=2, N=50, T=1.0/50.0, title_fft='NO TITLE'):
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111, projection='3d', title=title_fft)
    yticks = np.linspace(0, 15, 15*D+1)
    for y in yticks:
        xf, yf = get_rad_fft(func, y, N, T, 0)
        ax0.bar(xf, yf, zs=y, zdir = 'y', alpha = 0.8)
    ax0.set_xlabel('X (Digital Frequency)')
    ax0.set_ylabel('Y (R Position)')
    ax0.set_zlabel('Z (Amplitude)')
    #ax0.set_yticks(yticks)#decomment to enable all valid y coordinate
    return

def plot_z_v_a(func):
    # tool start from the edge, theta = 0
    # S = 1000rpm = 1000/60rps
    # take 600 sample points per round
    # 600*1000/60 points per second
    # time interval between two consecutive points = 60/(600*1000) = 0.1ms

    theta_var = np.linspace(0, 5*2*np.pi, 5*600+1)[:5*600]
    zfts = np.zeros(5*600)#zfts initalization
    vfts = np.zeros(np.size(zfts)-1)#vfts initalization
    afts = np.zeros(np.size(vfts)-1)#afts initalization
    non_rot_composition = get_nonrot_part(func)
    print('non_rot_composition function established')
    compensated = tool_rad_compensate(non_rot_composition, 0.5)
    print('compensated function established')
    

    # f = 1mm/min = 1mm/1000rounds = 1/(1000*2*np.pi)mm/rad
    # r's variation based on angle = 1/(1000*2*np.pi)mm/rad
    for i in np.arange(0, np.size(zfts)):
        r = 15-theta_var[i]*1/(1000*2*np.pi)

    #choose one of the following two lines to switch between compensated/non-compensated
        zfts[i] = compensated(r, theta_var[i])
#        zfts[i] = non_rot_composition(r, theta_var[i])
        print('zfts value calculated %i, %i in total'%(i, np.size(zfts)))
    for i in np.arange(0, np.size(vfts)):
        vfts[i] = (zfts[i+1]-zfts[i])/0.1
        print('vfts value calculated %i, %i in total'%(i, np.size(vfts)))
    for i in np.arange(0, np.size(afts)):
        afts[i] = (vfts[i+1]-vfts[i])/0.1
        print('afts value calculated %i, %i in total'%(i, np.size(afts)))
    time = np.arange(0, 4*600*0.1, 0.1)
    plt.subplot(3, 1, 1)
    plt.plot(time, zfts[:np.size(time)], '-')
    plt.title('Zfts')
    plt.ylabel('Zfts/mm')
    plt.xlabel('Time/ms')

    plt.subplot(3, 1, 2)
    plt.plot(time, vfts[:np.size(time)], '-')
    plt.title('Vfts')
    plt.ylabel('Vfts/(mm/ms)')
    plt.xlabel('Time/ms')

    plt.subplot(3, 1, 3)
    plt.plot(time, afts[:np.size(time)], '-')
    plt.title('Afts')
    plt.ylabel('Afts/(mm/ms^2)')
    plt.xlabel('Time/ms')

#    plt.show()
    return
##################          (validated)            #######################
'''
        Three ways to obtain rot composition
    1, Observe Zernike Polynomial literally, then pick up every term
       that is irrevelent to theta
    2, Process FFT, then pick up the zeroth term(accomplished)
    3, calculate the mean at every radius value
    Difference between original surface and rot composition 
    is non-rot composition
'''
##################        PLOTTING SECTION         #######################
# Number of devided times
#D = 2
# Number of sample points
#N = 50
# sample spacing
#T = 1.0 / 50.0
if __name__ == "__main__":
    plot_z_v_a(zfunc)
    plot_3d_fft(zfunc, edge_r=15, D=2, N=50, T=1.0/50.0, title_fft="Amplitude-Frequency Plot")
    plot_surface(zfunc, edge_r=15, title_sub='Curved Surface')
    plot_surface(get_rot_part(zfunc), edge_r=15, title_sub='Rot Component')
    plot_surface(get_nonrot_part(zfunc), edge_r=15, title_sub='NonRot Component')
    plt.show()#show all plotting objects
#next step:
#最后一问的修改.参照之前草图，修饰刀具z值
#对某一x值，等号左边即为刀具z值
#方法为：输入刀具的径向位置，该径向位置即为g()内的变量
#建立方程，求出实际切削点的径向位置（图中x）（使用牛顿迭代）
#将x带入等号左边，求出刀具z值

#NOTE: max/min should be generated within the circle edge(fixed)

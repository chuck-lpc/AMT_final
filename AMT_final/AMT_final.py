import matplotlib.pyplot as plt #绘图用的模块  
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数  
import numpy as np
from scipy.fftpack import fft

def polyfunc(r, theta):
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
    c = 1/594.51107
    k = 0
    numerator = c*r**2
    denominator = 1+(1-(1+k)*c**2*r**2)**0.5
    result = numerator/denominator
    return result

def zfunc(r, theta):
    result = polyfunc(r/200, theta)+rot(r, theta)
    return result

def rtheta2xy(func, x, y):
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
    result = np.zeros(N)
    vtheta = np.linspace(0, 2*np.pi, N)
    for i in np.arange(0, N):
        result[i] = rtfunc(r, vtheta[i])
    return result

def get_rad_fft(rtfunc, r, N, T, with_column_zero=1):
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
    

#next step:
##########################################################################
##################          (validated)            #######################
####    计算回转部分的三种方式：
####    1、直接观察Zernike多项式中与theta无关的项
####    2、进行FFT变换，第0项即为回转部分
####    3、在每个径向位置计算均值
####    原多项式与之相减得非回转部分

#X=np.arange(-15,15,0.5, dtype=np.double)  
#Y=np.arange(-15,15,0.5, dtype=np.double)#创建了从-2到2，步长为0.1的arange对象  
X=np.linspace(-15, 15, 61)
Y=np.linspace(-15, 15, 61)
#至此X,Y分别表示了取样点的横纵坐标的可能取值  
#用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点  
X,Y=np.meshgrid(X,Y)
Z = np.zeros((np.shape(X)[0], np.shape(Y)[0]), dtype = np.double)
for i in np.arange(0, np.shape(X)[0]):
    for j in np.arange(0, np.shape(Y)[0]):
        Z[i, j] = rtheta2xy(zfunc, X[i, j], Y[i, j])

# Number of sample points
N = 50
# sample spacing
T = 1.0 / 50.0

fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
yticks = np.linspace(0, 15, 31)
for y in yticks:
    xf, yf = get_rad_fft(zfunc, y, N, T, 0)
    ax0.bar(xf, yf, zs=y, zdir = 'y', alpha = 0.8)
ax0.set_xlabel('X (Digital Frequency)')
ax0.set_ylabel('Y (R Position)')
ax0.set_zlabel('Z (Amplitude)')
#ax0.set_yticks(yticks)

#plt.scatter(xf, 1.0/N * np.abs(yf))
#plt.grid()
#plt.show()
########################    PLOTTING SECTION    #################################

fig1=plt.figure()#创建一个绘图对象  
ax1=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)  
plt.title("Curved Surface")#总标题
theta_edge = np.linspace(0, 2*np.pi, 200)
r = 15
xr = r*np.cos(theta_edge)
yr = r*np.sin(theta_edge)
zr = np.zeros(np.size(theta_edge))
for i in np.arange(0, np.size(theta_edge)):
    zr[i] = zfunc(r, theta_edge[i])
ax1.plot(xr, yr, zr, label='curve')
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面  
ax1.set_xlabel('x label', color='r')  
ax1.set_ylabel('y label', color='g')  
ax1.set_zlabel('z label', color='b')#给三个坐标轴注明  
plt.show()#显示模块中的所有绘图对象

#numerical errors

import numpy as np
import matplotlib.pyplot as plt

def forward_diff(f, x, h):
    return (f(x+h)-f(x))/h

def central_diff(f,x,h):
    return (f(x+h)-f(x-h))/(2.0*h)

x = np.linspace(-2.0 * np.pi, 2.0* np.pi, 1000)
h = 0.5
#aproximace cos pomoci derivace
fig,ax = plt.subplots()
ax.plot(x, forward_diff(np.sin,x,h), color="red")
ax.plot(x, central_diff(np.sin,x,h), color="blue")
ax.plot(x, np.cos(x), color= "black", linestyle="--")
fig.show();
#zkoumani chyb
def forward_error(f,x,h,exat_value):
    return np.abs(forward_diff(f,x,h)-exat_value)/ np.abs(exat_value)

def central_error(f,x,h,exat_value):
    return np.abs(central_diff(f,x,h)-exat_value)/ np.abs(exat_value)

x=1.0

h = np.array([2.0**(-n) for n in range (1,60)])

fig, ax = plt.subplots()
ax.loglog(h, forward_error(np.sin,x,h,np.cos(x)))
ax.loglog(h, central_error(np.sin,x,h,np.cos(x)))
fig.show();


#numericka stabilita metody pri reseni y'+y=0 pro y(0)=1
h = 0.1
x = np.arange(0, 10, h)
y_1 = np.zeros(x.size)
y_2 = np.zeros(x.size)

y_1[0] = 1.0
for i in range(x.size - 1):
    y_1[i + 1] = y_1[i] - y_1[i] * h

y_2[0] = 1.0
for i in range(x.size - 1):
    y_2[i + 1] = y_2[i - 1] - y_2[i] * 2.0 * h

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(x, y_1, color="red")
ax[0].plot(x, np.exp(-x), color="black", linestyle="--")
ax[1].plot(x, y_2, color="blue")
ax[1].plot(x, np.exp(-x), color="black", linestyle="--")
fig.tight_layout()
fig.show();

#kontrola vypoctu zlateho rezu pomoci phi^n = phi^n-1 - phi^n a klasickeho nasobeni

phi = np.zeros(20, dtype=np.float16)
phi[0] = 1.0
phi[1] = (np.sqrt(5.0) - 1.0) / 2.0
for n in range(1, 19):
    phi[n + 1] = phi[n - 1] - phi[n]

phi_exact = np.zeros(20, dtype=np.float16)
phi_exact[0] = 1.0
phi_exact[1] = (np.sqrt(5.0) - 1) / 2.0
for n in range(1, 19):
    phi_exact[n + 1] = phi_exact[n] * phi_exact[1]

fig, ax = plt.subplots()
ax.plot(phi, color="red")
ax.plot(phi_exact, color="black", linestyle="--")
fig.show();
#

#condition number C_p of system x  + alpha*y = 1
                          #alpha*x +   y = 0
alpha = np.linspace(-2,2,10000)
C_p = 2.0 * alpha**2 / np.abs(1-alpha**2)

fig, ax = plt.subplots()
ax.plot(alpha, C_p)
fig.show()
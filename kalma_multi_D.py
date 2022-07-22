import numpy as np
import matplotlib.pyplot as plt

noise = np.random.normal(0,5,1000)
signal = np.full((1, 1000), 100.0) + noise
signal = np.squeeze(signal)
plt.plot(signal)
plt.show()

def Kalman(signal):
    out = np.zeros((len(signal),2))
    x = np.matrix([[0.0],[ 0.0]])
    dt = 1
    H = np.matrix( [0 ,1])
    P=np.matrix([[1.0,0],[0,1.0]])
    R=.0001
    A = np.matrix([[1,dt],[0,1]])
    for i in range(len(signal)):
        x = np.matmul(A,x)
        P = A * P * np.transpose(A)
        K = (P*np.transpose(H))* np.linalg.inv(H*P*np.transpose(H)+R)
        z = signal[i]
        resudial =  K * (z - H*x)
        x = x +resudial
        P = P- K * H * P
        out[i,0] = x[0]
        out[i,1] = x[1]
    return out

out = Kalman(signal)

plt.plot(out[:,0])

plt.show()

plt.plot(out[:,1])

plt.show()

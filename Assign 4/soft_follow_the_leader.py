import numpy as np
import numba

# numba used to speed up runtime
@numba.jit(nopython=True)
def soft_follow_the_leader():
    # Online learning with soft follow-the-leader
    eta = 1
    T = 10**6
    g_t = [0,1]
    total_loss = 0
    k = 0
    etas = []
    losses = []

    for iteration in range(2,T):
        p_1 = np.exp(-eta * g_t[0]) / (np.exp(-eta * g_t[0]) + np.exp(-eta * g_t[1]))
        chosen_val = 1 if np.random.random() < p_1 else 0
        f_x = -2 if iteration % 2 == 0 else 2
        total_loss += chosen_val * f_x

        g_t[0] += 0 * f_x
        g_t[1] += 1 * f_x

        losses.append(total_loss)
        etas.append(eta)

        k += 200
        if k > 10**4:
            k = 10**4        
        eta = 1 / k

    return etas,losses


import matplotlib.pyplot as plt

# Set matplotlib font size to 16
plt.rcParams.update({'font.size': 16})

etas, losses = soft_follow_the_leader()    
plt.xlabel('eta')
plt.ylabel('loss')
plt.title('Soft Follow-the-Leader (eta vs loss)')
plt.plot(etas, losses, linewidth=3)
plt.show()

for i in range(5):
    etas, losses = soft_follow_the_leader()    
    plt.plot(range(2,10**6), losses, label=f'{i} run')

plt.xlabel('T')
plt.ylabel('loss')
plt.title('Soft Follow-the-Leader (T vs loss)')
plt.legend()
plt.show()

        
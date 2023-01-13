import numpy as np

from display_helper import display_stats, animated_plot
from train_model import train_q_model

# Q = np.zeros((30, 40, 2))  # Initialization of the Q-values with zeros
Q = np.load('test1.npy')
# There are 30 angle states and 2 actions

# Q = np.load('test1.npy')

Q, x_values, y_values, td_ys = train_q_model(Q)
print(len(x_values))
print(len(y_values))
display_stats(td_ys)
ani = animated_plot(x_values, y_values, f"test1")

np.save('test1.npy', Q, allow_pickle=True)  # save Q model

import numpy as np
from tqdm import tqdm

from yacht_helper import vel, rew, angle_to_state, x_to_state


def train_q_model(Q, init_angle=0, l_r=0.01, full_time=100,
                  channel_abs_width=5, init_x_coord=0, init_y_coord=0, finish_x_coord=0, finish_y_coord=50):
    """

    :param Q:
    :param init_angle: |-1.57079633|wind from left|1.57079633|wind from right |0| wind from back |np.pi| wind from front
    :param l_r: 0.01
    :param full_time: 100
    :param channel_abs_width: 10
    :param init_x_coord: 0
    :param init_y_coord: 0
    :return:
    """
    rho = 0  # Initialize the average reward to 0
    td_ys = []
    x_arr = []
    amount_of_epochs = 10000
    for episode in tqdm(range(amount_of_epochs), desc='Running: Train agent on channel sea task'):  # run for 1000 episodes
        val_y = []
        val_x = []
        x = init_x_coord
        y = init_y_coord

        for i in range(full_time):
            state = angle_to_state(init_angle)
            x_state = x_to_state(x)

            p = np.exp(Q[state, x_state]) / np.sum(np.exp(Q[state, x_state]))  # Action selection using softmaxе
            a = np.random.choice(range(2), p=p)  # Sample the action from the softmax distribution

            out = [-0.1, 0.1][a]  # Get the change in angle as a result of the selected angle

            new_state = angle_to_state(init_angle + out)
            new_x_state = x_to_state(x + vel(init_angle + out) * np.sin(init_angle + out))

            # Calculate the prediction error
            if np.abs(x + vel(init_angle + out) * np.sin(init_angle + out)) > 10:
                delta = -10  # Terminal case
            else:
                delta = rew(init_angle + out) - rho + Q[new_state, new_x_state].max() - Q[state, x_state, a]

            # Update the average reward
            rho += l_r * (rew(init_angle + out) - rho)

            # Update the Q-value
            Q[state, x_state, a] += l_r * delta

            # Update the angle
            init_angle += out

            # Update the x position
            x += vel(init_angle + out) * np.sin(init_angle + out)
            y += vel(init_angle + out) * np.cos(init_angle + out)
            val_x.append(x)
            val_y.append(y)

            # Stop the episode if the pier was hit
            if np.abs(x) > channel_abs_width:
                break
        #     x_arr.append(x)
        td_ys.append(y)
    return Q, val_x, val_y, td_ys
import numpy as np
from matplotlib import pyplot as plt, animation


def display_stats(td_ys):
    td_mean = np.mean(td_ys[:])
    td_stq = np.std(td_ys[:])

    print('Results')
    print('| ===== agent ===== | ===== mean ===== | ===== std ===== |')
    print(f'{"| Trained":<20}| {td_mean:<17.2f}| {td_stq:<16.2f}|')


def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    return line,


def animated_plot(val_x, val_y, filename):
    x = val_x[:]
    y = val_y[:]

    fig, ax = plt.subplots()
    line, = ax.plot(x, y, color='red')

    fig.set_figwidth(3)
    fig.set_figheight(9)

    ax.set_facecolor('xkcd:lightblue')
    fig.patch.set_facecolor('xkcd:lightblue')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.xlim([-10, 10])
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line], interval=25, blit=True)
    ani.save(f'{filename}.gif')
    return ani

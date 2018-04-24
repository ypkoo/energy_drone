import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_graph(v, e, est, filename="plot.pdf"):

    fig, ax1 = plt.subplots()
    k = range(1, e.shape[0] + 1)
    ax1.plot(k, e, 'b-')
    ax1.plot(k, est, 'r-')
    ax1.set_xlabel('time')
    ax1.set_ylabel('energy', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(k, v, 'k:')
    ax2.set_ylabel('velocity', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.savefig(filename)


def draw_graph_ve(v, e, filename="plot.pdf"):

    fig, ax1 = plt.subplots()
    k = range(1, e.shape[0] + 1)
    ax1.plot(k, e, 'b-')
    ax1.set_xlabel('time')
    ax1.set_ylabel('energy', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(k, v, 'k:')
    ax2.set_ylabel('velocity', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.savefig(filename)


if __name__ == '__main__':
    import extract_data as ed
    import sys

    xy = ed.extract_data_speed(n_history=1, filename=sys.argv[1])
    x_data = xy[:, [-2]]
    y_data = xy[:, [-1]]

    draw_graph_ve(x_data, y_data, filename="plot-"+(sys.argv[1]).split('.')[0]+".pdf")



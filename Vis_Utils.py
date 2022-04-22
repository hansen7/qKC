#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import os, numpy as np, matplotlib.pyplot as plt
from matplotlib import animation


#  Ref: https://stackoverflow.com/questions/40642061/
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))

        if den == 1:
            if num == 0: return r'$0$'
            if num == 1: return r'$%s$' % latex
            elif num == -1: return r'$-%s$' % latex
            else: return r'$%s%s$' % (num, latex)
        else:
            if num == 1: return r'${%s}/{%s}$' % (latex, den)
            elif num == -1: return r'${-%s}/{%s}$' % (latex, den)
            else: return r'${%s%s}/{%s}$' % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


def PolygonGen(r=1., sideNum=15, pointNum=10):
    theta = np.linspace(0, 2 * np.pi, sideNum, False)
    x, y = r * np.sin(theta), r * np.cos(theta)
    x, y = np.append(x, x[0]), np.append(y, y[0])
    x, y = np.round(x, 6), np.round(y, 6)
    points = []
    for idx in range(sideNum):
        points.append(np.linspace([x[idx], y[idx]],
                                  [x[idx + 1], y[idx + 1]], pointNum, False))

    return np.concatenate(points, axis=0)


def CostPlot(n_qubits, configs, data_train_test,
             loss, circuit_params, born_probs_list, empirical_probs_list):
    cost_func = configs.cost_func
    train_, test_ = data_train_test
    len_train, len_test = len(train_), len(test_)
    n_born_samples = configs.n_samples['born']
    n_data_samples = configs.n_samples['data']
    n_kernel_samples = configs.n_samples['kernel']
    kernel_type = configs.stein_params['kernel_type']

    plot_colour = ['r', 'b']
    plt.rc("text", usetex=False)
    # plt.rc('font', family='serif')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss for %i qubits" % n_qubits)

    if kernel_type != 'quantum':
        if cost_func == 'mmd':
            plt.plot(loss[('mmd', 'train')], "%so-" % (plot_colour[0]),
                     label="mmd, %i training points,  %i Born samples for a %s kernel."
                           % (len_train, n_born_samples, kernel_type))
            plt.plot(loss[('mmd', 'test')], "%sx-" % (plot_colour[0]),
                     label="mmd, %i test points,  %i Born samples for a %s kernel."
                           % (len_test, n_born_samples, kernel_type))
        elif cost_func == 'stein':
            plt.plot(loss[('stein', 'train')], '%so-' % (plot_colour[1]),
                     label='stein, %i training points,  %i Born samples for a %s kernel.'
                           % (len_train, n_born_samples, kernel_type))
            plt.plot(loss[('stein', 'test')], '%sx-' % (plot_colour[1]),
                     label='stein, %i test points,  %i Born samples for a %s kernel.'
                           % (len_test, n_born_samples, kernel_type))
        elif cost_func == 'sinkhorn':
            plt.plot(loss[('sinkhorn', 'train')], '%so-' % (plot_colour[1]),
                     label='sinkhorn, %i training points,  %i Born samples for a Hamming cost.'
                           % (len_train, n_born_samples))
            plt.plot(loss[('sinkhorn', 'test')], '%sx-' % (plot_colour[1]),
                     label='sinkhorn, %i test points,  %i Born samples for a Hamming cost.'
                           % (len_test, n_born_samples))
        elif cost_func == 'TV':
            plt.plot(loss['TV'], '%so-' % (plot_colour[1]),
                     label='TV, %i data samples,  %i Born samples for a %s kernel.'
                           % (n_data_samples, n_born_samples, kernel_type))
        else:
            raise NotImplementedError
    else:
        if cost_func == 'mmd':
            plt.plot(loss[('mmd', 'train')], '%so-' % (plot_colour[0]),
                     label='mmd, %i training points,  %i Born samples for a %s kernel with %i measurements.'
                           % (len_train, n_born_samples, kernel_type, n_kernel_samples))
            plt.plot(loss[('mmd', 'test')], '%sx-' % (plot_colour[0]),
                     label='mmd, %i test points,  %i Born samples for a %s kernel with %i measurements.'
                           % (len_test, n_born_samples, kernel_type, n_kernel_samples))
        elif cost_func == 'stein':
            plt.plot(loss[('stein', 'train')], '%so' % (plot_colour[1]),
                     label='stein, %i training points,  %i Born samples for a %s kernel with %i measurements.'
                           % (len_train, n_born_samples, kernel_type, n_kernel_samples))
            plt.plot(loss[('stein', 'test')], '%so-' % (plot_colour[1]),
                     label='stein, %i test points,  %i Born samples for a %s kernel with %i measurements.'
                           % (len_test, n_born_samples, kernel_type, n_kernel_samples))
        elif cost_func == 'TV':
            plt.plot(loss['TV'], '%so-' % (plot_colour[1]),
                     label='TV, %i Data samples,  %i Born samples for a %s kernel with %i measurements.'
                           % (n_data_samples, n_born_samples, kernel_type, n_kernel_samples))
        else:
            raise NotImplementedError
    plt.show(block=False)
    plt.pause(15)
    plt.savefig("outputs/plot.png")
    plt.close()
    return loss, circuit_params, born_probs_list, empirical_probs_list


def SaveAnimation(framespersec, fig, N_epochs, N_qubits, N_born_samples,
                  cost_func, kernel_type, data_exact_dict,
                  born_probs_list, axs, N_data_samples):
    """ Animation, will do it later """
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)

    ani = animation.FuncAnimation(fig, animate, frames=len(born_probs_list), fargs=(
        N_qubits, N_born_samples, kernel_type, data_exact_dict, born_probs_list, axs, N_data_samples), interval=10)

    animations_path = './animations/'
    os.makedirs(animations_path, exist_ok=True)
    # MakeDirectory(animations_path)

    ani.save("animations/%s_%iQbs_%s_Kernel_%iSamples_%iEpochs.mp4"
             % (cost_func[0:1], N_qubits, kernel_type[0][0], N_born_samples, N_epochs))

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def PlotAnimate(N_qubits, N_epochs, N_born_samples, cost_func, kernel_type, data_exact_dict):
    """ same as SaveAnimation(), do it later"""
    plots_path = './plots/'
    os.makedirs(plots_path, exist_ok=True)
    plt.legend(prop={'size': 7}, loc='best')
    plt.savefig("plots/%s_%iQbs_%s_%iBSamps_%iEpoch.pdf"
                % (cost_func[0], N_qubits, kernel_type[0][0], N_born_samples, N_epochs))

    fig, axs = plt.subplots()

    axs.set_xlabel("Outcomes")
    axs.set_ylabel("Probability")
    axs.legend(('Born Probs', 'Data Probs'))
    axs.set_xticks(range(len(data_exact_dict)))
    axs.set_xticklabels(list(data_exact_dict.keys()), rotation=70)
    axs.set_title("%i Qubits, %s Kernel, %i Born Samples"
                  % (N_qubits, kernel_type[0][0], N_born_samples))

    plt.tight_layout()

    return fig, axs


def animate(i, N_qubits, N_born_samples, kernel_type, data_exact_dict, born_probs_list, axs, N_data_samples):
    """ Same, later"""
    axs.clear()
    plot_colour = ['r', 'b']
    x = np.arange(len(data_exact_dict))
    axs.bar(x, born_probs_list[i].values(), width=0.2, color=plot_colour[0], align='center')
    axs.bar(x - 0.2, data_exact_dict.values(), width=0.2, color='b', align='center')
    axs.set_title("%i Qbs, %s Kernel, %i Data Samples, %i Born Samples"
                  % (N_qubits, kernel_type[0][0], N_data_samples, N_born_samples))
    axs.set_xlabel("Outcomes")
    axs.set_ylabel("Probability")
    axs.legend(('Born Probs', 'Data Probs'))
    axs.set_xticks(range(len(data_exact_dict)))
    axs.set_xticklabels(list(data_exact_dict.keys()), rotation=70)

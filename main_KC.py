#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import os, time, yaml, shutil, numpy as np, itertools, argparse
from auxiliary_functions import rotate_2d, rotate_3d_z
from Train_Born import TrainBorn_KC
from Param_Init import ParamInit
from Vis_Utils import PolygonGen
from pyquil.api import get_qc


def get_inputs_yaml(file_name):
    args = yaml.load(open(file_name, 'r'), Loader=yaml.FullLoader)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Struct(**args)


def qubits2bins(n_qubits):
    lst = np.array(list(itertools.product([0, 1], repeat=n_qubits)))
    probs = np.ones(2**n_qubits)/(2**n_qubits)
    return lst, probs


if __name__ == "__main__":

    # config_f = sys.argv[1]
    # config_f = "config/input_kc.yml"

    parser = argparse.ArgumentParser("Experiments Configs")
    parser.add_argument('--config_f', type=str, help='config')
    parser.add_argument('--log_dir', type=str, help='log folder')
    parser.add_argument('--rot_angle', type=str, default=None, help='rotation angle')
    parser.add_argument('--noise_var', type=float, default=0.01, help='std of additive noise')
    parser.add_argument('--noise_percent', type=float, default=0., help='percentage with additive noise')

    cfgs = parser.parse_args()

    config_f = cfgs.config_f
    configs = get_inputs_yaml(config_f)
    os.makedirs(r'./log', exist_ok=True)
    n_Born_samples = configs.n_samples['born']
    data_circuit_choice = configs.data_circuit_choice
    logdir = os.path.join('./log', configs.log_folder)
    if cfgs.log_dir is not None:
        logdir = os.path.join('./log', cfgs.log_dir)
    qc = get_qc(configs.device_name, as_qvm=configs.as_qvm)
    n_qubits, data_type = len(qc.qubits()), configs.data_type

    if data_type == 'quantum':
        raise NotImplementedError
    elif data_type == 'classical':
        ''' Try with Synthesized Data '''
        # n_point = 10
        # X, Y = np.linspace(0, 1, n_point + 1), np.linspace(0, 1, n_point + 1)
        # points = []
        # for x in X:
        #     if x == 0 or x == 1:
        #         for y in Y:
        #             points.append([x, y])
        #     else:
        #         for y in [Y[0], Y[-1]]:
        #             points.append([x, y])
        # points_array = np.array(points) - np.array([0.5, 0.5])
        #
        # a_angle = np.pi * 0.2
        # a_array = points_array @ np.array([[np.cos(a_angle), np.sin(a_angle)],
        #                                    [-np.sin(a_angle), np.cos(a_angle)]])
        if configs.synthesized_data['use']:
            syn_cfg = configs.synthesized_data
            rot_angle = np.pi * eval(syn_cfg['rotAngle'])
            points = PolygonGen(r=1, sideNum=syn_cfg['sideNum'], pointNum=syn_cfg['pointNum'])
            rot_points = points @ np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                            [np.sin(rot_angle),  np.cos(rot_angle)]])
        else:
            data_source = configs.external_data
            points = np.load(data_source['dir1'])
            if not data_source['dir2']:
                rot_points = np.load(data_source['dir2'])
            else:
                if cfgs.rot_angle is not None:
                    rot_angle = np.pi * eval(cfgs.rot_angle)
                    print(rot_angle)
                else:
                    rot_angle = np.pi * eval(data_source['rotAngle'])

                if len(points[0]) == 2:
                    rot_points = points @ rotate_2d(rot_angle)
                else:
                    rot_points = points @ rotate_3d_z(rot_angle)
    else:
        raise NotImplementedError

    # np.random.shuffle(data_samples)  # random shuffle the binned data
    # data_train_test = TrainTestPartition(data_samples)  # 80% training, 20% testing

    # Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
    # Set random seed to 0 to initialise the actual Born machine to be trained
    random_seed = 0  # todo: do we need this?
    initial_params = ParamInit(qc, random_seed)
    # data_exact_dict = DataDictFromFile(data_type, n_qubits, 'infinite', data_circuit_choice)

    if cfgs.noise_percent > 0:
        num_dim = len(rot_points[0])
        num_points = len(rot_points)
        select_pts = np.random.randint(low=0, high=num_points,
                                       size=int(num_points * cfgs.noise_percent))
        rot_points[select_pts] += cfgs.noise_var * np.random.randn(len(select_pts), num_dim)

    start_time = time.time()
    os.makedirs(logdir, exist_ok=True)
    shutil.copyfile(config_f, logdir + config_f.split('/')[-1])
    loss, circuit_params, born_probs, empirical_probs = \
        TrainBorn_KC(qc, configs, initial_params, points, rot_points, logdir)
    print('Execution Time is:', time.time() - start_time)

    np.save(r"%s/loss.npy" % logdir, loss)
    np.save(r"%s/born_probs.npy" % logdir, born_probs)
    np.save(r"%s/circuit_params.npy" % logdir, circuit_params)
    np.save(r"%s/empirical_probs.npy" % logdir, empirical_probs)

    if configs.animate:
        pass
        # plt.figure(1)

        # CostPlot(n_qubits, kernel_type, data_train_test, N_samples,
        #          cost_func, loss, circuit_params, born_probs_list, empirical_probs_list)
        #
        # fig, axs = PlotAnimate(N_qubits, N_epochs, N_born_samples,
        #                        cost_func, kernel_type, data_exact_dict)
        # SaveAnimation(5, fig, N_epochs, N_qubits, N_born_samples,
        #               cost_func, kernel_type, data_exact_dict,
        #               born_probs_list, axs, N_data_samples)
        #
        # PrintFinalParamsToFile(
        #     cost_func, data_type, data_circuit_choice, N_epochs,
        #     learning_rate, loss, circuit_params, data_exact_dict,
        #     born_probs_list, empirical_probs_list, qc, kernel_type,
        #     N_samples, stein_params, sinkhorn_eps, run)

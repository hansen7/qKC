#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import os, datetime, numpy as np
from auxiliary_functions import MiniBatchSplit
from sample_gen import BornSampler, PlusMinusSampleGen
from cost_functions import CostFunction, CostGrad, TotalVariationCost, CostFunction_KC, CostGrad_KC

############################################################################
# Train Model Using Stein Discrepancy with either exact kernel and gradient
# or approximate one using samples
############################################################################


def TrainBorn(qc, configs, initial_params,
              data_train_test, data_exact_dict, flag):
    """
    :param qc: simulation instance of a quantum circuit in pyquil
    :param configs: training configurations
    :param initial_params: Circuits initial params
    :param data_train_test: [train data points, test data points]
    :param data_exact_dict: ground truth for the data distribution
    :param flag: `onfly` or `precompute` """

    # set up circuits
    circuit_choice = 'QAOA'
    lr_init = configs.lr_init
    n_qubits = len(qc.qubits())  # 2, qc.qubits() -> list(range(n_qubits))
    print(' === The lattice is: %s === ' % qc.name)
    cost_func, n_epochs = configs.cost_func, configs.n_epochs

    # initial circuit params, defined by NetworkParams() in Param_Init.py
    circuit_params = {('J', 0): initial_params['J'],
                      ('b', 0): initial_params['b'],
                      ('gamma', 0): initial_params['gamma'],
                      ('delta', 0): initial_params['delta'],
                      ('sigma', 0): initial_params['sigma']}

    # create arrays to store weight and bias of each qubit
    batch_size = configs.n_samples['batch']  # number of data points for parameter update
    data_train, data_test = data_train_test  # data_train -> to update the model, data_test -> to calculate the loss

    # todo: check their meanings
    weight_grad, bias_grad = np.zeros((n_qubits, n_qubits)), np.zeros(n_qubits)

    loss = {('mmd', 'train'): [], ('mmd', 'test'): [],
            ('stein', 'train'): [], ('stein', 'test'): [],
            ('sinkhorn', 'train'): [], ('sinkhorn', 'test'): [], 'TV': []}

    # for visualisation
    born_probs_list, emp_probs_list = [], []
    
    # zero init momentum and variance in Adam optimiser
    lr_list_b, lr_list_w_0, lr_list_w_1 = [], [], []
    [m_bias, v_bias] = [np.zeros(n_qubits) for _ in range(2)]
    [m_weights, v_weights] = [np.zeros((n_qubits, n_qubits)) for _ in range(2)]

    for epoch in range(n_epochs):

        print("\n")
        print("=========================================")
        print("Epoch: %3d" % epoch)

        ''' see Sec 3.1 of the 1904.02214 for the descriptions of the notations '''
        # gamma/delta is not to be trained, constant
        circuit_params[('gamma', epoch + 1)] = circuit_params[('gamma', epoch)]
        circuit_params[('delta', epoch + 1)] = circuit_params[('delta', epoch)]
        circuit_params_curr = {'J': circuit_params[('J', epoch)],
                               'b': circuit_params[('b', epoch)],
                               'gamma': circuit_params[('gamma', epoch)],
                               'delta': circuit_params[('delta', epoch)]}

        # generate samples from the IQP/QAOA/IQPy circuit according to the Born rule
        # born_samples: (n_sample, binary array, in (2^n_qubits))
        # born_probs_approx_dict: a dict record the distribution probability of each bins
        # born_probs_exact_dict: the exact probs calculated from the wave function
        born_samples, born_probs_approx_dict, born_probs_exact_dict = \
            BornSampler(qc, configs.n_samples['born'], circuit_params_curr, circuit_choice)
        born_probs_list.append(born_probs_approx_dict)
        emp_probs_list.append(born_probs_approx_dict)
        print('The Born machine outputs distribution:\n', '\t',
              {k: round(v, 3) for k, v in born_probs_approx_dict.items()})
        print('The exact distribution of the data is:\n', '\t',
              {k: round(v, 3) for k, v in data_exact_dict.items()},)

        loss[(cost_func, 'train')].append(CostFunction(
            qc, configs, data_train, data_exact_dict, born_samples, flag))
        loss[(cost_func, 'test')].append(CostFunction(
            qc, configs, data_test, data_exact_dict, born_samples, flag))

        print("The %s loss for epoch %3d " % (cost_func, epoch),
              "is %.4f" % loss[(cost_func, 'train')][epoch])

        # check Total Variation Distribution using the exact output probabilities
        loss['TV'].append(TotalVariationCost(data_exact_dict, born_probs_exact_dict))

        print("The total variation for epoch %3d " % epoch,
              "is %.4f" % loss['TV'][epoch])

        '''Updating bias b[r], control set to 'BIAS' '''
        for bias_index in range(n_qubits):
            born_samples_pm = PlusMinusSampleGen(
                qc, circuit_params_curr, 0, 0, bias_index, 0,
                circuit_choice, 'BIAS', configs.n_samples['batch'])

            # shuffle all samples to avoid bias in mini-batch training
            np.random.shuffle(data_train)
            np.random.shuffle(born_samples)

            # use only first mini-batch of samples for each update
            if batch_size > len(data_train) or batch_size > len(born_samples):
                raise IOError('The batch size is too large')
            else:
                data_batch = MiniBatchSplit(data_train, batch_size)
                born_batch = MiniBatchSplit(born_samples, batch_size)

            bias_grad[bias_index] = CostGrad(qc, configs, data_batch, data_exact_dict,
                                             born_batch, born_samples_pm, flag)

        '''updating weight J[p,q], control set to 'WEIGHTS' '''
        for q in range(n_qubits):
            for p in range(n_qubits):
                if p < q:
                    # draw samples from +/- pi/2 shifted circuits
                    # for each weight update, J_{p, q}
                    born_samples_pm = PlusMinusSampleGen(
                        qc, circuit_params_curr, p, q, 0, 0,
                        circuit_choice, 'WEIGHTS', configs.n_samples['batch'])

                    # shuffle all samples to avoid bias in mini-batch training
                    np.random.shuffle(data_train)
                    np.random.shuffle(born_samples)

                    # use only first mini-batch samples for each update
                    if batch_size > len(data_train) or batch_size > len(born_samples):
                        raise IOError('The batch size is too large')
                    else:
                        data_batch = MiniBatchSplit(data_train, batch_size)
                        born_batch = MiniBatchSplit(born_samples, batch_size)

                    weight_grad[p, q] = CostGrad(qc, configs, data_batch, data_exact_dict,
                                                 born_batch, born_samples_pm, flag)

        # update weights for next epoch
        lr_bias, m_bias, v_bias = AdamLR(lr_init, epoch, bias_grad, m_bias, v_bias)
        lr_weights, m_weights, v_weights = AdamLR(
            lr_init, epoch, weight_grad + np.transpose(weight_grad), m_weights, v_weights)

        lr_list_b.append(lr_bias)
        lr_list_w_0.append(lr_weights[0, :])
        lr_list_w_1.append(lr_weights[1, :])

        circuit_params[('b', epoch+1)] = circuit_params[('b', epoch)] - lr_bias
        circuit_params[('J', epoch+1)] = circuit_params[('J', epoch)] - lr_weights

    now = datetime.datetime.now()
    output_dir = r"./temp/" + now.strftime("%2m%2d_%2H_%2M")
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(r"%s/lr_eff_bias.txt" % output_dir, lr_list_b)
    np.savetxt(r"%s/lr_eff_weights_0.txt" % output_dir, lr_list_w_0)
    np.savetxt(r"%s/lr_eff_weights_1.txt" % output_dir, lr_list_w_1)

    return loss, circuit_params, born_probs_list, emp_probs_list


def TrainBorn_KC(quantum_circuit, configs, initial_params, pc1, pc2, logdir):
    """ Train Born machine for with Kernel Correlation Loss """

    ''' Now only considers MMD '''
    # set up circuits
    circuit_choice = 'QAOA'
    lr_init = configs.lr_init  # 0.01
    lr_decay_rate = configs.lr_rate
    lr_decay_step = configs.lr_step
    n_qubits = len(quantum_circuit.qubits())  # 2
    print(' === The lattice is: %s === ' % quantum_circuit.name)
    cost_func, n_epochs = configs.cost_func, configs.n_epochs
    # circuit params at epoch 0, initialised by NetworkParams() in Param_Init.py
    circuit_params = {('J', 0): initial_params['J'],
                      ('b', 0): initial_params['b'],
                      ('gamma', 0): initial_params['gamma'],
                      ('delta', 0): initial_params['delta'],
                      ('sigma', 0): initial_params['sigma']}

    # create arrays to store weight of bias of each qubit
    batch_size = configs.n_samples['batch']
    # data_train, data_test = data_train_test
    weight_grad, bias_grad = np.zeros((n_qubits, n_qubits)), np.zeros(n_qubits)
    loss = {('mmd', 'train'):      [], ('mmd', 'test'): [],
            ('stein', 'train'):    [], ('stein', 'test'): [],
            ('sinkhorn', 'train'): [], ('sinkhorn', 'test'): [], 'TV': []}

    born_probs_list, emp_probs_list = [], []

    # zero initialize momentum and variance in Adam optimiser
    [m_bias, v_bias] = [np.zeros(n_qubits) for _ in range(2)]
    [m_weights, v_weights] = [np.zeros((n_qubits, n_qubits)) for _ in range(2)]
    lr_list_b, lr_list_w_0, lr_list_w_1 = [], [], []

    for epoch in range(n_epochs):

        print("\n")
        print("=========================================")
        print("Epoch: %3d" % epoch)

        # gamma/delta are set as constants
        circuit_params[('gamma', epoch+1)] = circuit_params[('gamma', epoch)]
        circuit_params[('delta', epoch+1)] = circuit_params[('delta', epoch)]
        circuit_params_curr = {'J': circuit_params[('J', epoch)],
                               'b': circuit_params[('b', epoch)],
                               'gamma': circuit_params[('gamma', epoch)],
                               'delta': circuit_params[('delta', epoch)]}

        born_samples, born_probs_approx_dict, born_probs_exact_dict = \
            BornSampler(quantum_circuit, configs.n_samples['born'],
                        circuit_params_curr, circuit_choice)

        born_probs_list.append(born_probs_approx_dict)
        emp_probs_list.append(born_probs_approx_dict)
        print('The Born machine outputs distribution:\n', '\t',
              {k: round(v, 3) for k, v in born_probs_approx_dict.items()})

        # todo: add TV later
        # we don't have exact distribution of the \theta
        # print('The exact distribution of the data is:\n', '\t',
        #       {k: round(v, 3) for k, v in data_exact_dict.items()},)

        # todo: split train and test
        # qc, configs, born_samples, x, y
        loss[(cost_func, 'train')].append(CostFunction_KC(
            quantum_circuit, configs, born_samples, pc1, pc2))
        # loss[(cost_func, 'test')].append(CostFunction_KC(
        #     quantum_circuit, configs, born_samples, pc1, pc2))
        print("The training loss trajectory is: ",
              [round(l, 5) for l in loss[(cost_func, 'train')]])

        for bias_index in range(n_qubits):
            born_samples_pm = PlusMinusSampleGen(
                quantum_circuit, circuit_params_curr, 0, 0, bias_index, 0,
                circuit_choice, 'BIAS', configs.n_samples['batch'])

            # shuffle all samples to avoid bias in mini-batch training
            # np.random.shuffle(data_train)
            np.random.shuffle(born_samples)

            # use only first mini-batch of samples for each update
            assert batch_size <= len(born_samples), 'batch is larger than the Born samples'
            # data_batch = MiniBatchSplit(data_train, batch_size)
            born_batch = MiniBatchSplit(born_samples, batch_size)

            bias_grad[bias_index] = CostGrad_KC(
                quantum_circuit, configs, born_batch, born_samples_pm, pc1, pc2)

        for q in range(n_qubits):
            for p in range(n_qubits):
                if p < q:
                    # draw samples from +/- pi/2 shifted circuits
                    # for each weight update, J_{p, q}
                    born_samples_pm = PlusMinusSampleGen(
                        quantum_circuit, circuit_params_curr, p, q, 0, 0,
                        circuit_choice, 'WEIGHTS', configs.n_samples['batch'])

                    # shuffle all samples to avoid bias in mini-batch training
                    # np.random.shuffle(data_train)
                    np.random.shuffle(born_samples)
                    assert batch_size <= len(born_samples), 'batch is too large'
                    # data_batch = MiniBatchSplit(data_train, batch_size)
                    born_batch = MiniBatchSplit(born_samples, batch_size)
                    weight_grad[p, q] = CostGrad_KC(quantum_circuit, configs, born_batch,
                                                    born_samples_pm, pc1, pc2)

        lr = lr_init * (lr_decay_rate ** (epoch // lr_decay_step))

        print("Now learning rate is: %.4f" % lr)
        lr_bias, m_bias, v_bias = AdamLR(lr, epoch, bias_grad, m_bias, v_bias)
        lr_weights, m_weights, v_weights = AdamLR(
            lr, epoch, weight_grad + np.transpose(weight_grad), m_weights, v_weights)

        lr_list_b.append(lr_bias)
        lr_list_w_0.append(lr_weights[0, :])
        lr_list_w_1.append(lr_weights[1, :])

        circuit_params[('b', epoch+1)] = circuit_params[('b', epoch)] - lr_bias
        circuit_params[('J', epoch+1)] = circuit_params[('J', epoch)] - lr_weights

    if logdir is None:
        now = datetime.datetime.now()
        output_dir = r"./log/" + now.strftime("%2m%2d_%2H_%2M")
    else:
        output_dir = logdir
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(r"%s/lr_eff_bias.txt" % output_dir, lr_list_b)
    np.savetxt(r"%s/lr_eff_weights_0.txt" % output_dir, lr_list_w_0)
    np.savetxt(r"%s/lr_eff_weights_1.txt" % output_dir, lr_list_w_1)

    return loss, circuit_params, born_probs_list, emp_probs_list


def AdamLR(lr_init, timestep, gradient, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """ to compute Adam learning rate which includes momentum,
    beta1, beta2, epsilon are as recommended in original paper """
    timestep += 1
    m = np.multiply(beta1, m) + np.multiply((1 - beta1), gradient)
    v = np.multiply(beta2, v) + np.multiply((1 - beta2), gradient ** 2)
    corrected_m = np.divide(m, (1 - beta1 ** timestep))
    corrected_v = np.divide(v, (1 - beta2 ** timestep))

    return lr_init * (np.divide(corrected_m, np.sqrt(corrected_v) + epsilon)), m, v

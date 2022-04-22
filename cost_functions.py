#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import itertools, numpy as np
import stein_functions as sf
import auxiliary_functions as aux
import sinkhorn_functions as shornfun
from Quantum_Kernel import QuantumKernelArray, EncodingFunc, EncodingFunc_Nature
from Classical_Kernel import GaussianKernelArray
from file_operations_in import KernelDictFromFile
from Classical_Kernel_Corr import KC_Gaussian_LOO, KC_Gaussian_Set, KC_Gaussian_Set_Dual
from Quantum_Kernel_Corr import KC_Quantum_Set_Cache


def convert_bin2idx(bins):
    ret_list = []
    for bin in bins:
        ret_list.append(int(''.join([str(elem) for elem in bin]), 2))
    return ret_list


def qubits2bins(n_qubits):
    lst = np.array(list(itertools.product([0, 1], repeat=n_qubits)))
    probs = np.ones(2**n_qubits)/(2**n_qubits)
    return lst, probs


# todo: move this function to other files
def KernelSum(s1, s2, kernel_dict):
    """ Load precomputed MMD from the distributions of two sample arrays.
    :param s1 and s2: numpy array, samples from underlying distribution
    :param kernel_dict: contains the kernel values for all pairs of binary strings """
    # if type(samples1) is not np.ndarray or type(samples2) is not np.ndarray:
    #     raise TypeError('The input samples must be in numpy arrays')

    n_s1, n_s2 = s1.shape[0], s2.shape[0]
    kernel_array = np.zeros((n_s1, n_s2))

    for s1_idx in range(n_s1):
        for s2_idx in range(n_s2):
            kernel_array[s1_idx, s2_idx] = kernel_dict[(aux.ToStr(s1[s1_idx]),
                                                        aux.ToStr(s2[s2_idx]))]
    return kernel_array


def TotalVariationCost(dict_one, dict_two):
    """computes the total variation distance between two distributions"""
    if dict_one.keys() != dict_two.keys():
        raise ValueError('Keys are not the same')

    dict_abs_diff = dict()
    for variable in dict_one.keys():
        dict_abs_diff[variable] = abs(dict_one[variable] - dict_two[variable])
    variation_distance = (1 / 4) * sum(dict_abs_diff.values()) ** 2
    return variation_distance


def CostFunction(qc, Configs, data_samples, data_exact_dict, born_samples, flag='onfly'):
    """ compute the cost function between two distributions P and Q from samples from P and Q """

    n_qubits = len(qc.qubits())
    cost_func = Configs.cost_func  # 'mmd'
    kernel_choice = Configs.kernel_type  # 'guassian' or 'quantum'
    score_choice = Configs.stein_params['score']  # 'exact' or 'approx'
    n_kernels = Configs.n_samples['kernel']
    # number of samples for kernel calculation,
    # used in approximate scores

    # extract values and corresponding empirical probabilities from set of samples
    # *_emps: indices of the bins, e.g., [[0, 0], [0, 1], ...], dtype=int
    # *_emp_probs: probabilities of the bins,
    #              e.g., [freq of [0, 0], freq of [0, 1], ...], dtype=float
    born_emps, born_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples)
    data_emps, data_emp_probs, _, _ = aux.ExtractSampleInfo(data_samples)

    # bb -> (Born, Born)
    # bd -> (Born, data)
    # dd -> (data, data)
    if cost_func == 'mmd':
        # compute mmd scores using empirical data distributions
        if score_choice == 'approx':
            if flag == 'onfly':
                if kernel_choice == 'gaussian':
                    sigma = np.array([0.25, 10, 1000])
                    kernel_bb_emp = GaussianKernelArray(born_emps, born_emps, sigma)
                    kernel_bd_emp = GaussianKernelArray(born_emps, data_emps, sigma)
                    kernel_dd_emp = GaussianKernelArray(data_emps, data_emps, sigma)

                elif kernel_choice == 'quantum':
                    # essentially, kernel_bb_emp == kernel_bd_emp == kernel_dd_emp
                    kernel_bb_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_emps,
                                                                encode=eval(Configs.kernel_encode))
                    kernel_bd_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, data_emps,
                                                                encode=eval(Configs.kernel_encode))
                    kernel_dd_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_emps, data_emps,
                                                                encode=eval(Configs.kernel_encode))
                    assert (kernel_bb_emp == kernel_bd_emp).all()
                    assert (kernel_bd_emp == kernel_dd_emp).all()
            elif flag == 'precompute':
                kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
                kernel_bb_emp = KernelSum(born_emps, born_emps, kernel_dict)
                kernel_bd_emp = KernelSum(born_emps, data_emps, kernel_dict)
                kernel_dd_emp = KernelSum(data_emps, data_emps, kernel_dict)

            else:
                raise ValueError('flag must be either \'onfly\' or \'precompute\'')

            loss = np.dot(born_emp_probs @ kernel_bb_emp, born_emp_probs) - \
                2 * np.dot(born_emp_probs @ kernel_bd_emp, data_emp_probs) + \
                np.dot(data_emp_probs @ kernel_dd_emp, data_emp_probs)
            return loss

        # compute mmd scores using exact data distribution
        elif score_choice == 'exact':
            # a list strings -> numpy array
            data_exts = aux.SampleListToArray(list(data_exact_dict.keys()), n_qubits, 'int')
            data_ext_probs = np.asarray(list(data_exact_dict.values()))

            if flag == 'onfly':
                if kernel_choice == 'gaussian':
                    # a bit duplicated calculations
                    sigma = np.array([0.25, 10, 1000])
                    kernel_bb_emp = GaussianKernelArray(born_emps, born_emps, sigma)
                    kernel_bd_emp = GaussianKernelArray(born_emps, data_exts, sigma)
                    kernel_dd_emp = GaussianKernelArray(data_exts, data_exts, sigma)

                elif kernel_choice == 'quantum':
                    kernel_bb_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_emps)
                    kernel_bd_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, data_exts)
                    kernel_dd_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_exts, data_exts)

            elif flag == 'precompute':
                kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
                kernel_bb_emp = KernelSum(born_emps, born_emps, kernel_dict)
                kernel_bd_emp = KernelSum(born_emps, data_exts, kernel_dict)
                kernel_dd_emp = KernelSum(data_exts, data_exts, kernel_dict)

            else:
                raise ValueError('flag must be either \'onfly\' or \'precompute\'')

            # (2**n_qubits, ), (2**n_qubits, 2**n_qubits), (2**n_qubits, )
            loss = born_emp_probs@kernel_bb_emp@born_emp_probs - \
                   2 * born_emp_probs@kernel_bd_emp@data_ext_probs + \
                   data_ext_probs@kernel_dd_emp@data_ext_probs
            return loss

    elif cost_func == 'stein':
        if flag == 'onfly':
            if kernel_choice == 'gaussian':
                sigma = np.array([0.25, 10, 1000])
                kernel_array = GaussianKernelArray(born_emps, born_emps, sigma)

            elif kernel_choice == 'quantum':
                kernel_array, _, _, _ = QuantumKernelArray(qc, n_kernels, born_samples, born_samples)

            else:
                raise ValueError('Stein only supports Gaussian kernel currently')
        elif flag == 'precompute':
            kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
            kernel_array = KernelSum(born_emps, born_emps, kernel_dict)
        else:
            raise ValueError('\'flag\' must be either \'Onfly\' or \'Precompute\'')

        kernel_stein_weighted = sf.WeightedKernel(qc, Configs, kernel_array, data_samples,
                                                  data_exact_dict, born_emps, born_emps, 'precompute')

        return np.dot(np.dot(born_emp_probs, kernel_stein_weighted), born_emp_probs)

    elif cost_func == 'sinkhorn':
        return shornfun.FeydySink(born_samples, data_samples, Configs.sinkhorn_eps).item()

    else:
        raise ValueError('cost_func must be either mmd, stein, or sinkhorn')


def CostGrad(qc, Configs, data_samples, data_exact_dict,
             born_samples, born_samples_pm, flag):
    """ Computes the gradient of the cost function """

    cost_func = Configs.cost_func
    kernel_choice = Configs.kernel_type
    n_kernels = Configs.n_samples['kernel']
    score_choice = Configs.stein_params['score']
    [born_samples_plus, born_samples_minus] = born_samples_pm

    # extract unique samples, and corresponding probabilities from a list of samples
    born_emps, born_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples)
    data_emps, data_emp_probs, _, _ = aux.ExtractSampleInfo(data_samples)
    born_p_emps, born_p_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples_plus)
    born_m_emps, born_m_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples_minus)

    if cost_func == 'mmd':
        if score_choice == 'approx':
            if flag == 'onfly':
                if kernel_choice == 'gaussian':
                    sigma = np.array([0.25, 10, 1000])
                    # Compute the Gaussian kernel on the fly for all pairs of samples required
                    k_born_p_emp = GaussianKernelArray(born_emps, born_p_emps, sigma)
                    # k_born_m_emp = GaussianKernelArray(born_emps, born_m_emp_probs, sigma)
                    k_born_m_emp = GaussianKernelArray(born_emps, born_m_emps, sigma)
                    k_data_p_emp = GaussianKernelArray(data_emps, born_p_emps, sigma)
                    # k_data_m_emp = GaussianKernelArray(data_emps, born_m_emp_probs, sigma)
                    k_data_m_emp = GaussianKernelArray(data_emps, born_m_emps, sigma)

                elif kernel_choice == 'quantum':
                    # compute the quantum kernel on the fly for all pairs of samples required
                    # commented the original wrong implementations from:
                    #   https://github.com/BrianCoyle/IsingBornMachine
                    k_born_p_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_p_emps,
                                                               encode=eval(Configs.kernel_encode))
                    # k_born_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_m_emp_probs)
                    k_born_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_m_emps,
                                                               encode=eval(Configs.kernel_encode))
                    k_data_p_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_emps, born_p_emps,
                                                               encode=eval(Configs.kernel_encode))
                    # k_data_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_emps, born_m_emp_probs)
                    k_data_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_emps, born_m_emps,
                                                               encode=eval(Configs.kernel_encode))

                    assert (k_born_p_emp == k_data_p_emp).all()
                    assert (k_born_m_emp == k_data_m_emp).all()

            elif flag == 'precompute':
                # to speed up computation, read in precomputed kernel dictionary from a file.
                kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
                k_born_p_emp = KernelSum(born_emps, born_p_emps, kernel_dict)
                k_born_m_emp = KernelSum(born_emps, born_m_emps, kernel_dict)
                k_data_p_emp = KernelSum(data_emps, born_p_emps, kernel_dict)
                k_data_m_emp = KernelSum(data_emps, born_m_emps, kernel_dict)

            else:
                raise ValueError('\'flag\' must be either \'onfly\' or \'precompute\'')

            loss_grad = 2 * (np.dot(np.dot(born_emp_probs, k_born_m_emp), born_m_emp_probs) -
                             np.dot(np.dot(born_emp_probs, k_born_p_emp), born_p_emp_probs) -
                             np.dot(np.dot(data_emp_probs, k_data_m_emp), born_m_emp_probs) +
                             np.dot(np.dot(data_emp_probs, k_data_p_emp), born_p_emp_probs))

        elif score_choice == 'exact':
            # compute MMD using exact data probabilities if score is exact
            data_exacts = aux.SampleListToArray(list(data_exact_dict.keys()), len(qc.qubits()), 'int')
            data_exact_probs = np.asarray(list(data_exact_dict.values()))

            if flag == 'onfly':
                if kernel_choice == 'gaussian':
                    sigma = np.array([0.25, 10, 1000])
                    k_born_p_emp = GaussianKernelArray(born_emps, born_p_emps, sigma)

                    # original code also has this problems, raises an issue
                    # k_born_m_emp = GaussianKernelArray(born_emps, born_m_emp_probs, sigma)
                    k_born_m_emp = GaussianKernelArray(born_emps, born_m_emps, sigma)
                    k_data_p_emp = GaussianKernelArray(data_exacts, born_p_emps, sigma)
                    # k_data_m_emp = GaussianKernelArray(data_exacts, born_m_emp_probs, sigma)
                    k_data_m_emp = GaussianKernelArray(data_exacts, born_m_emps, sigma)

                elif kernel_choice == 'quantum':
                    # todo: this also should be changed
                    k_born_p_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_p_emps)
                    k_born_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_m_emp_probs)
                    k_data_p_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_exacts, born_p_emps)
                    k_data_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, data_exacts, born_m_emp_probs)

            elif flag == 'precompute':
                kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
                k_born_p_emp = KernelSum(born_emps, born_p_emps, kernel_dict)
                k_born_m_emp = KernelSum(born_emps, born_m_emps, kernel_dict)
                k_data_p_emp = KernelSum(data_exacts, born_p_emps, kernel_dict)
                k_data_m_emp = KernelSum(data_exacts, born_m_emps, kernel_dict)

            else:
                raise ValueError('\'flag\' must be either \'onfly\' or \'precompute\'')

            # essentially, k_born_m_emp == k_born_p_emp == k_data_m_emp == k_data_p_emp
            loss_grad = 2 * (born_emp_probs @ k_born_m_emp @ born_m_emp_probs -
                             born_emp_probs@k_born_p_emp@born_p_emp_probs -
                             data_exact_probs@k_data_m_emp@born_m_emp_probs +
                             data_exact_probs@ k_data_p_emp@born_p_emp_probs)

    elif cost_func == 'stein':

        [born_samples_plus, born_samples_minus] = born_samples_pm

        if flag == 'onfly':
            if kernel_choice == 'gaussian':
                sigma = np.array([0.25, 10, 1000])
                k_born_p_emp = GaussianKernelArray(born_emps, born_p_emps, sigma)
                k_born_m_emp = GaussianKernelArray(born_emps, born_m_emps, sigma)

            elif kernel_choice == 'quantum':
                k_born_p_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_p_emps)
                k_born_m_emp, _, _, _ = QuantumKernelArray(qc, n_kernels, born_emps, born_m_emp_probs)

        elif flag == 'precompute':
            kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
            k_born_p_emp = KernelSum(born_emps, born_p_emps, kernel_dict)
            k_born_m_emp = KernelSum(born_emps, born_m_emps, kernel_dict)

        kernel_plus_born_emp = np.transpose(k_born_p_emp)
        kernel_minus_born_emp = np.transpose(k_born_m_emp)

        # Compute the weighted kernel for each pair of samples required in the gradient of Stein Cost Function
        # kernel_stein_weighted = sf.WeightedKernel(qc, Configs, kernel_array, data_samples,
        #                                           data_exact_dict, born_emps, born_emps, 'precompute')

        kappa_q_born_bornplus = sf.WeightedKernel(qc, Configs, k_born_p_emp,
                                                  data_samples, data_exact_dict,
                                                  born_emps, born_p_emps, flag)
        kappa_q_bornplus_born = sf.WeightedKernel(qc, Configs, kernel_plus_born_emp,
                                                  data_samples, data_exact_dict,
                                                  born_p_emps, born_emps, flag)
        kappa_q_born_bornminus = sf.WeightedKernel(qc, Configs, k_born_m_emp,
                                                   data_samples, data_exact_dict,
                                                   born_emps, born_m_emps, flag)
        kappa_q_bornminus_born = sf.WeightedKernel(qc, Configs, kernel_minus_born_emp,
                                                   data_samples, data_exact_dict,
                                                   born_m_emps, born_emps, flag)

        loss_grad = np.dot(np.dot(born_emp_probs, kappa_q_born_bornminus), born_m_emp_probs) + \
                    np.dot(np.dot(born_m_emp_probs, kappa_q_bornminus_born), born_emp_probs) - \
                    np.dot(np.dot(born_emp_probs, kappa_q_born_bornplus), born_p_emp_probs) - \
                    np.dot(np.dot(born_p_emp_probs, kappa_q_bornplus_born), born_emp_probs)

    elif cost_func == 'sinkhorn':
        # loss_grad = shornfun.SinkhornGrad(born_samples_pm, data_samples, sinkhorn_eps)
        loss_grad = shornfun.SinkGrad(born_samples, born_samples_pm, data_samples, Configs.sinkhorn_eps)
    else:
        raise ValueError('cost_func must be either mmd, stein, or sinkhorn')

    return loss_grad


def CostFunction_KC(qc, configs, born_samples, x, y, data_exact_dict=None, rot_axis=0):
    """ Cost Function for KC on Points Clouds """

    n_pc, n_dim = x.shape
    n_qubits = len(qc.qubits())
    cost_func = configs.cost_func
    # n_kernels = Configs.n_samples['kernel']
    score_choice = configs.stein_params['score']
    kernel_choice = configs.kernel_type

    # extract values and corresponding empirical probabilities from set of samples
    # *_emps: indices of the bins, e.g., [[0, 0], [0, 1], ...], dtype=int
    # *_emp_probs: probabilities of the bins, e.g., [freq00, freq01, ...], dtype=float
    born_emps, born_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples)

    ''' sometime there might be zero samples from some bins '''
    # kc_tx_y = kc_tx_y[convert_bin2idx(born_emps)]
    # angle_emps = aux.born2angle(born_emps, (0, 2*np.pi))
    full_born_emps, _ = qubits2bins(n_qubits)
    angle_emps = aux.born2angle(full_born_emps, (0, 2*np.pi))
    if n_dim == 2:
        rot_emps = aux.rotate_2d(angle_emps)  # (num_bins, 2, 2)
    elif n_dim == 3:
        # raise NotImplementedError
        if configs.external_data['kc_cache'] is not None:
            kc_tx_y = np.load(configs.external_data['kc_cache'])
            return 2 * kc_tx_y[convert_bin2idx(born_emps)] @ born_emp_probs
        rot_emps = aux.rotate_3d_z(angle_emps)  # (num_bins, 2, 2)
    else:
        raise ValueError("point cloud should be either 2d or 3d")

    if cost_func == 'mmd':
        # compute mmd scores using empirical sampling probabilities
        if score_choice == 'approx':
            # compute the classical Gaussian kernel for all samples
            if kernel_choice == 'gaussian':
                # todo: find the best sigmas sets
                # sigmas = np.array([0.25, 10, 1000])
                # sigmas = np.array([0.001, 0.01, 0.1])
                # sigmas = np.array([0.01, 0.05, 0.1])
                sigmas = np.array([0.01])

                # kc_tx_tx and kc_y_y are constants
                kc_tx_tx = KC_Gaussian_LOO(x, sigmas, rot_emps[convert_bin2idx(born_emps)])
                kc_y_y = KC_Gaussian_LOO(y, sigmas)
                kc_tx_y = KC_Gaussian_Set(x, y, sigmas, rot_emps[convert_bin2idx(born_emps)])

            elif kernel_choice == 'quantum':
                # todo: under development
                kc_tx_y = KC_Quantum_Set_Cache(n_qubits, configs.synthesized_data['sideNum'])
                kc_tx_y = kc_tx_y[convert_bin2idx(born_emps)]
                # raise NotImplementedError("tomorrow's work")

            else:
                raise NotImplementedError

            # `born_emp_probs @ kc_tx_tx' and `kc_y_y' are constants
            # return born_emp_probs @ kc_tx_tx + kc_y_y
            # + 2 * kc_tx_y @ born_emp_probs
            return 2 * kc_tx_y @ born_emp_probs

        elif score_choice == 'exact':
            raise NotImplementedError

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def CostGrad_KC(qc, configs, born_samples, born_samples_pm, x, y):
    loss_grad = 0
    n_qubits = len(qc.qubits())
    n_point, n_dim = x.shape
    cost_func = configs.cost_func
    # n_kernels = configs.n_samples['kernel']
    score_choice = configs.stein_params['score']
    kernel_choice = configs.stein_params['kernel_type']
    [born_samples_plus, born_samples_minus] = born_samples_pm

    born_emps, born_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples)
    born_p_emps, born_p_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples_plus)
    born_m_emps, born_m_emp_probs, _, _ = aux.ExtractSampleInfo(born_samples_minus)

    full_born_emps, _ = qubits2bins(n_qubits)
    angle_emps = aux.born2angle(full_born_emps, (0, 2*np.pi))

    # angle_emps = aux.born2angle(born_emps, (0, 2*np.pi))
    # angle_p_emps = aux.born2angle(born_p_emps, (0, 2*np.pi))
    # angle_n_emps = aux.born2angle(born_m_emps, (0, 2*np.pi))

    if n_dim == 2:
        rot_emps = aux.rotate_2d(angle_emps)
        rot_p_emps = rot_emps[convert_bin2idx(born_p_emps)]
        rot_m_emps = rot_emps[convert_bin2idx(born_m_emps)]
        rot_emps = rot_emps[convert_bin2idx(born_emps)]

    elif n_dim == 3:
        if configs.external_data['kc_cache'] is not None:
            kc_tx_y = np.load(configs.external_data['kc_cache'])
            kc_tpx_y = kc_tx_y[convert_bin2idx(born_p_emps)]
            kc_tmx_y = kc_tx_y[convert_bin2idx(born_m_emps)]
            loss_grad = - kc_tpx_y @ born_p_emp_probs + born_m_emp_probs @ kc_tmx_y
            return loss_grad
        raise NotImplementedError
    else:
        raise ValueError("point cloud should be either 2d or 3d")

    assert score_choice == 'approx'
    if cost_func == 'mmd' and n_dim == 2:
        if kernel_choice == 'gaussian':
            # sigmas = np.array([0.25, 10, 1000])
            # sigmas = np.array([0.1, 1, 10])
            # sigmas = np.array([0.01, 0.05, 0.1])
            sigmas = np.array([0.01])

            kc_tmx_tx = KC_Gaussian_Set_Dual(x, y, sigmas, rot_m_emps, rot_emps)
            kc_tpx_tx = KC_Gaussian_Set_Dual(x, y, sigmas, rot_p_emps, rot_emps)
            kc_tmx_y = KC_Gaussian_Set(x, y, sigmas, rot_m_emps)
            kc_tpx_y = KC_Gaussian_Set(x, y, sigmas, rot_p_emps)

            bbm_emp_probs = np.outer(born_m_emp_probs, born_emp_probs)
            bbp_emp_probs = np.outer(born_p_emp_probs, born_emp_probs)
            first_two_terms = - (kc_tmx_tx * bbm_emp_probs).sum() + (kc_tpx_tx * bbp_emp_probs).sum()
            last_two_terms = - kc_tpx_y @ born_p_emp_probs + born_m_emp_probs @ kc_tmx_y
            loss_grad = first_two_terms + last_two_terms
            loss_grad = last_two_terms
            # print("first two grad terms: %.8f" % first_two_terms)
            # print("last two grad terms: %.8f" % last_two_terms)

        elif kernel_choice == 'quantum':
            # todo: quantum kernel, unresolved
            kc_tx_y = KC_Quantum_Set_Cache(n_qubits, configs.synthesized_data['sideNum'])
            kc_tpx_y = kc_tx_y[convert_bin2idx(born_p_emps)]
            kc_tmx_y = kc_tx_y[convert_bin2idx(born_m_emps)]
            loss_grad = - kc_tpx_y @ born_p_emp_probs + born_m_emp_probs @ kc_tmx_y
            # raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return loss_grad


if __name__ == "__main__":

    ''' Test codes in CostFunction_KC() '''

    born_emps = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    born_emp_probs = np.ones(8)/8
    x = np.random.randn(100, 2)
    y = np.random.randn(100, 2)

    angle_emps = aux.born2angle(born_emps, (0, 2*np.pi))
    n_pc, n_dim = x.shape
    rot_emps = aux.rotate_2d(angle_emps)
    sigmas = np.array([0.25, 10, 1000])
    kc_tx_tx = KC_Gaussian_LOO(x, sigmas, rot_emps)
    kc_y_y = KC_Gaussian_LOO(y, sigmas)
    kc_tx_y = KC_Gaussian_Set(x, y, sigmas, rot_emps)

    loss = born_emp_probs @ kc_tx_tx + kc_y_y - 2 * kc_tx_y @ born_emp_probs

    ''' Test codes in CostGrad_KC() '''

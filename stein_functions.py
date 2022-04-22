#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import numpy as np
import stein_score as ss
from Classical_Kernel import GaussianKernel
from file_operations_in import KernelDictFromFile
from auxiliary_functions import ShiftString, ToStr


def DeltaTerms(qc, Configs, kernel_array, s1, s2, flag='onfly'):
    """ computes the shifted *Delta* terms used in the Stein Discrepancy
    :param qc:
    :param Configs:
    :param kernel_array:
    :param s1 and s2:
    :param flag:
    :return:
    """
    # todo: prune
    n_qubits = len(qc.qubits())
    n_s1, n_s2 = len(s1), len(s2)
    n_kernels = Configs.n_samples.kernel
    kernel_choice = Configs.stein_params.kernel_type
    kernel_x_shifted = np.zeros((n_s1, n_s2, n_qubits))
    kernel_y_shifted, kernel_xy_shifted = kernel_x_shifted.copy(), kernel_x_shifted.copy()
    delta_x_kernel, delta_y_kernel, delta_xy_kernel = [kernel_x_shifted.copy() for _ in range(3)]

    assert flag is 'only' or 'precompute'
    for s1_idx in range(n_s1):
        for s2_idx in range(n_s2):
            for qubit in range(n_qubits):
                if flag is 'onfly':
                    kernel = kernel_array[s1_idx][s2_idx]
                    sample1, sample2 = s1[s1_idx], s2[s2_idx]
                    shiftedsample1 = ShiftString(sample1, qubit)
                    shiftedsample2 = ShiftString(sample2, qubit)

                    if kernel_choice is 'gaussian':
                        sigma = np.array([0.25, 10, 1000])
                        kernel_x_shifted[s1_idx, s2_idx, qubit] = GaussianKernel(shiftedsample1, sample2, sigma)
                        kernel_y_shifted[s1_idx, s2_idx, qubit] = GaussianKernel(sample1, shiftedsample2, sigma)
                        kernel_xy_shifted[s1_idx, s2_idx, qubit] = GaussianKernel(shiftedsample1, shiftedsample2, sigma)

                else:
                    kernel_dict = KernelDictFromFile(qc, n_kernels, kernel_choice)
                    sample1, sample2 = ToStr(s1[s1_idx]), ToStr(s2[s2_idx])
                    shiftedsample1 = ToStr(ShiftString(sample1, qubit))
                    shiftedsample2 = ToStr(ShiftString(sample2, qubit))
                    kernel = kernel_dict[(sample1, sample2)]

                    kernel_x_shifted[s1_idx][s2_idx][qubit] = kernel_dict[(shiftedsample1, sample2)]
                    kernel_y_shifted[s1_idx][s2_idx][qubit] = kernel_dict[(sample1, shiftedsample2)]
                    kernel_xy_shifted[s1_idx][s2_idx][qubit] = kernel_dict[(shiftedsample1, shiftedsample2)]

                delta_x_kernel[s1_idx, s2_idx, qubit] = kernel - kernel_x_shifted[s1_idx, s2_idx, qubit]
                delta_y_kernel[s1_idx, s2_idx, qubit] = kernel - kernel_y_shifted[s1_idx, s2_idx, qubit]
                delta_xy_kernel[s1_idx, s2_idx, qubit] = kernel - kernel_xy_shifted[s1_idx, s2_idx, qubit]

    trace = n_qubits * kernel_array - kernel_x_shifted.sum(axis=2) - \
            kernel_y_shifted.sum(axis=2) + kernel_xy_shifted.sum(axis=2)

    return delta_x_kernel, delta_y_kernel, trace


def WeightedKernel(qc, Configs, kernel_array, samples, probs, s1, s2, flag='onfly'):
    """ computes the weighted kernel for all samples from the two distributions s1 and s2"""

    delta_x_kernel, delta_y_kernel, trace = DeltaTerms(qc, Configs, kernel_array, s1, s2, flag)

    assert Configs.stein_params.kernel_type is 'gaussian'
    score_approx = Configs.stein_params.score
    stein_sigma = np.array([0.25, 10, 1000])
    J = Configs.stein_params.eigen_vecs
    chi = Configs.stein_params.eta

    if score_approx is 'exact':
        score_matrix_1 = ss.MassSteinScore(s1, probs)
        score_matrix_2 = ss.MassSteinScore(s2, probs)
    elif score_approx is 'identity':
        score_matrix_1 = ss.IdentitySteinScore(samples, "gaussian", chi, stein_sigma)
        score_matrix_2 = ss.IdentitySteinScore(samples, "gaussian", chi, stein_sigma)
    elif score_approx is 'spectral':
        # compute score matrix using spectral method for all samples,
        # x and y according to the data distribution.
        score_matrix_1 = ss.SpectralSteinScore(s1, samples, J, stein_sigma)
        score_matrix_2 = ss.SpectralSteinScore(s2, samples, J, stein_sigma)
    else:
        raise IOError('Please enter \'exact\', \'identity\' or \'spectral\' for score_approx')

    n_s1, n_s2 = len(s1), len(s2)
    w_kernel = np.zeros((n_s1, n_s2))  # weighted kernel

    for s1_idx in range(n_s1):
        for s2_idx in range(n_s2):
            delta_x = np.transpose(delta_x_kernel[s1_idx][s2_idx])
            delta_y = delta_y_kernel[s1_idx][s2_idx]
            kernel = kernel_array[s1_idx][s2_idx]

            w_kernel[s1_idx, s2_idx] = np.dot(np.transpose(score_matrix_1[s1_idx]),
                                                     kernel * score_matrix_2[s2_idx]) \
                                              - np.dot(np.transpose(score_matrix_1[s1_idx]), delta_y) \
                                              - np.dot(delta_x, score_matrix_2[s1_idx]) \
                                              + trace[s1_idx][s2_idx]

    return w_kernel

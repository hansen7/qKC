#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import numpy as np
from pyquil.quil import Program
from pyquil.gates import PHASE, CPHASE
from pyquil.api import WavefunctionSimulator
from Param_Init import Hadamard2All
from auxiliary_functions import IntegerToString


def EncodingFunc(n_qubits, sample):
    """
    Non-Linear function for encoded samples for Quantum Kernel Circuit
    to act as graph weights/biases """

    assert type(sample) is np.ndarray, "sample should be a numpy array"
    Z, ZZ = np.zeros(n_qubits), np.zeros((n_qubits, n_qubits))
    for qubit in range(n_qubits):
        j = 0
        Z[qubit] = (np.pi/4) * int(sample[qubit])
        while j < qubit:
            ZZ[qubit, j] = (np.pi/4 - int(sample[qubit])) * (np.pi/4 - int(sample[j]))
            ZZ[j, qubit] = ZZ[qubit, j]
            j += 1
    encoded_sample = {'Interaction': ZZ, 'Local': Z}
    return encoded_sample


def EncodingFunc_Nature(n_qubits, sample):
    """
    Non-Linear function for encoded samples for Quantum Kernel Circuit
    to act as graph weights/biases """

    assert type(sample) is np.ndarray, "sample should be a numpy array"
    Z, ZZ = np.zeros(n_qubits), np.zeros((n_qubits, n_qubits))
    for qubit in range(n_qubits):
        j = 0
        Z[qubit] = int(sample[qubit])
        while j < qubit:
            ZZ[qubit, j] = (np.pi - int(sample[qubit])) * (np.pi - int(sample[j]))
            ZZ[j, qubit] = ZZ[qubit, j]
            j += 1
    encoded_sample = {'Interaction': ZZ, 'Local': Z}
    return encoded_sample


def TwoQubitGate(prog, two_q_arg, qubit_1, qubit_2):
    """  """
    return prog.inst(CPHASE(4 * two_q_arg, qubit_1, qubit_2)).inst(
        PHASE(-2 * two_q_arg, qubit_1)).inst(PHASE(-2 * two_q_arg, qubit_2))


def IQPLayer(prog, qubits, phi_Z, phi_ZZ):
    """todo: some descriptions"""
    n_qubits = len(qubits)
    for j in range(n_qubits):
        # apply local Z rotations (b) to each qubit
        # if the particular qubit sample == 0, apply no gate
        if phi_Z[j]:  # if (phi_Z[j] != False):
            prog.inst(PHASE(-2 * phi_Z[j], qubits[j]))
        # apply control-phase (Phi_ZZ_1) gates to each qubit
        for i in range(n_qubits):
            if i < j:  # if the particular qubit sample pair == 0, apply no gate
                if phi_ZZ[i, j]:
                    prog = TwoQubitGate(prog, phi_ZZ[i, j], qubits[i], qubits[j])
    return prog


def KernelCircuit(qc, sample1, sample2, encode=EncodingFunc):
    """
    Compute Quantum kernel given samples from the Born Machine (born_samples)
    and the Data Distribution (data_samples). This must be done for every sample
    from each distribution (batch gradient descent), (x, y) """

    '''First layer, sample from first distribution (1), parameters phi_ZZ_1, phi_Z_1'''
    qubits = qc.qubits()
    n_qubits = len(qubits)

    prog = Program()
    kernel_circuit_params1 = encode(n_qubits, sample1)
    kernel_circuit_params2 = encode(n_qubits, sample2)

    phi_ZZ_1 = kernel_circuit_params1['Interaction']
    phi_ZZ_2 = kernel_circuit_params2['Interaction']

    phi_Z_1 = kernel_circuit_params1['Local']
    phi_Z_2 = kernel_circuit_params2['Local']

    ###################################################################
    '''First Layer, encoding samples from first distributions, (y)'''
    ###################################################################
    '''First layer of Hadamard'''
    prog = Hadamard2All(prog, qubits)
    '''First IQP layer, encoding sample y'''
    prog = IQPLayer(prog, qubits, phi_Z_1, phi_ZZ_1)

    ###################################################################
    '''Second Layer, encoding samples from both distributions, (x, y)'''
    ###################################################################
    '''Second layer of Hadamard'''
    prog = Hadamard2All(prog, qubits)
    '''Second IQP layer, encoding samples (x, y)'''
    prog = IQPLayer(prog, qubits, phi_Z_1 - phi_Z_2, phi_ZZ_1 - phi_ZZ_2)

    ###################################################################
    '''Third Layer, encoding samples from first distributions, (y)'''
    ###################################################################
    '''Third layer of Hadamard'''
    prog = Hadamard2All(prog, qubits)
    '''Second IQP layer, encoding samples (x, y)'''
    prog = IQPLayer(prog, qubits, -phi_Z_2, -phi_ZZ_2)  # minus sign for complex conjugate

    '''Final layer of Hadamard'''
    prog = Hadamard2All(prog, qubits)

    return prog


def QuantumKernel(qc, n_kernel_samples, sample1, sample2, encode=EncodingFunc):
    """ Computes the Quantum kernel for a single pair of samples """

    # if type(sample1) is np.ndarray and sample1.ndim != 1:
    #     # Check if there is only a single sample in the array of samples
    #     raise IOError('sample1 must be a 1D numpy array')
    # if type(sample2) is np.ndarray and sample2.ndim != 1:
    #     # Check if there is only a single sample in the array of samples
    #     raise IOError('sample2 must be a 1D numpy array')

    qubits = qc.qubits()
    n_qubits = len(qc.qubits())
    make_wf = WavefunctionSimulator()

    # run quantum circuit for a single pair of encoded samples
    prog = KernelCircuit(qc, sample1, sample2, encode)
    kernel_outcomes = make_wf.wavefunction(prog).get_outcome_probs()

    # Create zero string to read off probability
    zero_string = '0' * n_qubits
    kernel_exact = kernel_outcomes[zero_string]

    if n_kernel_samples == 'infinite':
        # If the kernel is computed exactly, approximate kernel is equal to exact kernel
        kernel_approx = kernel_exact
    else:

        # Index list for classical registers we want to put measurement outcomes into.
        # Measure the kernel circuit to compute the kernel approximately,
        # the kernel is the probability of getting (00...000) outcome.
        # All (N_qubits) qubits are measured at once into dictionary, convert into array
        kernel_measurements_all_qubits_dict = qc.run_and_measure(prog, n_kernel_samples)
        kernel_measurements_used_qubits = np.flip(
            np.vstack([kernel_measurements_all_qubits_dict[q] for q in sorted(qubits)]).T, 1)

        # m is total number of samples, n is the number of used qubits
        (m, n) = kernel_measurements_used_qubits.shape

        n_zero_strings = m - np.count_nonzero(np.count_nonzero(kernel_measurements_used_qubits, 1))
        # The kernel is given by
        # [Number of times outcome (00...000) occurred]/[Total number of measurement runs]
        kernel_approx = n_zero_strings / n_kernel_samples

    return kernel_exact, kernel_approx


def QuantumKernelArray(qc, n_kernel_samples, samples1, samples2, encode=EncodingFunc):
    """
    This function computes the quantum kernel for all pairs of samples
    using QuantumKernel(...)
    sample1, samples2: all the bins """

    n_qubits = len(qc.qubits())
    if type(samples1) is np.ndarray:
        n_samples1 = 1 if samples1.ndim == 1 else samples1.shape[0]
        # Check if there is only a single sample in the array of samples
    else:  # List/Tuple
        n_samples1 = len(samples1)

    if type(samples2) is np.ndarray:
        n_samples2 = 1 if samples2.ndim == 1 else samples2.shape[0]
    else:
        n_samples2 = len(samples2)

    # Gram matrix, essentially
    kernel_approx_array = np.zeros((n_samples1, n_samples2))
    kernel_exact_array = np.zeros((n_samples1, n_samples2))
    kernel_approx_dict, kernel_exact_dict = {}, {}

    for s_idx2 in range(n_samples2):
        for s_idx1 in range(s_idx2 + 1):

            kernel_approx_array[s_idx1, s_idx2], \
            kernel_exact_array[s_idx1, s_idx2], = QuantumKernel(
                qc, n_kernel_samples, samples1[s_idx1], samples2[s_idx2], encode)

            # kernel is commutative, k(x,y) = k(y,x)
            kernel_approx_array[s_idx2, s_idx1] = kernel_approx_array[s_idx1, s_idx2]
            kernel_exact_array[s_idx2, s_idx1] = kernel_exact_array[s_idx1, s_idx2]

            ''' save in the dictionary, symmetric matrix '''
            s_temp1 = IntegerToString(s_idx1, n_qubits)
            s_temp2 = IntegerToString(s_idx2, n_qubits)
            kernel_approx_dict[s_temp1, s_temp2] = kernel_approx_array[s_idx1, s_idx2]
            kernel_approx_dict[s_temp2, s_temp1] = kernel_approx_dict[s_temp1, s_temp2]
            kernel_exact_dict[s_temp1, s_temp2] = kernel_exact_array[s_idx1, s_idx2]
            kernel_exact_dict[s_temp2, s_temp1] = kernel_exact_dict[s_temp1, s_temp2]

    return kernel_approx_array, kernel_exact_array, kernel_approx_dict, kernel_exact_dict

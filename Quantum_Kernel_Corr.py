#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import numpy as np, time, auxiliary_functions as aux
from pyquil.api import WavefunctionSimulator
from pyquil.gates import PHASE, CPHASE
from Param_Init import Hadamard2All
from Vis_Utils import PolygonGen
from pyquil.quil import Program
from pyquil.api import get_qc


def bin_pc(nq_x, nq_y, pc):
    """ encode a 2d point cloud coordinates as 1d bit string"""
    x_bins = np.linspace(0, 1 + 1e-8, 2 ** nq_x + 1)
    y_bins = np.linspace(0, 1 + 1e-8, 2 ** nq_y + 1)

    x_digits = np.digitize(pc[:, 0], x_bins, right=False) - 1
    y_digits = np.digitize(pc[:, 1], y_bins, right=False) - 1

    pc_bin = []
    for x_, y_ in zip(x_digits, y_digits):
        x_coords = np.binary_repr(x_, width=nq_x)
        y_coords = np.binary_repr(y_, width=nq_y)
        pc_bin.append(list(map(int, x_coords)) + list(map(int, y_coords)))
    return np.array(pc_bin)


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


def TwoQubitGate(prog, two_q_arg, qubit_1, qubit_2):
    return prog.inst(CPHASE(4 * two_q_arg, qubit_1, qubit_2)).inst(
        PHASE(-2 * two_q_arg, qubit_1)).inst(PHASE(-2 * two_q_arg, qubit_2))


def IQPLayer(prog, qubits, phi_Z, phi_ZZ):
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


def KernelCircuit(qc, sample1, sample2):
    """
    Compute Quantum kernel given samples from the Born Machine (born_samples)
    and the Data Distribution (data_samples). This must be done for every sample
    from each distribution (batch gradient descent), (x, y) """

    '''First layer, sample from first distribution (1), parameters phi_ZZ_1, phi_Z_1'''
    qubits = qc.qubits()
    # n_qubits = len(qubits)
    n_qubits = len(sample1)

    prog = Program()
    kernel_circuit_params1 = EncodingFunc(n_qubits, sample1)
    kernel_circuit_params2 = EncodingFunc(n_qubits, sample2)

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


def Kernel_Quantum_Pair(qc, n_kernel_samples, sample1, sample2):
    """ Computes the Quantum kernel for a single pair of samples
    samples1 and samples2 can be two n-d continuous vectors """

    qubits = qc.qubits()
    n_qubits = len(qc.qubits())
    make_wf = WavefunctionSimulator()

    # run quantum circuit for a single pair of encoded samples
    prog = KernelCircuit(qc, sample1, sample2)
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
        # All (n_qubits) qubits are measured at once into dictionary, convert into array
        kernel_measurements_all_qubits_dict = qc.run_and_measure(prog, n_kernel_samples)
        kernel_measurements_used_qubits = np.flip(
            np.vstack([kernel_measurements_all_qubits_dict[q] for q in sorted(qubits)]).T, 1)

        # m is total number of samples, n is the number of used qubits
        (m, n) = kernel_measurements_used_qubits.shape

        n_zero_strings = m - np.count_nonzero(np.count_nonzero(kernel_measurements_used_qubits, 1))
        # The kernel is given by
        # [Number of times outcome (00...000) occurred]/[Total number of measurement runs]
        kernel_approx = n_zero_strings / n_kernel_samples

    return kernel_approx, kernel_exact


# def KC_Quantum_Set(qc, n_kernel_samples, samples1, samples2):
#     """This function computes the quantum kernel for all pairs of samples"""
#
#     assert len(samples1) == len(samples2)
#     n_samples = len(samples1)
#
#     kernel_approx = np.zeros((n_samples, n_samples))
#     kernel_exact = np.zeros((n_samples, n_samples))
#
#     for s_idx2 in range(n_samples):
#         for s_idx1 in range(s_idx2 + 1):
#             kernel_approx[s_idx1, s_idx2], kernel_exact[s_idx1, s_idx2] = Kernel_Quantum_Pair(
#                 qc, n_kernel_samples, samples1[s_idx1], samples2[s_idx2])
#
#             # kernel is commutative, k(x,y) = k(y,x)
#             kernel_approx[s_idx2, s_idx1] = kernel_approx[s_idx1, s_idx2]
#             kernel_exact[s_idx2, s_idx1] = kernel_exact[s_idx1, s_idx2]
#
#     return kernel_approx, kernel_exact


def KC_Quantum_Set(x, y, rot_mats, qc, n_kernel_samples):
    """ Last term in our MMD loss (rotate the first point set)
    :param x & y: two point sets, in a shape of (num_p, num_dim)
    :param rot_mats: a list of rotation matrices """
    sum_list, (num_p1, num_dim1), (num_p2, num_dim2) = [], x.shape, y.shape
    assert num_dim1 == num_dim2, "two points have different dimensions"
    rot_xs = x @ rot_mats  # (2**n_qubits, num_point, num_dim)
    for rot_x in rot_xs:
        sum = 0
        for xi in rot_x:
            for yi in y:
                approx, exact = Kernel_Quantum_Pair(qc, n_kernel_samples, xi, yi)
                sum += exact
        sum_list.append(sum/(num_p1*num_p2))
    return 2 * np.array(sum_list)
    # return - 2 * np.array(sum_list)


def KC_Quantum_Set_Norm(x, y, rot_mats, qc, n_kernel_samples):
    """ Last term in our MMD loss (rotate the first point set)
    :param x & y: two point sets, in a shape of (num_p, num_dim)
    :param rot_mats: a list of rotation matrices """
    sum_list, (num_p1, num_dim1), (num_p2, num_dim2) = [], x.shape, y.shape
    assert num_dim1 == num_dim2, "two points have different dimensions"
    centroids = x.mean(axis=0)
    rot_xs = (x-centroids) @ rot_mats + centroids
    # (2**n_qubits, num_point, num_dim)
    for rot_x in rot_xs:
        sum = 0
        for xi in rot_x:
            for yi in y:
                approx, exact = Kernel_Quantum_Pair(qc, n_kernel_samples, xi, yi)
                # sum += approx
                sum += exact
        sum_list.append(sum/(num_p1*num_p2))
    return 2 * np.array(sum_list)
    # return - 2 * np.array(sum_list)


def KC_Quantum_Set_Cache(n_qubit, n_side):
    """ Last term in our MMD loss (rotate the first point set) """
    return np.load("qkernel_cache/%dq_side%d.npy" % (n_qubit, n_side))


def KC_Quantum_Set_Bin(x, y, rot_mats, qc, n_kernel_samples, nq_x, nq_y):
    """ Last term in our MMD loss (rotate the first point set)
    :param x & y: two point sets, in a shape of (num_p, num_dim)
    :param rot_mats: a list of rotation matrices """
    sum_list, (num_p1, num_dim1), (num_p2, num_dim2) = [], x.shape, y.shape
    assert num_dim1 == num_dim2, "two points have different dimensions"
    centroids = x.mean(axis=0)
    rot_xs = (x - centroids) @ rot_mats + centroids
    y_binary = bin_pc(nq_x, nq_y, y)

    for rot_x in rot_xs:
        sum = 0
        binary_rot_x = bin_pc(nq_x, nq_y, rot_x)
        for xi in binary_rot_x:
            for yi in y_binary:
                approx, exact = Kernel_Quantum_Pair(qc, n_kernel_samples, xi, yi)
                sum += exact
                # print(sum)
        print(sum / (num_p1 * num_p2))
        sum_list.append(sum / (num_p1 * num_p2))
    return - 2 * np.array(sum_list)


if __name__ == "__main__":
    from main_KC import qubits2bins
    # qc = get_qc('4q-qvm', as_qvm=True)
    # points = PolygonGen(r=1, sideNum=4, pointNum=10)
    # a_angle = np.pi * 3 / 8
    # rot_points = points @ np.array([[np.cos(a_angle), -np.sin(a_angle)],
    #                                 [np.sin(a_angle), np.cos(a_angle)]])
    #
    # # qc_kernel = get_qc('2q-qvm', as_qvm=True)
    # n_qubits = 2
    # born_emps, born_emp_probs = qubits2bins(n_qubits)
    # angle_emps = aux.born2angle(born_emps)
    # rot_emps = aux.rotate_2d(angle_emps)
    # print(angle_emps)
    # print(rot_emps.shape)
    #
    # a_pc = np.random.randn(100, 2)
    # b_pc = np.random.randn(100, 2)
    # kc_cross = KC_Quantum_Set(x=aux.pc_normalize(a_pc, scale=2),
    #                           y=aux.pc_normalize(b_pc, scale=2),
    #                           rot_mats=rot_emps, qc=qc, n_kernel_samples=2000)
    # print(kc_cross)

    # for side_num in range(8):
    for side_num in [8]:
        a_angle = np.pi * 5 / 16
        points = PolygonGen(r=1, sideNum=side_num, pointNum=5)
        rot_points = points @ np.array([[np.cos(a_angle), -np.sin(a_angle)],
                                        [np.sin(a_angle), np.cos(a_angle)]])

        points = aux.pc_normalize(points, scale=2)
        rot_points = aux.pc_normalize(rot_points, scale=2)
        points += [0.5, 0.5]
        rot_points += [0.5, 0.5]

        n_qubits = 4
        born_emps, born_emp_probs = qubits2bins(n_qubits)
        angles = aux.born2angle(born_emps, angle_range=(0, 2 * np.pi))
        rot_emps = aux.rotate_2d(angles)

        qc = get_qc('6q-qvm', as_qvm=True)
        start = time.time()
        kc_tx_y = KC_Quantum_Set_Bin(
            x=points,
            y=rot_points,
            rot_mats=rot_emps,
            qc=qc,
            n_kernel_samples=100,
            nq_x=3,
            nq_y=3)  # exact result from WaveFunction simulation
        print("total time usage: ", str(time.time() - start), "seconds")
        np.save("qkernel_cache/qk_highres_6q_new_side%d.npy" % side_num, kc_tx_y)

    a_angle = np.pi * 5 / 16
    points = np.load("2D_data/fish1.npy")
    rot_points = points @ np.array([[np.cos(a_angle), -np.sin(a_angle)],
                                    [np.sin(a_angle), np.cos(a_angle)]])

    points = aux.pc_normalize(points, scale=2)
    rot_points = aux.pc_normalize(rot_points, scale=2)
    points += [0.5, 0.5]
    rot_points += [0.5, 0.5]

    n_qubits = 4
    born_emps, born_emp_probs = qubits2bins(n_qubits)
    angles = aux.born2angle(born_emps, angle_range=(0, 2 * np.pi))
    rot_emps = aux.rotate_2d(angles)

    qc = get_qc('6q-qvm', as_qvm=True)
    start = time.time()
    kc_tx_y = KC_Quantum_Set_Bin(
        x=points,
        y=rot_points,
        rot_mats=rot_emps,
        qc=qc,
        n_kernel_samples=100,
        nq_x=3,
        nq_y=3)  # exact result from WaveFunction simulation
    print("total time usage: ", str(time.time() - start), "seconds")
    np.save("qkernel_cache/qk_highres_6q_new_fish.npy", kc_tx_y)

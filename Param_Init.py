
#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import random, numpy as np, pyquil.paulis as pl
from numpy import pi
from pyquil.quil import Program
from pyquil.gates import H, CPHASE, PHASE  # , RESET, MEASURE


def Hadamard2All(prog, qubits):
    """ Apply Hadamard gates to all qubits """
    for qubit_index in qubits:
        prog.inst(H(qubit_index))
    return prog


def ParamInit(qc, random_seed, zero_init=True):
    """
    Randomly initialised weights and biases for all qubits
    J, b randomly chosen on interval [0, pi/4]
    gamma, delta and sigma are set to constants (pi/4)

    see "Supervised learning with quantum-enhanced feature spaces", Nature 2019,
    and
        "The Born supremacy: quantum advantage and training of an Ising Born machine", NPJ Quantum 2020,
    for the reasons of this setting """

    n_qubits = len(qc.qubits())
    J = np.zeros((n_qubits, n_qubits))
    [b, gamma, delta, sigma] = [np.zeros(n_qubits) for _ in range(4)]

    # Set random seed to be fixed for reproducibility,
    # set random_seed differently depending on whether quantum data
    # is generated, or whether the actual Born machine is being used.
    random.seed(random_seed)
    for j in range(n_qubits):
        if zero_init:
            b[j] = 0
        else:
            b[j] = random.uniform(0, pi/4)

        '''
        If delta to be trained also and variable for each qubit
            gamma[j] = rand.uniform(0,pi/4)
        If delta to be trained also and variable for each qubit
            delta[j] = uniform(0,pi/4) '''

        # constants for all qubits
        gamma[j] = pi / 4
        delta[j] = pi / 4
        sigma[j] = pi / 4

        if not zero_init:
            for i in range(n_qubits):
                if i < j:
                    J[i][j] = random.uniform(0, pi/4)
                    J[j][i] = J[i][j]

    initial_params = {'J': J, 'b': b,
                      'gamma': gamma,
                      'delta': delta,
                      'sigma': sigma}

    return initial_params


# def NetworkParamsSingleQubitGates(qc, layers):
#     """ Initialises single-qubit trainable parameters """
#
#     n_qubits = len(qc.qubits())
#
#     # Initialise arrays for parameters
#     single_qubit_params = np.zeros(n_qubits, n_qubits, n_qubits, layers)
#     # layers is the number of single qubit layers,
#     # each 'layer', l, consists of three gates,
#     # R_z(\theta_l^1)R_x(\theta_l^2)R_x(\theta_l^3)
#
#     # Set random seed to be fixed for reproducibility
#     random.seed(0)
#     for j in range(n_qubits):
#         for l in range(layers):
#             # initialise all single qubit gates at random
#             single_qubit_params[j, :, :, l] = random.uniform(0, pi / 4)
#             single_qubit_params[:, j, :, l] = random.uniform(0, pi / 4)
#             single_qubit_params[:, :, j, l] = random.uniform(0, pi / 4)
#
#     return single_qubit_params


def StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, sign):
    """
    Initialise Quantum State created after application of gate sequence, then
    computes the state produced after the given circuit, either QAOA, IQP or IQPy """

    # sign = 'POSITIVE' for the positive probability version,
    # sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
    # final_layer is either 'IQP', 'QAOA', 'IQPy'
    #   for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
    # control = 'BIAS' for updating biases,
    # control = 'WEIGHTS' for updating weights,
    # control = 'NEITHER' for neither

    # Initialise an empty quantum program,
    # with QuantumComputer and WavefunctionSimulator objects
    '''with active qubit reset'''
    # prog = Program(RESET())
    '''without active qubit reset'''
    prog = Program()
    qubits = qc.qubits()
    n_qubits = len(qubits)

    # Unpack circuit parameters at current epochs
    J, b = circuit_params['J'], circuit_params['b']
    gamma, delta = circuit_params['gamma'], circuit_params['delta']

    # Apply Hadamard to all qubits in computation
    prog = Hadamard2All(prog, qubits)

    # Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
    # with local Z corrections to neighbouring qubits, coming from the decomposition of the Ising gate
    # If weight J_{p,q} is updated, add a +/- pi/2 rotation
    if qc.name.lower() == 'aspen-3-3q-b-qvm':
        ''''Specific entanglement structure for Rigetti Aspen-3-2Q-C'''
        if control == 'WEIGHTS' and p == 0 and q == 1:
            # first weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[0, 1] + (-1) ** sign * pi / 2, qubits[0], qubits[1]))
            prog.inst(PHASE(-2 * J[0, 1] + (-1) ** sign * pi / 2, qubits[0]))
            prog.inst(PHASE(-2 * J[0, 1] + (-1) ** sign * pi / 2, qubits[1]))

        elif control == 'WEIGHTS' and p == 1 and q == 2:
            # second weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[1, 2] + (-1) ** sign * pi / 2, qubits[1], qubits[2]))
            prog.inst(PHASE(-2 * J[1, 2] + (-1) ** sign * pi / 2, qubits[1]))
            prog.inst(PHASE(-2 * J[1, 2] + (-1) ** sign * pi / 2, qubits[2]))

        elif (control == 'NEITHER' or 'BIAS' or 'GAMMA') and sign == 'NEITHER':
            prog.inst(CPHASE(4 * J[0, 1], qubits[0], qubits[1]))
            prog.inst(PHASE(-2 * J[0, 1], qubits[0]))
            prog.inst(PHASE(-2 * J[0, 1], qubits[1]))

            prog.inst(CPHASE(4 * J[1, 2], qubits[1], qubits[2]))
            prog.inst(PHASE(-2 * J[1, 2], qubits[1]))
            prog.inst(PHASE(-2 * J[1, 2], qubits[2]))

    elif qc.name.lower() == 'aspen-4-3q-a' or qc.name.lower() == 'aspen-4-3q-a-qvm':

        '''
        Specific entanglement structure for Rigetti Aspen-4-3Q-A
        17 - 10 - 11
        '''
        if control == 'WEIGHTS' and p == 0 and q == 1:
            # first weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[0, 1] + (-1) ** sign * pi / 2, qubits[0], qubits[1]))
            prog.inst(PHASE(-2 * J[0, 1] + (-1) ** sign * pi / 2, qubits[0]))
            prog.inst(PHASE(-2 * J[0, 1] + (-1) ** sign * pi / 2, qubits[1]))

        elif control == 'WEIGHTS' and p == 1 and q == 2:
            # second weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[0, 2] + (-1) ** sign * pi / 2, qubits[0], qubits[2]))
            prog.inst(PHASE(-2 * J[0, 2] + (-1) ** sign * pi / 2, qubits[0]))
            prog.inst(PHASE(-2 * J[0, 2] + (-1) ** sign * pi / 2, qubits[2]))

        elif (control == 'NEITHER' or 'BIAS' or 'GAMMA') and sign == 'NEITHER':
            prog.inst(CPHASE(4 * J[0, 1], qubits[0], qubits[1]))
            prog.inst(PHASE(-2 * J[0, 1], qubits[0]))
            prog.inst(PHASE(-2 * J[0, 1], qubits[1]))

            prog.inst(CPHASE(4 * J[0, 2], qubits[0], qubits[2]))
            prog.inst(PHASE(-2 * J[0, 2], qubits[0]))
            prog.inst(PHASE(-2 * J[0, 2], qubits[2]))

    elif qc.name.lower() == 'aspen-4-4q-a' or qc.name.lower() == 'aspen-4-4q-a-qvm':
        ''''
        Specific entanglement structure for Rigetti Aspen-4-4Q-A 
        7 - 0 - 1 - 2'''
        if control.lower() == 'weights' and p == 0 and q == 1:
            # first weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[0, 1] + (-1) ** sign * pi / 2, qubits[0], qubits[1]))
            prog.inst(PHASE(-2 * J[0, 1] + (-1) ** sign * pi / 2, qubits[0]))
            prog.inst(PHASE(-2 * J[0, 1] + (-1) ** sign * pi / 2, qubits[1]))

        elif control.lower() == 'weights' and p == 1 and q == 2:
            # second weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[1, 2] + (-1) ** sign * pi / 2, qubits[1], qubits[2]))
            prog.inst(PHASE(-2 * J[1, 2] + (-1) ** sign * pi / 2, qubits[1]))
            prog.inst(PHASE(-2 * J[1, 2] + (-1) ** sign * pi / 2, qubits[2]))

        elif control.lower() == 'weights' and p == 0 and q == 3:
            # second weight parameter between qubit[1] and qubit[2]
            prog.inst(CPHASE(4 * J[0, 3] + (-1) ** sign * pi / 2, qubits[0], qubits[3]))
            prog.inst(PHASE(-2 * J[0, 3] + (-1) ** sign * pi / 2, qubits[0]))
            prog.inst(PHASE(-2 * J[0, 3] + (-1) ** sign * pi / 2, qubits[3]))

        elif (control.lower() == 'neither' or 'bias' or 'gamma') and sign.lower() == 'neither':
            prog.inst(CPHASE(4 * J[0, 1], qubits[0], qubits[1]))
            prog.inst(PHASE(-2 * J[0, 1], qubits[0]))
            prog.inst(PHASE(-2 * J[0, 1], qubits[1]))

            prog.inst(CPHASE(4 * J[1, 2], qubits[1], qubits[2]))
            prog.inst(PHASE(-2 * J[1, 2], qubits[1]))
            prog.inst(PHASE(-2 * J[1, 2], qubits[2]))

            prog.inst(CPHASE(4 * J[0, 3], qubits[0], qubits[3]))
            prog.inst(PHASE(-2 * J[0, 3], qubits[0]))
            prog.inst(PHASE(-2 * J[0, 3], qubits[3]))

    else:
        for j in range(n_qubits):
            for i in range(n_qubits):
                if i < j:  # connection is symmetric, to prevent over-counting entangling gates
                    if control == 'WEIGHTS' and i == p and j == q:
                        prog.inst(CPHASE(4 * J[i, j] + (-1) ** sign * pi / 2, qubits[i], qubits[j]))
                        prog.inst(PHASE(-2 * J[i, j] + (-1) ** sign * pi / 2, qubits[i]))
                        prog.inst(PHASE(-2 * J[i, j] + (-1) ** sign * pi / 2, qubits[j]))

                    # elif (control.lower() == 'neither' or 'bias' or 'gamma' and sign.lower() == 'neither'):
                    else:
                        prog.inst(CPHASE(4 * J[i, j], qubits[i], qubits[j]))
                        prog.inst(PHASE(-2 * J[i, j], qubits[i]))
                        prog.inst(PHASE(-2 * J[i, j], qubits[j]))

    # Apply local Z rotations (b) to each qubit
    # (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
    for j in range(n_qubits):
        if control == 'BIAS' and j == r:
            prog.inst(PHASE(-2 * b[j] + (-1) ** sign * pi / 2, qubits[j]))
        # elif (control == 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
        else:
            prog.inst(PHASE(-2 * b[j], qubits[j]))

    # Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
    if circuit_choice == 'IQP':
        prog = Hadamard2All(prog, qubits)
        # If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)
    elif circuit_choice == 'QAOA':
        # If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
        for k in range(n_qubits):
            # if (control == 'GAMMA' and k == s):
            # 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])+ (-1)**(sign)*pi/2))

            # elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
            H_temp = (-float(gamma[k])) * pl.sX(qubits[k])
            prog.inst(pl.exponential_map(H_temp)(1.0))
        # print('GAMMA IS:',-float(gamma[k]))
    elif circuit_choice == 'IQPy':
        # If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
        for k in qubits:
            H_temp = (-float(delta[k])) * pl.sY(qubits[k])
            prog.inst(pl.exponential_map(H_temp)(1.0))
    else:
        raise NotImplementedError

    '''Insert explicit measure instruction if required'''
    # ro = prog.declare('ro', 'BIT', len(qubits))
    # prog.inst([MEASURE(qubit, ro[idx]) for idx, qubit in enumerate(qubits)])

    return prog


if __name__ == '__main__':

    class IsingBornMachine:

        """ Not Used """

        def __init__(self, qc, circuit_params, meas_choice):
            self.circuit = Program()
            self.qubits = qc.qubits()

            self._num_qubits = len(self.qubits)
            self._meas_choice = meas_choice

        def _params(self, circuit_params):
            # Unpack circuit parameters from dictionary
            self.J = circuit_params['J']
            self.b = circuit_params['b']
            self.gamma = circuit_params['gamma']
            self.delta = circuit_params['delta']

        def _hadamard_to_all(self):
            """Adds Hadamard to all qubits in qubit list"""
            self.circuit = self.circuit + [H(qubit_index) for qubit_index in self.qubits]
            return

        # for j in range(0, N_qubits):
        # 		for i in range(0, N_qubits):
        # 				if (i < j): #connection is symmetric, so don't overcount entangling gates
        # 					if (control.lower() == 'weights' and i == p and j == q):
        # 						prog.inst(CPHASE(4*J[i, j] + (-1)**(sign)*pi/2, qubits[i], qubits[j]))
        # 						prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[i]))
        # 						prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[j]))

        # 					elif (control.lower() == 'neither' or 'bias' or 'gamma' and sign.lower() == 'neither'):
        # 						prog.inst(CPHASE(4*J[i, j], qubits[i], qubits[j]))
        # 						prog.inst(PHASE(-2*J[i, j], qubits[i]))
        # 						prog.inst(PHASE(-2*J[i, j], qubits[j]))

        def _measurement_layer(self, meas_choice):
            self._meas_choice = meas_choice
            # Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
            if self._meas_choice.lower() == 'iqp':
                self.circuit = self.circuit + [H(qubit_index) for qubit_index in self.qubits]

            # elif (circuit_choice =='QAOA'):
        # 	#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
        # 	for k in range(0, N_qubits):
        # 		# if (control == 'GAMMA' and k == s):
        # 		# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])+ (-1)**(sign)*pi/2))

        # 		# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
        # 		H_temp = (-float(gamma[k]))*pl.sX(qubits[k])
        # 		self.circuit += pl.exponential_map(H_temp)(1.0)
        # 		# print('GAMMA IS:',-float(gamma[k]))
        # elif (circuit_choice == 'IQPy' ):
        # 	#If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
        # 	for k in qubits:
        # 		H_temp = (-float(delta[k]))*pl.sY(qubits[k])
        # 		prog.inst(pl.exponential_map(H_temp)(1.0))

    # device_name = '2q-qvm'
    # as_qvm_value = True
    # qc = get_qc(device_name, as_qvm = as_qvm_value)

    # params = NetworkParams(qc, 123342)
    # ibm = IsingBornMachine(qc, params)
    # ibm._hadamard_to_all()
    # print(ibm.circuit)
    # ibm._measurement_layer( 'IQP')
    # print(ibm.circuit)

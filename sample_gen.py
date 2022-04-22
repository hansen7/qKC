#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import numpy as np
from Param_Init import StateInit
from pyquil.api import WavefunctionSimulator
from auxiliary_functions import EmpiricalDist


def BornSampler(qc, n_born_samples, circuit_params, circuit_choice):
    """
    Generate samples from the output distribution of the IQP/QAOA/IQPy
    circuit according to the Born Rule:
        P(z) = |<z|U|s>|^2
    where |s> is the uniform superposition (|+>) """
    n_qubits = len(qc.qubits())
    make_wf = WavefunctionSimulator()
    prog = StateInit(qc, circuit_params, 0, 0, 0, 0,
                     circuit_choice, 'NEITHER', 'NEITHER')
    # StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, sign)
    # circuit_choice = ('IQP' for IQP), = ('QAOA' for QAOA), = ('IQPy' for Y-Rot)
    # control = 'BIAS' for updating biases,
    # 		  = 'WEIGHTS' for updating weights,
    # 		  = 'GAMMA' for gamma params,
    # 		  = 'NEITHER' for neither
    # sign  = 'POSITIVE' to run the positive circuit,
    # 		= 'NEGATIVE' for the negative circuit,
    #		= 'NEITHER' for neither

    '''Generate # of `n_born_samples` data points from 
    the output distribution on (n_qubits) circuits '''
    # measure all qubit status after simulating n_born_samples times
    # {qubit_0: binary numpy array, qubit_0: another binary numpy array, ...}
    all_qubits_dict = qc.run_and_measure(prog, n_born_samples)

    # numpy array: (n_born_samples, n_qubits), still simulated output
    born_samples = np.flip(np.vstack([all_qubits_dict[q] for q in range(n_qubits)]).T, 1)

    # e.g., {'00': freq, '01': freq, ...}
    born_probs_approx_dict = EmpiricalDist(born_samples, n_qubits, 'full_dist')

    # compute exact distribution of the output
    born_probs_exact_dict = make_wf.wavefunction(prog).get_outcome_probs()

    return born_samples, born_probs_approx_dict, born_probs_exact_dict


def PlusMinusSampleGen(qc, circuit_params, p, q, r, s, circuit_choice, control, batch_size):
    """ computes the samples required in the estimator,
    in the +/- terms of the MMD loss function gradient w.r.t parameter,
    J_{p, q} (control = 'WEIGHTS') , b_r (control = 'BIAS') or gamma (control == 'GAMMA') """

    n_qubits = len(qc.qubits())
    # probs_minus, probs_plus are the exact probabilities output from the circuit
    prog_plus = StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, 0)
    prog_minus = StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, 1)

    born_samples_pm = []
    # generate batch_size samples from measurements of +/- shifted circuits
    born_samples_plus_all_qbs_dict = qc.run_and_measure(prog_plus, batch_size)
    born_samples_minus_all_qbs_dict = qc.run_and_measure(prog_minus, batch_size)
    # put outcomes into a list of arrays
    born_samples_pm.append(np.flip(np.vstack(
        [born_samples_plus_all_qbs_dict[q] for q in range(n_qubits)]).T, 1))
    born_samples_pm.append(np.flip(np.vstack(
        [born_samples_minus_all_qbs_dict[q] for q in range(n_qubits)]).T, 1))

    return born_samples_pm

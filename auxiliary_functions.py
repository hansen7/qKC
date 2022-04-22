#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import pdb, h5py, torch, numpy as np, matplotlib.pyplot as plt
from collections import Counter


def AllBinaryStrings(n_qubits):
    """ Generate array of all binary strings of length n_qubits """
    binary_strings_array = np.zeros((2 ** n_qubits, n_qubits))
    for integer in range(0, 2 ** n_qubits):
        qubit_string = IntegerToString(integer, n_qubits)
        for qubit in range(0, n_qubits):
            binary_strings_array[integer][qubit] = float(qubit_string[qubit])
    return binary_strings_array


def IntegerToString(integer, n_qubits):
    """ converts a integer to a binary string """
    assert type(integer) is int and type(n_qubits) is int
    return "0" * (n_qubits - len(format(integer, 'b'))) + format(integer, 'b')


def StringToList(string):
    """ Convert a binary string to a list of bits """
    if type(string) is not str:
        raise TypeError('\'string\' must be a str')
    string_list = []
    for element in range(len(string)):
        string_list.append(int(string[element]))
    return string_list


def ShiftString(string, shift_index):
    """ This function shifts the (shift_index)th element of a string by 1 (mod 2).
        This is the shift operator -> on a binary space. Returns a binary 1D array """
    # todo: prune
    if type(string) is np.ndarray:
        if string.ndim != 1:
            raise IOError('If \'string\' is a numpy array, it must be one dimensional.')
        shifted_string = string

        for i in range(len(string)):
            if i is shift_index:
                shifted_string[i] = (string[i] + 1) % 2
    elif type(string) is str:
        string_list = StringToList(string)
        shifted_string = np.ndarray((len(string_list)), dtype=int)
        for i in range(len(string_list)):
            if i is shift_index:
                shifted_string[i] = (string_list[i] + 1) % 2
            else:
                shifted_string[i] = string_list[i]
    else:
        raise ValueError("input string type is not supported")
    return shifted_string


def ToStr(input_object):
    """ converts an input (integer, list, numpy array) to string """
    # todo: prune
    print("ToStr(), input object type is ", type(input_object))
    if type(input_object) is np.ndarray:
        if input_object.ndim != 1:
            pdb.set_trace()
            raise IOError('If \'input\' is numpy array it must be 1D')
        else:
            input_as_string = ''.join([str(bit) for bit in list(input_object)])
    elif type(input_object) is list:
        pdb.set_trace()
        input_as_string = ''.join([str(bit) for bit in input_object])
    elif type(input_object) is str:
        pdb.set_trace()
        input_as_string = input_object
    elif type(input_object) is int:
        pdb.set_trace()
        input_as_string = bin(input_object)[2:]
        # bin() -> return the binary representation of an integer.
    else:
        raise ValueError("input_object type is not supported")
    return input_as_string


def StringToArray(string):
    """ breaks a string into a np.array """
    string_array = np.zeros((len(string)), dtype=int)
    for bit in range(len(string)):
        string_array[bit] = int(string[bit])
    return string_array


def SampleListToArray(samples_liststr, n_qubits, array_type):
    """ converts a list of strings, into a numpy array (float/int), where
    the (i,j)-th entry of the new array is the jth letter of the ith string """

    n_samples = len(samples_liststr)
    assert (array_type == 'float') or (array_type == 'int')
    converter = lambda x: int(x) if array_type == 'int' else float(x)
    sample_array = np.zeros((n_samples, n_qubits), dtype=array_type)

    for sample in range(n_samples):
        for outcome in range(n_qubits):
            sample_array[sample][outcome] = converter(samples_liststr[sample][outcome])
    return sample_array


def SampleArrayToList(sample_array):
    """ converts a np.array where rows are samples
    into a list of length N_samples"""
    # if number of samples in array is just one, handle separately
    if sample_array.ndim == 1:
        sample_list = [''.join(str(e) for e in (sample_array.tolist()))]
    else:
        sample_list = []
        n_samples = sample_array.shape[0]
        for sample in range(n_samples):
            sample_list.append(''.join(str(int(e)) for e in (sample_array[sample][:].tolist())))
    return sample_list


def EmpiricalDist(samples, n_qubits, *arg):
    """ outputs the empirical probability distribution given samples in a numpy array
    as a dictionary, with keys as outcomes, and values as probabilities """
    if (type(samples) is not np.ndarray) and (type(samples) is not list):
        raise TypeError('samples must be either a numpy array, or a list')

    if type(samples) is list:
        samples = np.array(samples)

    # Convert numpy array of samples to a list of strings
    # i.e., [[1, 1], [1, 0], ...] -> ['11', '10', ...]
    string_list = list()
    n_samples = samples.shape[0]
    for sample in range(n_samples):
        string_list.append(''.join(map(str, samples[sample].tolist())))

    # Convert co-occurrences to relative frequencies
    counts = Counter(string_list)
    for element in counts:
        counts[element] /= n_samples
    if 'full_dist' in arg:
        for index in range(2 ** n_qubits):
            # If a binary string has not appeared, set its frequency to zero
            if IntegerToString(index, n_qubits) not in counts:
                counts[IntegerToString(index, n_qubits)] = 0.

    # e.g., {'00': freq, '01': freq, ...}
    sorted_samples_dict = dict()
    for key in sorted(counts):
        sorted_samples_dict[key] = counts[key]

    return sorted_samples_dict


def ExtractSampleInfo(samples):
    """ Convert an array of samples as the empirical distribution,
        and extracts empirical probabilities and corresponding sample values,
        and convert to PyTorch tensors """
    if type(samples) is np.ndarray:
        # in fact, we merely use single qubit circuit
        n_qubits = len(samples) if samples.ndim == 1 else len(samples[0])
    else:
        n_qubits = len(samples[0])

    # e.g., {'00': freq00, '01': freq01, ...}
    emp_dist_dict = EmpiricalDist(samples, n_qubits)
    # i.e., [[0., 0.], [0., 1.], ...]
    samples = SampleListToArray(list(emp_dist_dict.keys()), n_qubits, 'float')
    # i.e., [[0, 0], [0, 1], ...]
    samples_int = SampleListToArray(list(emp_dist_dict.keys()), n_qubits, 'int')

    # i.e., [freq00, freq01, ...]
    probs = np.asarray(list(emp_dist_dict.values()))

    # convert to torch tensors if necessary
    # pylint: disable=E1101
    samples_tensor = torch.from_numpy(samples).view(len(samples), -1)
    probs_tensor = torch.from_numpy(probs).view(len(probs), -1)
    # pylint: enable=E1101

    return samples_int, probs, samples_tensor, probs_tensor


def ConvertStringToVector(string):
    """ converts a string to a np.array """
    string_vector = np.zeros(len(string), dtype=int)
    for bit in range(len(string)):
        # if string[bit] == '0' or string[bit] == '1':
        string_vector[bit] = int(string[bit])
        # else:
        #     raise IOError('Please enter a binary string')
    return string_vector


def L2Norm(input1, input2):
    """computes the squared L2 norm between two binary vectors"""
    # todo: prune
    if (type(input1) is str) and (type(input2) is str):
        pdb.set_trace()
        l2norm = (np.linalg.norm(np.abs(ConvertStringToVector(input1)
                                        - ConvertStringToVector(input2)), 2)) ** 2
    elif (type(input1) is np.ndarray) and (type(input2) is np.ndarray):
        l2norm = (np.linalg.norm(np.abs(input1 - input2), 2)) ** 2
    else:
        raise IOError('The inputs must be 1D numpy arrays, or strings')
    return l2norm


def TrainTestPartition(samples):
    # todo: maybe try instead sklearn.model_selection.train_test_split()
    return np.split(samples, [round(len(samples) * 0.8), ], axis=0)


def MiniBatchSplit(samples, batch_size):
    """ takes the first mini-batch out of the full sample set """
    assert type(samples) is np.ndarray, 'input must be a np.ndarray'
    return np.split(samples, [batch_size, len(samples)], axis=0)[0]


def num_bytes_needed(num_bits):
    num_bytes = num_bits // 8
    if num_bits % 8 != 0:
        num_bytes += 1
    return num_bytes


def rotate_point_cloud_z(batch_data, small=True, angle=None):
    """ Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        if angle is None:
            rotation_angle = np.random.uniform() * 2 * np.pi
            if small:
                rotation_angle *= 0.01
        else:
            rotation_angle = angle

        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud. """
    #     B, N, C = batch_data.shape
    #     B = batch_data.shape[0]
    shifts = np.random.uniform(-shift_range, shift_range, (3,))
    #     for batch_index in range(B):
    #     batch_data[batch_index, :, :] += shifts[batch_index, :]
    batch_data += shifts
    return batch_data


def loadh5DataFile(PathtoFile):
    f = h5py.File(PathtoFile, 'r')
    return f['data'][:], f['label'][:]


def pc_normalize(pc, scale=1):
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / (m * scale)


def plot_pcd_three_views(pcds, sizes=None, cmap='viridis', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for _ in range(len(pcds))]
    fig = plt.figure(figsize=(9, len(pcds) * 3))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(len(pcds), 3, j * len(pcds) + i + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                       zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.show(fig)


def born2angle(born_emps, angle_range=(0, 2 * np.pi)):
    """
    :param born_emps: possible binary samples from circuits, e.g,
                        np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    :param angle_range: the range of the corresponding \theta,
                        default as (0, 2*np.pi)
    :return: the interval values of each bins, (n_bins, )
    """
    num_bins = len(born_emps)
    half_step = (angle_range[1] - angle_range[0]) / (2 * num_bins)
    val_bins = [half_step * num for num in range(1, 2 * num_bins, 2)]
    return np.array(val_bins)


def rotate_2d(angle):
    if type(angle) is not np.ndarray:  # float, return (2, 2)
        cosval, sinval = np.cos(angle), np.sin(angle)
        return np.array([[cosval, -sinval], [sinval, cosval]])
    else:  # a list of angles, return (n_bins, 2, 2)
        return np.concatenate([rotate_2d(ang) for ang in angle]).reshape((-1, 2, 2))


def rotate_3d_z(angle):
    if type(angle) is not np.ndarray:  # float, return (2, 2)
        cosval, sinval = np.cos(angle), np.sin(angle)
        return np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0., 0., 1]])
    else:  # a list of angles, return (n_bins, 2, 2)
        return np.concatenate([rotate_3d_z(ang) for ang in angle]).reshape((-1, 3, 3))


def rotate_3d_x(angle):
    if type(angle) is not np.ndarray:  # float, return (2, 2)
        cosval, sinval = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                         [0, cosval, -sinval],
                         [0, sinval, cosval]])
    else:  # a list of angles, return (n_bins, 2, 2)
        return np.concatenate([rotate_3d_x(ang) for ang in angle]).reshape((-1, 3, 3))


def rotate_3d_y(angle):
    if type(angle) is not np.ndarray:  # float, return (2, 2)
        cosval, sinval = np.cos(angle), np.sin(angle)
        return np.array([[cosval, 0, -sinval],
                         [0, 1, 0],
                         [sinval, 0, cosval]])
    else:  # a list of angles, return (n_bins, 2, 2)
        return np.concatenate([rotate_3d_y(ang) for ang in angle]).reshape((-1, 3, 3))

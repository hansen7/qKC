#  Copyright (c) 2021. Hanchen Wang, hc.wang96@gmail.com

import numpy as np, auxiliary_functions as aux


def KC_Gaussian_Pair(xi, xj, sigmas):
    """
    KC_G() defined in the paper
    :param xi: a point, (1, num_dim)
    :param xj: a point or a list of points, (1, num_dim) or (num_point, num_dim)
    :param sigmas: a list of Gaussian variances """

    sum, (_, num_dim) = 0, xi.shape
    for sigma in sigmas:
        # sum += np.exp(-(((xj - xi) ** 2).sum(axis=1) / (2 * sigma ** 2))).sum() * \
        #        ((2 * np.pi * sigma ** 2) ** (-num_dim / 2))
        sum += np.exp(-(((xj - xi) ** 2).sum(axis=1) / (2 * sigma))).sum()
    return sum


def KC_Gaussian_LOO(x, sigmas, rot_mats=None):
    """
    Leave-One-Out Kernel Correlation, first two terms in our MMD loss
    :param x: a point set, in a shape of (num_p, num_dim)
    :param sigmas: a list of Gaussian kernel variances
    :param rot_mats: a list of 2D/3D rotation matrices """

    '''assume there are not overlapping points (duplicates)'''
    sum, (num_p, num_dim) = 0, x.shape

    if rot_mats is None:
        for idx, xi in enumerate(x):
            sum += KC_Gaussian_Pair(xi.reshape(1, num_dim),
                                    np.delete(x, idx, axis=0),
                                    sigmas)  # x[x != xi].reshape(-1, num_dim)
        return sum/(num_p*(num_p-1))

    else:
        num_bins, num_dim2, _ = rot_mats.shape
        assert num_dim == num_dim2, "points and rotations have different dimensions"
        # essentially they are identical before weighted by the probabilities
        # return np.ones(num_bins) * KC_Gaussian_LOO(x, sigmas)
        # (num_bins, num_p, num_dim) = (num_p, num_dim) @ (num_bins, num_dim, num_dim)
        rot_xs = x @ rot_mats
        sum_list = []
        for rot_x in rot_xs:
            sum_list.append(KC_Gaussian_LOO(rot_x, sigmas))
        return np.array(sum_list)


def KC_Gaussian_Set(x, y, sigmas, rot_mats):
    """ Last term in our MMD loss (rotate the first point set)
    :param x & y: two point sets, in a shape of (num_p, num_dim)
    :param sigmas: a list of Gaussian kernel variances
    :param rot_mats: a list of rotation matrices """
    sum_list, (num_p1, num_dim1), (num_p2, num_dim2) = [], x.shape, y.shape
    assert num_dim1 == num_dim2, "two points have different dimensions"
    rot_xs = x @ rot_mats
    for rot_x in rot_xs:
        sum = 0
        for xi in rot_x:
            sum += KC_Gaussian_Pair(xi.reshape(1, num_dim1), y, sigmas)
        sum_list.append(sum/(num_p1*num_p2))
    return - 2 * np.array(sum_list)


def KC_Gaussian_Set_Dual(x, y, sigmas, rot_mats1, rot_mats2):
    """ First two terms in our MMD gradients, not used
    :param x & y: two point sets, in a shape of (num_p, num_dim)
    :param sigmas: a list of Gaussian kernel variances
    :param rot_mats1 & rot_mats2: two lists of rotation matrices"""
    sum_list, (num_p1, num_dim1), (num_p2, num_dim2) = [], x.shape, y.shape
    num_rotx, num_roty = len(rot_mats1), len(rot_mats2)
    assert num_dim1 == num_dim2, "two points have different dimensions"
    rot_xs = x @ rot_mats1
    rot_ys = y @ rot_mats2
    for rot_y in rot_ys:
        for rot_x in rot_xs:
            sum = 0
            # for xi in rot_x:
            #     sum += KC_Gaussian_Pair(xi.reshape(1, num_dim1), rot_y, sigmas)
            for xi, yi in zip(rot_x, rot_y):
                sum += KC_Gaussian_Pair(xi.reshape(1, num_dim1),
                                        yi.reshape(1, num_dim1), sigmas)
            # sum_list.append(sum/(num_p1*num_p2))
            sum_list.append(sum / num_p1)

    return - 2 * np.array(sum_list).reshape(num_rotx, num_roty)


if __name__ == "__main__":
    born_emps = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    angle_emps = aux.born2angle(born_emps, angle_range=(0, 2 * np.pi))
    print(angle_emps)
    rot_emps = aux.rotate_2d(angle_emps)
    print(rot_emps.shape)

    a_pc = np.random.randn(100, 2)
    kc_loo = KC_Gaussian_LOO(x=a_pc, sigmas=[.1, 1, 10])
    print(kc_loo)
    kc_loos = KC_Gaussian_LOO(x=a_pc, sigmas=[.1, 1, 10], rot_mats=rot_emps)
    print(kc_loos)

    b_pc = np.random.randn(100, 2)
    kc_cross = KC_Gaussian_Set(x=a_pc, y=b_pc, sigmas=[.1, 1, 10], rot_mats=rot_emps)
    print(kc_cross)

    kc_dual = KC_Gaussian_Set_Dual(x=a_pc, y=b_pc, sigmas=[.1, 1, 10],
                                   rot_mats1=rot_emps, rot_mats2=rot_emps)
    print(kc_dual)

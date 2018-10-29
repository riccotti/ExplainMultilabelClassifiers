import numpy as np
import pandas as pd

#definisco qui la distanza:
def normalized_euclidean_distance(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))

def simple_match_distance(x, y):
    count = 0
    for xi, yi in zip(x, y):
        if xi == yi:
            count += 1
    sim_ratio = 1.0 * count / len(x)
    return 1.0 - sim_ratio


def normalized_square_euclidean_distance(ranges):
    def actual(x, y, xy_ranges):
        return np.sum(np.square(np.abs(x - y) / xy_ranges))
    return lambda x, y: actual(x, y, ranges)


def mad_distance(x, y, mad):
    val = 0.0
    for i in range(len(mad)):
        # print i, 0.0 if mad[i] == 0.0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
        # print i, np.abs(x[i] - y[i]) / mad[i]
        # val += 0.0 if mad[i] != 0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
        val += 0.0 if mad[i] == 0.0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
    # print val
    return val


def mixed_distance(x, y, discrete, continuous, classes_name, ddist, cdist):
    # type: (pandas.Series, pandas.Series, list, list, list, function, function) -> double
    """
    This function return the mixed distance between instance x and instance y
    :param x: pandas.Series, instance 1
    :param y: pandas.Series, instance 2
    :param discrete: list of str, column names containing categorical variables
    :param continuous: list of str, column names containing non categorical variables
    :param classes_name: list of str, array of column names containing the label
    :param ddist: function, distance function for discrete variables
    :param cdist: function, distance function for continuos variables
    :return: double
    """
    xd = [x[att] for att in discrete if att not in classes_name]
    wd = 0.0
    dd = 0.0
    if len(xd) > 0:
        yd = [y[att] for att in discrete if att not in classes_name]
        wd = 1.0 * len(discrete) / (len(discrete) + len(continuous))
        dd = ddist(xd, yd)

    xc = np.array([x[att] for att in continuous if att not in classes_name])
    wc = 0.0
    cd = 0.0
    if len(xc) > 0:
        yc = np.array([y[att] for att in continuous if att not in classes_name])
        wc = 1.0 * len(continuous) / (len(discrete) + len(continuous))
        cd = cdist(xc, yc)

    return wd * dd + wc * cd


def sorted_distances_df(X2E, i2e, discrete_var, continuous_var, classes_name,label_distance='distance'):
    """
    This function returns the neighours of the instance sorted by closeness,
        the distance metric used is `mixed_distance()`

    :param X2E: dataframe, each row is an instance and the label was given by the black box, should NOT contain column(s) with labels
    :param i2e: pd.Series, instance to be explained
    :param discrete_var: array of str, names of X2E columns containing discrete features
    :param continuous_var: array of str, names of X2E columns containing continuous features
    :param class_name: array of str, name(s) of the column(s) containing the label
    :return: pandas dataframe
    """
    indexes_values = X2E.index.values
    # distance between instance to explain and other instances
    distances = [mixed_distance(i2e,X2E.loc[i],discrete=discrete_var,continuous=continuous_var,classes_name=classes_name,ddist=simple_match_distance,cdist=normalized_euclidean_distance) for i in indexes_values]
    output = X2E.reset_index().rename(columns={'index':'old_index_'+label_distance})#.drop('index',1)
    output[label_distance] = pd.Series(distances)
    output = output.sort_values(by=label_distance,ascending=True).reset_index().drop('index',1)
 
    return output

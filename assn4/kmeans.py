import os
import math
from utils import converged, plot_2d, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
import matplotlib.pyplot as plt


# problem for students
def euclidean_distance(dp1, dp2):
    """Calculate the Euclidean distance between two data points.

    Arguments:
        dp1: a list of floats representing a data point
        dp2: a list of floats representing a data point

    Returns: the Euclidean distance between two data points
    """
    dist = 0.0
    for i in range(len(dp1)):
        dist += ((dp1[i] - dp2[i])**2)
    return dist ** 0.5


# problem for students
def assign_data(data_point, centroids):
    """Assign a single data point to the closest centroid. You should use
    the euclidean_distance function (that you previously implemented).

    Arguments:
        data_point: a list of floats representing a data point
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a string as the key name of the closest centroid to the data point
    """
    val = float("inf")
    ret = ""
    for i in list(centroids.keys()):
        dist = euclidean_distance(data_point, centroids[i])
        if (val > dist):
            val = dist
            ret = i
    return ret


# problem for students
def update_assignment(data, centroids):
    """Assign all data points to the closest centroids. You should use
    the assign_data function (that you previously implemented).

    Arguments:
        data: a list of lists representing all data points
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a new dictionary whose keys are the centroids' key names and
             values are lists of points that belong to the centroid. If a
             given centroid does not have any data points closest to it,
             do not include the centroid in the returned dictionary.
    """
    dic = dict()
    for point in data:
        key_name = assign_data(point, centroids)
        if key_name not in dic:
            dic[key_name] = [point]
        else:
            dic[key_name].append(point)
    return dic
            

# problem for students
def mean_of_points(data):
    """Calculate the mean of a given group of data points. You should NOT
    hard-code the dimensionality of the data points).

    Arguments:
        data: a list of lists representing a group of data points

    Returns: a list of floats as the mean of the given data points
    """
    ret = []
    for i in range(len(data[0])):
        m = 0.0
        for j in range(len(data)):
            m += (data[j][i] / len(data))
        ret.append(m)
    return ret


# problem for students
def update_centroids(assignment_dict):
    """Update centroid locations as the mean of all data points that belong
    to the cluster. You should use the mean_of_points function (that you
    previously implemented).

    Arguments:
        assignment_dict: the dictionary returned by update_assignment function

    Returns: A new dictionary representing the updated centroids
    """
    dic = dict()
    for i in list(assignment_dict.keys()):
        dic[i] = mean_of_points(assignment_dict[i])
    return dic

def main(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    centroids = init_centroids
    old_centroids = None
    step = 0
    while not converged(centroids, old_centroids):
        # save old centroid
        old_centroids = centroids
        # new assignment
        assignment_dict = update_assignment(data, old_centroids)
        # update centroids
        centroids = update_centroids(assignment_dict)
        # plot centroid
        fig = plot_2d(assignment_dict, centroids)
        plt.title(f"step{step}")
        fig.savefig(os.path.join("results", "2D", f"step{step}.png"))
        plt.clf()
        step += 1
    print(f"K-means converged after {step} steps.")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/data_2d.csv")
    init_c = load_centroids("data/2d_init_centroids.csv")
    final_c = main(data, init_c)
    write_centroids_tofile("2d_final_centroids.csv", final_c)

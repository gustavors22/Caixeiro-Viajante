import numpy as np
from scipy import spatial


class Map:
    def __init__(self, distances_file):
        self.__distances_file = distances_file

    def run(self):
        points_coordinate = self.__read_file()
        num_points = points_coordinate.shape[0]
        
        distance_matrix = spatial.distance.cdist(
            points_coordinate, points_coordinate, metric='euclidean')

        return (distance_matrix, num_points, points_coordinate)

    def __read_file(self):
        return np.loadtxt(self.__distances_file, delimiter=',')

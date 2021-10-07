from sko.GA import GA_TSP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class PCV_genetic_algorithm:
    def __init__(self, size_population, generations_number, mutation_rate, routes):
        self.__size_population = size_population
        self.__generations_number = generations_number
        self.__mutation_rate = mutation_rate
        self.__distances_matrix = routes[0]
        self.__num_points = routes[1]
        self.__points_coordinate = routes[2]

    def run(self):
        self.ga_tsp = GA_TSP(func=self.__calculate_total_distance,
                             n_dim=self.__num_points, size_pop=self.__size_population, max_iter=self.__generations_number, prob_mut=self.__mutation_rate)

        best_points, best_distance = self.ga_tsp.run()
        best_points_ = np.concatenate([best_points, [best_points[0]]])
        best_points_coordinate = self.__points_coordinate[best_points_, :]

        print(f"Best Route => {best_points}")
        print(f"Best Distance => {best_distance}")

        self.__plot_charts(best_points_coordinate)

    def __calculate_total_distance(self, routine):
        num_points, = routine.shape
        return sum([self.__distances_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    def __plot_charts(self, best_points_coordinate):
        fig, ax = plt.subplots(1, 3)

        ax[0].plot(self.ga_tsp.generation_best_Y)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Distance")

        ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
                   marker='o', markerfacecolor='b', color='c', linestyle='-')
        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Latitude")

        ax[2].plot(self.__points_coordinate[:, 0], self.__points_coordinate[:, 1],
                        marker='o', markerfacecolor='b', color='c', linestyle='-')
        ax[2].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[2].set_xlabel("Longitude")
        ax[2].set_ylabel("Latitude")
        
        plt.show()

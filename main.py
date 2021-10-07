from generate_map.map_generator import Map
from genetic_algorithm.PCV_genetic_algorithm import PCV_genetic_algorithm

mapCities = Map('distancias2.txt')

route = mapCities.run()

pcv_genetic_algorithm = PCV_genetic_algorithm(
    size_population=16, generations_number=500, mutation_rate=1, routes=route)

pcv_genetic_algorithm.run()

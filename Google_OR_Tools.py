import pandas as pd
import time
import math
import ortools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from haversine import haversine


def read_data(file_path, nrows):
    df = pd.read_csv(file_path, nrows=nrows)
    places = df['Place_Name'].tolist()
    coordinates = list(zip(df['Latitude'], df['Longitude']))
    return places, coordinates

def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = math.ceil(haversine(coordinates[i], coordinates[j]))
    return distance_matrix

def create_data_model(coordinates):
    data = {}
    data["distance_matrix"] = calculate_distance_matrix(coordinates)
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data

def print_solution(manager, routing, solution, places):
    """Prints solution on console and returns the optimal sequence."""
    print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    sequence = []
    while not routing.IsEnd(index):
        place_index = manager.IndexToNode(index)
        sequence.append(place_index)
        plan_output += f" {place_index} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    sequence.append(manager.IndexToNode(index))
    print(plan_output)
    plan_output += f"Route distance: {route_distance} miles\n"

    # Remove the last depot if it's the same as the first depot
    if sequence[-1] == 0:
        sequence = sequence[:-1]

    # Sort sequence based on the order of visitation
    return sequence

def save_to_csv(sequence, places, output_file_path):
    """Saves the solution to a CSV file."""
    result = pd.DataFrame({'seq': list(range(len(sequence))), 'place_name': [places[i] for i in sequence]})
    result.to_csv(output_file_path, index=False)

def main(coordinates, places, output_file_path):
    """Entry point of the program."""
    start_time = time.time()

    # Instantiate the data problem.
    data = create_data_model(coordinates)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console and save to CSV.
    if solution:
        sequence = print_solution(manager, routing, solution, places)
        save_to_csv(sequence, places, output_file_path)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")

if __name__ == "__main__":
    data_file_path = r'C:\Users\jishn\TSP\pythonProject1\Data1\tsp_input.csv'
    output_file_path =r'C:\Users\jishn\TSP\pythonProject1\Output\sample_tsp_input_100_warm.csv'
    places, coordinates = read_data(data_file_path, nrows=70)
    main(coordinates, places, output_file_path)

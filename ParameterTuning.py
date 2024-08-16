import pulp
from pulp import GLPK, GUROBI
import folium
import pandas as pd
from haversine import haversine



def read_data(file_path):
    df = pd.read_csv(file_path)
    places = df['Place_Name'].head(70).unique().tolist()
    coordinates = list(zip(df['Latitude'].head(70), df['Longitude'].head(70)))
    return places, coordinates


def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = haversine(coordinates[i], coordinates[j])
    return distance_matrix


def read_sequence(sequence_file_path):
    df = pd.read_csv(sequence_file_path)
    sequence = df.set_index('place_name')['seq'].to_dict()
    return sequence


def build_model(places, distance_matrix, sequence=None):
    # instantiate the problem - Python PuLP model
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    # ***************************************************
    #   Defining decision variables
    # ***************************************************
    x = {}  # Binary: x_i,j:= 1 if I am visiting city j after city i; otherwise 0
    for i in places:
        for j in places:
            if i != j:
                x[(i, j)] = pulp.LpVariable("x_" + str(places.index(i)) + '_' + str(places.index(j)), cat='Binary')
    s = {}  # Integer: s_i is the sequence number when we are visiting city i
    for i in places:
        s[i] = pulp.LpVariable("s_" + str(places.index(i)), cat='Integer', lowBound=0)

    # ********************************************
    # Objective
    # ********************************************
    # Minimize total travel distance
    obj_val = 0
    for i in places:
        for j in places:
            if i != j:
                obj_val += x[(i, j)] * distance_matrix[places.index(i)][places.index(j)]
    prob += obj_val

    # Constraint 1
    for i in places:
        aux_sum = 0
        for j in places:
            if i != j:
                aux_sum += x[(i, j)]
        prob += aux_sum == 1, 'Outgoing_sum_' + str(places.index(i))

    # Constraint 2
    for j in places:
        aux_sum = 0
        for i in places:
            if i != j:
                aux_sum += x[(i, j)]
        prob += aux_sum == 1, 'Incoming_sum_' + str(places.index(j))

    # Sub tour elimination constraint
    for i in places[:-1]:
        for j in places[:-1]:
            if i != j:
                prob += s[j] >= s[i] + 1 - (len(places) + 1) * (1 - x[(i, j)]), 'sub_tour_' + str(
                    places.index(i)) + '_' + str(places.index(j))

    # Set warm start values if provided
    if sequence:
        for place, seq in sequence.items():
            s[place].setInitialValue(seq)

    return prob, x


def solve_tsp(prob, x, places):
    # Solve the problem with warm start
    solver = 'GUROBI'  # Solver choice
    print('-' * 50)
    print('Optimization solver', solver, 'called')
    prob.writeLP("../output/tsp.lp")
    if solver == 'GUROBI':
        # Set Gurobi parameters
        prob.solve(pulp.GUROBI_CMD(options=[
            ("MIPFocus", 2),
            ("Cuts", 3),
            ("Heuristics", 0.5)
        ]))


    else:
        print(solver, 'not available')
        exit()
    print(f'Status: {pulp.LpStatus[prob.status]}')

    if pulp.LpStatus[prob.status] == 'Optimal':
        n = len(places)
        optimal_route = [places[0]]
        for i in range(1, len(places)):
            for j in range(len(places)):
                if i != j:
                    if x[(places[i], places[j])].value() is not None:
                        if x[(places[i], places[j])].value() > 0.5:
                            optimal_route.append(places[j])
                            break

        optimal_route.append(optimal_route[0])  # Return to the starting place

        total_distance = pulp.value(prob.objective)

        print("Optimal Route:", " -> ".join(optimal_route))
        print("Total Distance:", total_distance)

        return optimal_route, total_distance
    else:
        print("No optimal solution found.")
        return None, None


def plot_route(optimal_route, coordinates, places):
    # Create a map centered around the first place
    start_coord = coordinates[places.index(optimal_route[0])]
    m = folium.Map(location=start_coord, zoom_start=6)

    # Add markers for each place with tooltips
    for idx in optimal_route:
        folium.Marker(
            location=coordinates[places.index(idx)],
            popup=coordinates[places.index(idx)],
            tooltip=coordinates[places.index(idx)]
        ).add_to(m)

    # Add lines to show the route
    route_coords = [coordinates[places.index(i)] for i in optimal_route]
    folium.PolyLine(locations=route_coords, color='blue').add_to(m)

    # Save the map
    m.save('../output/tsp_route.html')


if __name__ == "__main__":
    data_file_path = r'C:\Users\jishn\TSP\pythonProject1\Data1\sample_tsp_input_40.csv'
    sequence_file_path = r'C:\Users\jishn\TSP\pythonProject1\Output\sample_tsp_input_40_warm.csv'

    places, coordinates = read_data(data_file_path)
    distance_matrix = calculate_distance_matrix(coordinates)
    sequence = read_sequence(sequence_file_path)
    problem, x = build_model(places, distance_matrix, sequence)
    optimal_route, total_distance = solve_tsp(problem, x, places)

    if optimal_route:
        plot_route(optimal_route, coordinates, places)
    print(f'Execution complete')

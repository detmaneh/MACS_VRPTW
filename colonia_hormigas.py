import random
import re
import time
import numpy as np
import matplotlib.pyplot as plt


# CARGA DE INSTANCIA

def cargar_instancia(archivo):
    datos = []
    with open(archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            partes = re.findall(r'-?\d+', linea)
            if len(partes) >= 7:
                datos.append([int(x) for x in partes[:7]])
    return np.array(datos)


# DISTANCIAS

def calc_dist(nodes):
    nodes = np.array(nodes)
    return np.sqrt(np.sum((nodes[:, None] - nodes[None, :]) ** 2, axis=-1))


def calcular_distancia_total(solucion, dist_matrix):
    distancia_total = 0.0
    for ruta in solucion:
        for i in range(len(ruta) - 1):
            u = ruta[i]
            v = ruta[i + 1]
            distancia_total += dist_matrix[u][v]
    return distancia_total


# VALIDACIÓN DE RUTAS Y SOLUCIONES

def validar_ruta(ruta, datos, vehicle_capacity, dist_matrix):
    demands = datos[:, 3]
    service_time = datos[:, 6]
    windows = datos[:, 4:6]

    tiempo = 0.0
    carga = 0.0
    dist_total = 0.0

    for i in range(len(ruta) - 1):
        u, v = ruta[i], ruta[i + 1]
        d = dist_matrix[u][v]
        llegada = tiempo + d

        # Si llega después del due date, no es factible
        if llegada > windows[v][1]:
            return False, 0.0, 0.0

        tiempo = max(llegada, windows[v][0]) + service_time[v]
        carga += demands[v]
        dist_total += d

    return carga <= vehicle_capacity, dist_total, tiempo


def validar_solucion_completa(routes, datos, vehicle_capacity, dist_matrix):
    n = len(datos)
    clientes_esperados = set(range(1, n))
    visitados = []

    for ruta in routes:
        valida, _, _ = validar_ruta(ruta, datos, vehicle_capacity, dist_matrix)
        if not valida:
            return False
        visitados.extend([nodo for nodo in ruta if nodo != 0])

    if len(visitados) != len(set(visitados)):
        return False

    if set(visitados) != clientes_esperados:
        return False

    return True


# INSERCIÓN DE NODOS OLVIDADOS

def insertion_procedure(routes, unvisited, datos, vehicle_capacity, dist_matrix):
    sin_visitar = unvisited[:]

    for nodo in unvisited:
        mejor_incremento = float('inf')
        mejor_posicion = None

        for i, ruta in enumerate(routes):
            _, dist_original, _ = validar_ruta(ruta, datos, vehicle_capacity, dist_matrix)

            for j in range(1, len(ruta)):
                nueva_ruta = ruta[:j] + [nodo] + ruta[j:]
                es_valida, dist_nueva, _ = validar_ruta(nueva_ruta, datos, vehicle_capacity, dist_matrix)

                if es_valida:
                    incremento = dist_nueva - dist_original
                    if incremento < mejor_incremento:
                        mejor_incremento = incremento
                        mejor_posicion = (i, j)

        if mejor_posicion is not None:
            indice_ruta, posicion = mejor_posicion
            routes[indice_ruta].insert(posicion, nodo)
            if nodo in sin_visitar:
                sin_visitar.remove(nodo)

    return routes, sin_visitar


# BÚSQUEDA LOCAL

def local_search_intraruta(solucion, nodos, vehicle_capacity, dist_matrix):
    nueva_sol = []
    dist_opt = 0.0

    for ruta in solucion:
        mejor_r = ruta[:]
        mejor_d = validar_ruta(mejor_r, nodos, vehicle_capacity, dist_matrix)[1]

        mejora = True
        while mejora:
            mejora = False
            for i in range(1, len(mejor_r) - 2):
                for j in range(i + 1, len(mejor_r) - 1):
                    temp_r = mejor_r[:i] + mejor_r[i:j + 1][::-1] + mejor_r[j + 1:]
                    valido, d_temp, _ = validar_ruta(temp_r, nodos, vehicle_capacity, dist_matrix)
                    if valido and d_temp < mejor_d:
                        mejor_r = temp_r
                        mejor_d = d_temp
                        mejora = True

        nueva_sol.append(mejor_r)
        dist_opt += mejor_d

    return nueva_sol, dist_opt


# BÚSQUEDA LOCAL

def local_search_interruta(solucion, datos, vehicle_capacity, dist_matrix):
    mejor_sol = [ruta[:] for ruta in solucion]
    mejor_dist = calcular_distancia_total(mejor_sol, dist_matrix)

    mejora = True
    while mejora:
        mejora = False

        for r1 in range(len(mejor_sol)):
            for r2 in range(len(mejor_sol)):
                if r1 == r2:
                    continue

                ruta1 = mejor_sol[r1]
                ruta2 = mejor_sol[r2]

                # no mover depósito
                for i in range(1, len(ruta1) - 1):
                    nodo = ruta1[i]

                    nueva_ruta1 = ruta1[:i] + ruta1[i + 1:]

                    # evitar rutas vacías del tipo [0,0] si no quieres eliminarlas todavía
                    if len(nueva_ruta1) < 2:
                        continue

                    val1, _, _ = validar_ruta(nueva_ruta1, datos, vehicle_capacity, dist_matrix)
                    if not val1:
                        continue

                    for j in range(1, len(ruta2)):
                        nueva_ruta2 = ruta2[:j] + [nodo] + ruta2[j:]

                        val2, _, _ = validar_ruta(nueva_ruta2, datos, vehicle_capacity, dist_matrix)
                        if not val2:
                            continue

                        candidata = [ruta[:] for ruta in mejor_sol]
                        candidata[r1] = nueva_ruta1
                        candidata[r2] = nueva_ruta2

                        # eliminar rutas vacías tipo [0,0]
                        candidata = [r for r in candidata if len(r) > 2]

                        if not validar_solucion_completa(candidata, datos, vehicle_capacity, dist_matrix):
                            continue

                        dist_candidata = calcular_distancia_total(candidata, dist_matrix)

                        if dist_candidata + 1e-9 < mejor_dist:
                            mejor_sol = candidata
                            mejor_dist = dist_candidata
                            mejora = True
                            break

                    if mejora:
                        break
                if mejora:
                    break
            if mejora:
                break

    return mejor_sol, mejor_dist


# PARÁMETROS ACO

phi = 0.05   # evaporación local
rho = 0.1    # evaporación global
beta = 4     # peso heurístico
q0 = 0.6     # menor que antes => más exploración


# CLASE PRINCIPAL

class MACS_VRPTW:
    def __init__(self, datos, vehicle_capacity):
        self.nodes_full = datos
        self.nodes = datos[:, 1:3]
        self.capacity = vehicle_capacity
        self.demands = datos[:, 3]
        self.service_time = datos[:, 6]
        self.n = len(self.nodes)
        self.windows = datos[:, 4:6]
        self.dist_matrix = calc_dist(self.nodes)

        # Feromona inicial
        self.pheromone = np.ones(self.dist_matrix.shape) * 0.1

        self.best_routes = None
        self.best_dist = float('inf')

        # Inicialmente el número de vehículos permitido es alto
        self.min_v = 25

        self.history_dist = []
        self.history_vehicles = []

        # Penalización para clientes no servidos
        self.IN = np.ones(self.n)

        # Control del proceso de reducción de vehículos
        self.stable_iterations_on_current_v = 0
        self.best_dist_for_current_v = float('inf')

    def run_macs(self, iterations=100, refine_iterations_before_reduce=10, callback=None):
        for i in range(iterations):
            # Construcción base con el número actual de vehículos
            sol_time, unvisited_time = self.build_solution(self.min_v)

            if unvisited_time:
                sol_time, unvisited_time = insertion_procedure(
                    sol_time, unvisited_time, self.nodes_full, self.capacity, self.dist_matrix
                )

            # Si es factible, refinar distancia
            if not unvisited_time and validar_solucion_completa(sol_time, self.nodes_full, self.capacity, self.dist_matrix):
                sol_opt, dist_opt = local_search_intraruta(
                    sol_time, self.nodes_full, self.capacity, self.dist_matrix
                )

                sol_opt, dist_opt = local_search_interruta(
                    sol_opt, self.nodes_full, self.capacity, self.dist_matrix
                )

                if dist_opt < self.best_dist:
                    self.best_dist = dist_opt
                    self.best_routes = [ruta[:] for ruta in sol_opt]

                if dist_opt < self.best_dist_for_current_v:
                    self.best_dist_for_current_v = dist_opt
                    self.stable_iterations_on_current_v = 0
                else:
                    self.stable_iterations_on_current_v += 1

                # Actualización global de feromonas
                if self.best_routes is not None:
                    self.global_update(self.best_routes, self.best_dist)

            else:
                self.stable_iterations_on_current_v += 1

            # Intentar reducir vehículos solo después de varias
            #  iteraciones estables refinando el valor actual
            if self.stable_iterations_on_current_v >= refine_iterations_before_reduce and self.min_v > 1:
                candidato_v = self.min_v - 1
                sol_vei, unvisited_vei = self.build_solution(candidato_v)

                visitados = {nodo for ruta in sol_vei for nodo in ruta}
                for nodo in range(1, self.n):
                    if nodo in visitados:
                        self.IN[nodo] = 1
                    else:
                        self.IN[nodo] += 1

                if unvisited_vei:
                    sol_vei, unvisited_vei = insertion_procedure(
                        sol_vei, unvisited_vei, self.nodes_full, self.capacity, self.dist_matrix
                    )

                if not unvisited_vei and validar_solucion_completa(sol_vei, self.nodes_full, self.capacity, self.dist_matrix):
                    # Sí se logró con menos vehículos
                    self.min_v = candidato_v

                    sol_vei, dist_vei = local_search_intraruta(
                        sol_vei, self.nodes_full, self.capacity, self.dist_matrix
                    )
                    sol_vei, dist_vei = local_search_interruta(
                        sol_vei, self.nodes_full, self.capacity, self.dist_matrix
                    )

                    if dist_vei < self.best_dist:
                        self.best_dist = dist_vei
                        self.best_routes = [ruta[:] for ruta in sol_vei]

                    print(f"Iteracion {i}: Vehiculos reducidos a {self.min_v}")

                # reiniciar contador del nivel actual, pero NO reiniciar feromonas
                self.best_dist_for_current_v = float('inf')
                self.stable_iterations_on_current_v = 0

            self.history_dist.append(self.best_dist if self.best_routes is not None else np.nan)
            self.history_vehicles.append(self.min_v)

            if i % 10 == 0:
                dist_txt = f"{self.best_dist:.2f}" if self.best_routes is not None else "NA"
                print(f"Iter {i:03d} | Vehículos actuales: {self.min_v} | Mejor distancia: {dist_txt}")

    def build_solution(self, vehicles):
        unvisited = list(range(1, len(self.nodes)))
        routes = []

        for _ in range(vehicles):
            if not unvisited:
                break

            current_route = [0]
            current_time = 0.0
            current_load = 0.0
            curr = 0

            while unvisited:
                feasible = []

                for j in unvisited:
                    arrival_time = current_time + self.dist_matrix[curr][j]

                    start_service = max(arrival_time, self.windows[j][0])
                    finish_service = start_service + self.service_time[j]

                    if current_load + self.demands[j] <= self.capacity and start_service <= self.windows[j][1]:
                        feasible.append(j)

                if not feasible:
                    break

                next_node = self.select_next(curr, feasible, current_time)
                unvisited.remove(next_node)

                arrival_time = current_time + self.dist_matrix[curr][next_node]
                current_time = max(arrival_time, self.windows[next_node][0]) + self.service_time[next_node]
                current_load += self.demands[next_node]

                current_route.append(next_node)

                # actualización local de feromona
                self.pheromone[curr][next_node] = (1 - phi) * self.pheromone[curr][next_node] + phi * (1 / (self.n * self.n))
                self.pheromone[next_node][curr] = self.pheromone[curr][next_node]

                curr = next_node

            current_route.append(0)
            routes.append(current_route)

        return routes, unvisited

    def select_next(self, i, feasible, current_time):
        scores = []

        for j in feasible:
            arrival_time = current_time + self.dist_matrix[i][j]
            wait_time = max(0, self.windows[j][0] - arrival_time)
            delivery_urgency = max(1, self.windows[j][1] - arrival_time)

            eta = 1.0 / (self.dist_matrix[i][j] + wait_time + 0.5 * delivery_urgency)
            tau = self.pheromone[i][j]

            score = tau * (eta ** beta) * (self.IN[j] ** 2)
            scores.append(score)

        if random.random() < q0:
            return feasible[int(np.argmax(scores))]
        else:
            total_score = sum(scores)
            if total_score <= 0:
                return random.choice(feasible)
            probs = [s / total_score for s in scores]
            return np.random.choice(feasible, p=probs)

    def global_update(self, best_routes, best_dist):
        self.pheromone *= (1 - rho)

        deposit = 1.0 / max(best_dist, 1e-9)

        for route in best_routes:
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                self.pheromone[u][v] += rho * deposit
                self.pheromone[v][u] = self.pheromone[u][v]


# IMPRESIÓN DE RESULTADOS

def print_detailed_routes(best_routes, best_dist, vehicle_count, demands, dist_matrix):
    print("\n" + "=" * 60)
    print("RESUMEN DE LA SOLUCION")
    print(f"Vehiculos Totales: {vehicle_count}")
    print(f"Distancia Total: {best_dist:.2f}")
    print("=" * 60)

    for i, tour in enumerate(best_routes):
        load = sum(demands[n] for n in tour)

        d_ruta = 0.0
        for j in range(len(tour) - 1):
            d_ruta += dist_matrix[tour[j], tour[j + 1]]

        ruta_str = " -> ".join(map(str, tour))

        print(f"VEHICULO {i + 1}:")
        print(f"  Ruta:  {ruta_str}")
        print(f"  Carga: {load:.0f}/200 | Distancia: {d_ruta:.2f}")
        print("-" * 30)


# GRÁFICAS

def plot_final_solution(nodes, best_routes, best_dist):
    plt.figure(figsize=(12, 8))
    plt.scatter(nodes[1:, 0], nodes[1:, 1], c='blue', s=30, label='Clientes')
    plt.scatter(nodes[0, 0], nodes[0, 1], c='red', marker='s', s=150, label='Deposito')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(best_routes)))
    for idx, tour in enumerate(best_routes):
        route = nodes[tour]
        plt.plot(route[:, 0], route[:, 1], color=colors[idx], alpha=0.7, linewidth=2)

    plt.title(f"Solucion MACS-VRPTW\nDistancia: {best_dist:.2f} | Vehiculos: {len(best_routes)}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_convergence(history_dist):
    plt.figure(figsize=(10, 5))
    plt.plot(history_dist, color='green', linewidth=2)
    plt.title("Convergencia del Algoritmo MACS-VRPTW")
    plt.xlabel("Iteracion")
    plt.ylabel("Mejor Distancia Total")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# EJECUCIÓN

if __name__ == "__main__":
    archivo_instancia = "c103.txt"
    capacidad = 200

    datos = cargar_instancia(archivo_instancia)

    print("Archivo cargado:", archivo_instancia)
    print("Filas leídas:", datos.shape[0])
    print("Primeras 5 filas:")
    print(datos[:5])

    solver = MACS_VRPTW(datos, capacidad)


    solver.run_macs(iterations=400, refine_iterations_before_reduce=12)

    if solver.best_routes is not None:
        print_detailed_routes(
            solver.best_routes,
            solver.best_dist,
            len(solver.best_routes),
            solver.demands,
            solver.dist_matrix
        )

        plot_convergence(solver.history_dist)
        plot_final_solution(solver.nodes, solver.best_routes, solver.best_dist)
    else:
        print("No se encontró una solución factible.")
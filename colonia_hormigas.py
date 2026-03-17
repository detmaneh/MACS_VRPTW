import random, re, numpy as np, matplotlib.pyplot as plt

# Cargar datos del txt
def cargar_instancia(archivo):
    datos = []
    with open(archivo, 'r') as f:
        for linea in f:
            partes = re.findall(r'\d+', linea)
            if len(partes) >= 7:
                datos.append([int(x) for x in partes[:7]])
    return np.array(datos)

# Calcular distancia euclidiana entre nodos -> Matriz de distancia
def calc_dist(nodes):
    nodes = np.array(nodes)
    return np.sqrt(np.sum((nodes[:, None] - nodes[None, :])**2, axis=-1))

# Calcular la distancia total en una solucion
def calcular_distancia_total(solucion, dist_matrix):
    distancia_total = 0
    
    for ruta in solucion:
        for i in range(len(ruta) - 1): # Calcular distancia entre cada par de nodos
            u = ruta[i]   # Nodo origen
            v = ruta[i+1] # Nodo destino
            distancia_total += dist_matrix[u][v]
    return distancia_total

# Validar que la ruta sea factible
def validar_ruta(ruta, datos, vehicle_capacity, dist_matrix):
    capacity = vehicle_capacity # Capacidad del vehiculo
    demands = datos[:, 3] # Demanda del cliente
    service_time = datos[:, 6] # Tiempo de servicio
    windows = datos[:, 4:6] # Ventanas de tiempo [Ready time, Due date]

    tiempo, carga, dist_total = 0, 0, 0

    for i in range(len(ruta)-1):
        u, v = ruta[i], ruta[i+1]
        d = dist_matrix[u][v] # Distancia entre nodos
        llegada = tiempo + d # Tiempo de llegada al nodo
        if llegada > windows[v][1]: # Violacion de Due Date
            return False, 0, 0 
        tiempo = max(llegada, windows[v][0]) + service_time[v] # Tiempo de llegada + servicio
        carga += demands[v] # Aumentar la carga en el vehiculo en la ruta
        dist_total += d
    return carga <= capacity, dist_total, tiempo

# Insertar nodos que ha sido olvidados en las rutas
def insertion_procedure(routes, unvisited, datos, vehicle_capacity, dist_matrix):
        sin_visitar = unvisited[:] # Lista de nodos sin visitar

        for nodo in unvisited:
            mejor_incremento = float('inf') # Aumento de la distancia
            mejor_posicion = None # Indice de la ruta donde un nodo puede encajar

            for i, ruta in enumerate(routes):
                # Calcular la distancia total de una ruta
                _, dist_original, _ = validar_ruta(ruta, datos, vehicle_capacity, dist_matrix)
                
                for j in range(1, len(ruta)):
                    nueva_ruta = ruta[:j] + [nodo] + ruta[j:] # Ruta candidata -> [nodo anterior] + [nuevo_nodo] + [nodo posterior]
                    
                    # Validar que la ruta candidata respete capacidad y ventanas de tiempo
                    es_valida, dist_nueva, _ = validar_ruta(nueva_ruta, datos, vehicle_capacity, dist_matrix)
                    
                    if es_valida:
                        incremento = dist_nueva - dist_original # Calcular el aumento de la distancia total
                        if incremento < mejor_incremento: # Distancia calculada es menor que la registrada previamente
                            mejor_incremento = incremento
                            mejor_posicion = (i, j) # Actualizar posible posicion para el nodo

            if mejor_posicion: # Se encontro una posicion para el nodo
                indice_ruta, posicion = mejor_posicion
                routes[indice_ruta].insert(posicion, nodo) # Insertar nodo en la posicion
                if nodo in sin_visitar:
                    sin_visitar.remove(nodo) # Eliminar el nodo de la lista de no visitados

        return routes, sin_visitar

# Busqueda local - 2opt -> Elimina los cruces en las rutas para reducir distancia total
def local_search(solucion, nodos, vehicle_capacity, dist_matrix):
    nueva_sol = []
    dist_opt = 0
    for ruta in solucion:
        mejor_r = ruta[:]
        mejor_d = validar_ruta(mejor_r, nodos, vehicle_capacity, dist_matrix)[1] # Distancia total para la ruta
        for i in range(1, len(ruta) - 2):
            for j in range(i + 1, len(ruta) - 1):
                temp_r = ruta[:i] + ruta[i:j+1][::-1] + ruta[j+1:] # Invertir los segmentos de los nodos
                valido, d_temp, _ = validar_ruta(temp_r, nodos, vehicle_capacity, dist_matrix) # Calcular la validez y distancia de la ruta con nodos invertidos
                if valido and d_temp < mejor_d: # La ruta invertida tiene menor distancia que la ruta original
                    mejor_r, mejor_d = temp_r, d_temp 
        nueva_sol.append(mejor_r)
        dist_opt += mejor_d
    return nueva_sol, dist_opt

# Parametros
phi = 0.1 # Evaporacion local
rho = 0.3 # Evaporacion global -> Rango recomendado 0.1-0.5
beta = 3 # Peso de la visibilidad -> Rango recomendado 2-5
q0 = 0.9 # Probabilidad de eleccion determinista

class MACS_VRPTW():
    def __init__(self, datos, vehicle_capacity):
        self.nodes_full = datos
        self.nodes = datos[:, 1:3] # Coordenadas [x, y]
        self.capacity = vehicle_capacity # Capacidad del vehiculo
        self.demands = datos[:, 3] # Demanda del cliente
        self.service_time = datos[:, 6] # Tiempo de servicio
        self.n = len(self.nodes)
        self.windows = datos[:, 4:6] # Ventanas de tiempo [Ready time, Due date]
        self.dist_matrix = calc_dist(self.nodes) # Matriz de distancia entre nodos
        self.pheromone = np.ones(self.dist_matrix.shape) * 0.1 # Matriz de feromonas inicial

        self.best_routes = None # Rutas
        self.min_v = 25 # Numero de vehiculos
        self.best_dist = float('inf') # Distancia total
        self.history_dist = [] # Historial de las distancias minimizadas por solucion

        self.IN = np.ones(self.n) # Registro del # de veces que un nodo no ha sido incluido en la ruta
    
    def run_macs(self, iterations=20, callback=None):
        for i in range(iterations):
            # 1. Colonia ACS-VEI: Intenta reducir el # de vehiculos actuales
            sol_vei, unvisited = self.build_solution(self.min_v - 1)

            # Actualizacion de matriz IN
            visitados = {nodo for ruta in sol_vei for nodo in ruta}
            for nodo in range(1, self.n): # Omite el deposito [0]
                if nodo in visitados:
                    self.IN[nodo] = 1 # Nodo fue visitado - Valor se mantiene
                else:
                    self.IN[nodo] += 1 # Nodo olvidado - Aumenta su prioridad

            if unvisited: # Sobraron nodos en la solucion
                sol_vei, unvisited = insertion_procedure(sol_vei, unvisited, self.nodes_full, self.capacity, self.dist_matrix)

            if not unvisited: # Todos los nodos fueron visitados
                self.min_v -= 1 # Disminuir la cantidad de vehiculos a v-1
                self.IN = np.ones(self.n) # Reiniciar el registro IN
                self.best_routes = sol_vei # Ruta generada en VEI = Mejor ruta generada al momento
                self.best_dist = calcular_distancia_total(sol_vei, self.dist_matrix) # Calcular la distancia total de la solucion
                self.pheromone = np.ones(self.dist_matrix.shape) * 0.1 # Reiniciar las feromonas
                print(f"Iteracion {i}: Vehiculos reducidos a {self.min_v}")

            # 2. Colonia ACS-TIME: Optimizar la distancia para un # fijo de vehiculos
            sol_time, unvisited_time = self.build_solution(self.min_v)
            if not unvisited_time:
                sol_opt, dist_opt = local_search(sol_time, self.nodes_full, self.capacity, self.dist_matrix) # Busqueda local
                if dist_opt < self.best_dist: # Se optimizo la distancia
                    self.best_dist = dist_opt
                    self.best_routes = sol_opt

            # 3. Actualizar feromonas (Global)
            if not unvisited_time: # Solo se actualiza si todos los nodos fueron visitados
                self.global_update(self.best_routes, self.best_dist)

            self.history_dist.append(self.best_dist) # Agregar distancia total encontrada en la solucion al historial
            if callback:
                callback(i, self.min_v, self.best_dist)
    
    def build_solution(self, vehicles): # Construccion de rutas -> Logica de la hormiga
        unvisited = list(range(1, len(self.nodes))) # Nodos que no han sido visitados
        routes = [] # Rutas generadas

        for v in range(vehicles):
            if not unvisited: break # Nodo ya fue visitado
            current_route = [0] # Ruta generada
            current_time = 0 # Tiempo de traslado
            current_load = 0 # Carga actual en el vehiculo
            curr = 0 # Nodo actual
            
            while unvisited:
                feasible = [] # Nodos factibles
                for j in unvisited:
                    arrival_time = current_time + self.dist_matrix[curr][j] + self.service_time[j]

                    # Verificar la capacidad y ventana de tiempo
                    if (current_load + self.demands[j] <= self.capacity and arrival_time <= self.windows[j][1]):
                        feasible.append(j)
                if not feasible: break # No se encontraron nodos factibles

                # Regla de transicion -> Como se elige el siguiente nodo
                next_node = self.select_next(curr, feasible, current_time)
                unvisited.remove(next_node) # Eliminar nodo visitado de la lista

                # Actualizar estado
                arrival_time = current_time + self.dist_matrix[curr][next_node] # Tiempo de llegada al siguiente nodo
                current_time = max(arrival_time, self.windows[next_node][0]) + self.service_time[next_node]
                current_load += self.demands[next_node]

                current_route.append(next_node) # Añadir nodo a la ruta

                # Actualizacion local de feromona
                self.pheromone[curr][next_node] = (1 - phi) * self.pheromone[curr][next_node] + phi * (1 / (self.n * self.n))
                curr = next_node

            current_route.append(0) # El vehiculo regresa al deposito
            routes.append(current_route)
        return routes, unvisited
    
    def select_next(self, i, feasible, current_time):
        scores = []
        for j in feasible:
            # Calcular tiempo de llegada + espera
            arrival_time = current_time + self.dist_matrix[i][j]
            wait_time = max(0, self.windows[j][0] - arrival_time)

            # Urgencia temporal -> Priorizar nodos cuyo fin de ventana es mas cercano
            delivery_urgency = self.windows[j][1] - arrival_time

            eta = 1.0 / (self.dist_matrix[i][j]**2 + wait_time + delivery_urgency) # Penalizar distancias lejanas entre nodos
            tau = self.pheromone[i][j]
            # Calcular el atractivo (score) de cada nodo factible
            # Score = (Feromona * (1 / Distancia + Tiempo de espera + Urgencia)^Beta * IN)
            scores.append(tau * (eta ** beta) * (self.IN[j]**2)) # Priorizar nodos que no han sido visitados -> IN

        if random.random() < q0:
            # Explotacion
            return feasible[np.argmax(scores)] # Seleccionar el nodo con el atractivo mas alto
        
        else:
            # Exploracion -> Probabilistico (Rueda de ruleta)
            total_score = sum(scores)
            probs = [s / total_score for s in scores] # Normalizar puntajes para que sumen 1
            return np.random.choice(feasible, p=probs) # Seleccion aleatoria basada en pesos (probabilidades)
    
    def global_update(self, best_routes, best_dist):
            # Evaporacion y refuerzo de la mejor hormiga
            self.pheromone *= (1 - rho)
            deposit = 1.0 / best_dist
            for route in best_routes:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i+1]
                    self.pheromone[u][v] += rho * deposit
        
def print_detailed_routes(best_routes, best_dist, min_v, demands, dist_matrix):
    print("\n" + "="*60)
    print(f"RESUMEN DE LA SOLUCION")
    print(f"Vehiculos Totales: {min_v}")
    print(f"Distancia Total: {best_dist:.2f}")
    print("="*60)

    for i, tour in enumerate(best_routes):
        load = sum(demands[n] for n in tour) # Calcular carga total del vehiculo
        
        # Calcular distancia de ruta en especifico
        d_ruta = 0
        for j in range(len(tour)-1):
            d_ruta += dist_matrix[tour[j], tour[j+1]]
            
        ruta_str = " -> ".join(map(str, tour)) # Formato de la ruta: 0 -> 13 -> 17 -> 0
        
        print(f"VEHICULO {i+1}:")
        print(f"  Ruta:  {ruta_str}")
        print(f"  Carga: {load:.0f}/200 | Distancia: {d_ruta:.2f}")
        print("-" * 30)

def plot_final_solution(nodes, best_routes, best_dist):
    # Creamos la figura y el eje explícitamente
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(nodes[1:, 0], nodes[1:, 1], c='blue', s=30, label='Clientes')
    ax.scatter(nodes[0, 0], nodes[0, 1], c='red', marker='s', s=150, label='Deposito')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(best_routes)))
    for idx, tour in enumerate(best_routes):
        route = nodes[tour]
        ax.plot(route[:, 0], route[:, 1], color=colors[idx], alpha=0.7, linewidth=2)
        
    ax.set_title(f"Solucion MACS-VRPTW para la instancia\nDistancia: {best_dist:.2f} | Vehiculos: {len(best_routes)}")
    ax.legend()
    ax.grid(True)
    
    # IMPORTANTE: Retornamos 'fig' en lugar de usar plt.show()
    return fig

# Grafica de convergencia - Distancia total
def plot_convergence(history_dist):
    # Creamos la figura y el eje explícitamente
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(history_dist, color='green', linewidth=2)
    ax.set_title("Convergencia del Algoritmo MACS-VRPTW")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Mejor Distancia Total")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # IMPORTANTE: Retornamos 'fig' en lugar de usar plt.show()
    return fig

if __name__ == "__main__":
    # Generar una instancia del solver
    solver = MACS_VRPTW(cargar_instancia('c103.txt'), 200)
    solver.run_macs(iterations=100)

    print_detailed_routes(solver.best_routes, solver.best_dist, solver.min_v, solver.demands, solver.dist_matrix)
    plot_convergence(solver.history_dist)
    plot_final_solution(solver.nodes, solver.best_routes, solver.best_dist)
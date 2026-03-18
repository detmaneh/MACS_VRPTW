import random, re, numpy as np, matplotlib.pyplot as plt

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
            _, dist_original, _ = validar_ruta(ruta, datos, vehicle_capacity, dist_matrix)
            for j in range(1, len(ruta)):
                nueva_ruta = ruta[:j] + [nodo] + ruta[j:] 
                es_valida, dist_nueva, _ = validar_ruta(nueva_ruta, datos, vehicle_capacity, dist_matrix)
                if es_valida:
                    incremento = dist_nueva - dist_original 
                    if incremento < mejor_incremento: 
                        mejor_incremento = incremento
                        mejor_posicion = (i, j) 

        if mejor_posicion: 
            indice_ruta, posicion = mejor_posicion
            routes[indice_ruta].insert(posicion, nodo) 
            if nodo in sin_visitar:
                sin_visitar.remove(nodo) 

    return routes, sin_visitar

# Busqueda local ORIGINAL - 2opt -> Elimina los cruces en las rutas para reducir distancia total
def local_search(solucion, nodos, vehicle_capacity, dist_matrix):
    nueva_sol = []
    dist_opt = 0
    for ruta in solucion:
        mejor_r = ruta[:]
        mejor_d = validar_ruta(mejor_r, nodos, vehicle_capacity, dist_matrix)[1] # Distancia total para la ruta
        for i in range(1, len(ruta) - 2):
            for j in range(i + 1, len(ruta) - 1):
                temp_r = ruta[:i] + ruta[i:j+1][::-1] + ruta[j+1:] # Invertir los segmentos de los nodos
                valido, d_temp, _ = validar_ruta(temp_r, nodos, vehicle_capacity, dist_matrix) 
                if valido and d_temp < mejor_d: 
                    mejor_r, mejor_d = temp_r, d_temp 
        nueva_sol.append(mejor_r)
        dist_opt += mejor_d
    return nueva_sol, dist_opt

class MACS_VRPTW():
    def __init__(self, datos, vehicle_capacity, num_ants=10, initial_vehicles=25):
        # Parametros limpios y encapsulados
        self.phi = 0.1
        self.rho = 0.1  # Global evaporation según la lectura
        self.beta = 1   # Peso heurístico balanceado
        self.q0 = 0.9   # Probabilidad de explotación
        self.num_ants = num_ants 
        
        self.nodes_full = datos
        self.nodes = datos[:, 1:3] # Coordenadas [x, y]
        self.capacity = vehicle_capacity # Capacidad del vehiculo
        self.demands = datos[:, 3] # Demanda del cliente
        self.service_time = datos[:, 6] # Tiempo de servicio
        self.n = len(self.nodes)
        self.windows = datos[:, 4:6] # Ventanas de tiempo [Ready time, Due date]
        self.dist_matrix = calc_dist(self.nodes) # Matriz de distancia entre nodos

        self.tau0 = 1.0 / (self.n * 2000.0) # Feromona inicial equilibrada

        self.best_routes = None 
        self.min_v = initial_vehicles
        # self.min_v = 25 # Numero de vehiculos inicial
        self.best_dist = float('inf') 
        self.history_dist = [] 

        self.IN = np.ones(self.n) 
        self._actualizar_matrices_expandidas() 
        
    def _actualizar_matrices_expandidas(self):
        num_clientes = self.n - 1
        n_total = self.min_v + num_clientes
        self.demands_exp = np.concatenate(([0] * self.min_v, self.demands[1:]))
        self.service_time_exp = np.concatenate(([0] * self.min_v, self.service_time[1:]))
        ventanas_deposito = np.tile(self.windows[0], (self.min_v, 1))
        self.windows_exp = np.vstack((ventanas_deposito, self.windows[1:]))
        
        self.IN_exp = np.ones(n_total)
        self.IN_exp[self.min_v:] = self.IN[1:]
        
        self.dist_matrix_exp = np.zeros((n_total, n_total))
        for i in range(self.min_v):
            self.dist_matrix_exp[i, self.min_v:] = self.dist_matrix[0, 1:]
            self.dist_matrix_exp[self.min_v:, i] = self.dist_matrix[1:, 0]
        self.dist_matrix_exp[self.min_v:, self.min_v:] = self.dist_matrix[1:, 1:]
        
        # Inicialización correcta con tau0
        self.pheromone_vei = np.ones((n_total, n_total)) * self.tau0
        self.pheromone_time = np.ones((n_total, n_total)) * self.tau0
    
    def decodificar_tour(self, tour_gigante):
        rutas_normales = []
        ruta_actual = []
        for nodo in tour_gigante:
            if nodo < self.min_v: 
                if ruta_actual:
                    ruta_actual.append(0)
                    rutas_normales.append(ruta_actual)
                ruta_actual = [0]
            else:
                cliente_original = nodo - self.min_v + 1
                ruta_actual.append(cliente_original)
        if ruta_actual and len(ruta_actual) > 1:
            ruta_actual.append(0)
            rutas_normales.append(ruta_actual)
        return rutas_normales
    
    def run_macs(self, iterations=20, callback=None):
        for i in range(iterations):
            # 1. Colonia ACS-VEI
            mejor_unvisited_vei = float('inf')
            mejor_sol_vei, mejor_sobrantes_vei = None, None
            for ant in range(self.num_ants):
                # use_IN = True para que reduzca vehículos
                sol_vei, unvisited = self.build_solution(self.min_v - 1, self.pheromone_vei, use_IN=True)
                if len(unvisited) < mejor_unvisited_vei:
                    mejor_unvisited_vei = len(unvisited)
                    mejor_sol_vei, mejor_sobrantes_vei = sol_vei, unvisited
            
            visitados = {nodo for ruta in mejor_sol_vei for nodo in ruta}
            for nodo in range(1, self.n): 
                if nodo in visitados:
                    self.IN[nodo] = 1 
                else:
                    self.IN[nodo] += 1 

            self.IN_exp[self.min_v:] = self.IN[1:] 

            if mejor_sobrantes_vei:
                mejor_sol_vei, mejor_sobrantes_vei = insertion_procedure(mejor_sol_vei, mejor_sobrantes_vei, self.nodes_full, self.capacity, self.dist_matrix)
            
            if not mejor_sobrantes_vei:
                self.min_v -= 1 
                self.IN = np.ones(self.n) 
                self.best_routes = mejor_sol_vei 
                self.best_dist = calcular_distancia_total(mejor_sol_vei, self.dist_matrix) 
                self._actualizar_matrices_expandidas() 

            # 2. Colonia ACS-TIME
            mejor_dist_time = float('inf')
            mejor_sol_time = None
            for ant in range(self.num_ants):
                # use_IN = False para que optimice distancia pura
                sol_time, unvisited_time = self.build_solution(self.min_v, self.pheromone_time, use_IN=False)
                if not unvisited_time:
                    dist_actual = calcular_distancia_total(sol_time, self.dist_matrix)
                    if dist_actual < mejor_dist_time:
                        mejor_dist_time = dist_actual
                        mejor_sol_time = sol_time
            
            if mejor_sol_time:
                # Retorna el 2-opt original que mantiene la distancia baja
                sol_opt, dist_opt = local_search(mejor_sol_time, self.nodes_full, self.capacity, self.dist_matrix)
                if dist_opt < self.best_dist: 
                    self.best_dist = dist_opt
                    self.best_routes = sol_opt

            # 3. Actualizar feromonas (Global)
            if self.best_routes is not None:
                self.global_update(self.best_routes, self.best_dist, self.pheromone_time)
                self.global_update(self.best_routes, self.best_dist, self.pheromone_vei)

            self.history_dist.append(self.best_dist) 
            if callback:
                callback(i, self.min_v, self.best_dist)
    
    def build_solution(self, num_vehiculos_activos, pheromone_matrix, use_IN): 
        clientes_sin_visitar = list(range(self.min_v, self.min_v + self.n - 1)) 
        depositos_disponibles = list(range(num_vehiculos_activos)) 
        curr = random.choice(depositos_disponibles)
        depositos_disponibles.remove(curr)
        tour_gigante = [curr]
        current_time, current_load = 0, 0
        
        while clientes_sin_visitar:
            nodos_factibles = []
            for j in clientes_sin_visitar:
                arrival_time = current_time + self.dist_matrix_exp[curr][j] + self.service_time_exp[j]
                if (current_load + self.demands_exp[j] <= self.capacity and arrival_time <= self.windows_exp[j][1]):
                    nodos_factibles.append(j)
            
            if not nodos_factibles and depositos_disponibles:
                nodos_factibles.extend(depositos_disponibles)
            if not nodos_factibles: break 
            
            next_node = self.select_next(curr, nodos_factibles, current_time, pheromone_matrix, use_IN)
            
            if next_node < self.min_v: 
                current_time, current_load = 0, 0
                depositos_disponibles.remove(next_node)
            else: 
                arrival_time = current_time + self.dist_matrix_exp[curr][next_node]
                current_time = max(arrival_time, self.windows_exp[next_node][0]) + self.service_time_exp[next_node]
                current_load += self.demands_exp[next_node]
                clientes_sin_visitar.remove(next_node)
            
            tour_gigante.append(next_node)

            # Actualizacion local de feromona segura
            pheromone_matrix[curr][next_node] = (1 - self.phi) * pheromone_matrix[curr][next_node] + (self.phi * self.tau0)
            curr = next_node

        rutas_formateadas = self.decodificar_tour(tour_gigante)
        unvisited_original = [nodo - self.min_v + 1 for nodo in clientes_sin_visitar]
        return rutas_formateadas, unvisited_original
    
    def select_next(self, i, feasible, current_time, pheromone_matrix, use_IN):
        scores = []
        for j in feasible:
            # Fórmula de atractividad original (equilibrada)
            arrival_time = current_time + self.dist_matrix_exp[i][j]
            wait_time = max(0, self.windows_exp[j][0] - arrival_time)
            delivery_urgency = self.windows_exp[j][1] - arrival_time
            
            eta = 1.0 / (self.dist_matrix_exp[i][j]**2 + wait_time + delivery_urgency + 0.0001)
            tau = pheromone_matrix[i][j]
            
            # Solo aplicar el penalizador IN si la colonia lo requiere
            in_multiplier = (self.IN_exp[j]**2) if use_IN else 1.0
            scores.append(tau * (eta ** self.beta) * in_multiplier)

        if random.random() < self.q0:
            return feasible[np.argmax(scores)]  
        else:
            total_score = sum(scores)
            if total_score == 0: return random.choice(feasible)
            probs = [s / total_score for s in scores]
            return np.random.choice(feasible, p=probs)  
    
    def global_update(self, best_routes, best_dist, pheromone_matrix):
        pheromone_matrix *= (1 - self.rho)
        deposit = self.rho / best_dist
        deposito_actual = 0
        for route in best_routes:
            if len(route) <= 2: continue 
            primer_cliente = route[1] + self.min_v - 1
            pheromone_matrix[deposito_actual][primer_cliente] += deposit
            for i in range(1, len(route) - 2):
                u = route[i] + self.min_v - 1
                v = route[i+1] + self.min_v - 1
                pheromone_matrix[u][v] += deposit
            ultimo_cliente = route[-2] + self.min_v - 1
            siguiente_deposito = deposito_actual + 1 if deposito_actual + 1 < self.min_v else deposito_actual
            pheromone_matrix[ultimo_cliente][siguiente_deposito] += deposit
            deposito_actual += 1

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
    return fig

def plot_convergence(history_dist):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(history_dist, color='green', linewidth=2)
    ax.set_title("Convergencia del Algoritmo MACS-VRPTW")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Mejor Distancia Total")
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig
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
    def __init__(self, datos, vehicle_capacity, num_ants=10): # Inicializacion, 10 Hormigas por ciclo
        self.nodes_full = datos
        self.nodes = datos[:, 1:3] 
        self.capacity = vehicle_capacity 
        self.demands = datos[:, 3] 
        self.service_time = datos[:, 6] 
        self.n = len(self.nodes)
        self.windows = datos[:, 4:6] 
        self.dist_matrix = calc_dist(self.nodes) 
        
        self.num_ants = num_ants 
        self.min_v = 25 
        
        self.IN = np.ones(self.n) 
        
        self.best_routes = None 
        self.best_dist = float('inf') 
        self.history_dist = [] 
        
        # Inicializar matrices expandidas
        self._actualizar_matrices_expandidas()
        
    def _actualizar_matrices_expandidas(self):
        num_clientes = self.n - 1
        n_total = self.min_v + num_clientes
        
        # Expandir demandas y tiempos de servicio
        self.demands_exp = np.concatenate(([0] * self.min_v, self.demands[1:]))
        self.service_time_exp = np.concatenate(([0] * self.min_v, self.service_time[1:]))
        
        # Expandir ventanas de tiempo
        ventanas_deposito = np.tile(self.windows[0], (self.min_v, 1))
        self.windows_exp = np.vstack((ventanas_deposito, self.windows[1:]))
        
        # Expandir matriz IN
        self.IN_exp = np.ones(n_total)
        self.IN_exp[self.min_v:] = self.IN[1:]
        
        # Expandir matriz de distancias (Distancia 0 entre depósitos)
        self.dist_matrix_exp = np.zeros((n_total, n_total))
        for i in range(self.min_v):
            self.dist_matrix_exp[i, self.min_v:] = self.dist_matrix[0, 1:]
            self.dist_matrix_exp[self.min_v:, i] = self.dist_matrix[1:, 0]
        self.dist_matrix_exp[self.min_v:, self.min_v:] = self.dist_matrix[1:, 1:]
        
        # Matrices de feromonas expandidas e independientes
        self.pheromone_vei = np.ones((n_total, n_total)) * 0.1
        self.pheromone_time = np.ones((n_total, n_total)) * 0.1
    
    def decodificar_tour(self, tour_gigante):
        """Convierte el tour gigante [Dep1, C1, Dep2, C2] a rutas normales [[0, 1, 0], [0, 2, 0]]"""
        rutas_normales = []
        ruta_actual = []
        
        for nodo in tour_gigante:
            if nodo < self.min_v: # Es un depósito
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
        
    def run_macs(self, iterations=20):
        for i in range(iterations):
            # 1. Colonia ACS-VEI
            mejor_unvisited_vei = float('inf')
            mejor_sol_vei, mejor_sobrantes_vei = None, None

            for ant in range(self.num_ants):
                # VEI opera con (vehículos actuales - 1)
                sol_vei, unvisited = self.build_solution(self.min_v - 1, self.pheromone_vei)
                if len(unvisited) < mejor_unvisited_vei:
                    mejor_unvisited_vei = len(unvisited)
                    mejor_sol_vei, mejor_sobrantes_vei = sol_vei, unvisited

            visitados = {nodo for ruta in mejor_sol_vei for nodo in ruta}
            for nodo in range(1, self.n):
                if nodo in visitados: self.IN[nodo] = 1
                else: self.IN[nodo] += 1
            self.IN_exp[self.min_v:] = self.IN[1:] 

            if mejor_sobrantes_vei:
                mejor_sol_vei, mejor_sobrantes_vei = insertion_procedure(mejor_sol_vei, mejor_sobrantes_vei, self.nodes_full, self.capacity, self.dist_matrix)

            if not mejor_sobrantes_vei:
                self.min_v -= 1
                self.IN = np.ones(self.n)
                self.best_routes = mejor_sol_vei
                self.best_dist = calcular_distancia_total(mejor_sol_vei, self.dist_matrix)
                
                # Regenerar matrices con el nuevo límite de vehículos
                self._actualizar_matrices_expandidas()
                print(f"Iteración {i}: ¡Vehículos reducidos a {self.min_v}!")

            # 2. Colonia ACS-TIME
            mejor_dist_time = float('inf')
            mejor_sol_time = None

            for ant in range(self.num_ants):
                sol_time, unvisited_time = self.build_solution(self.min_v, self.pheromone_time)
                if not unvisited_time:
                    dist_actual = calcular_distancia_total(sol_time, self.dist_matrix)
                    if dist_actual < mejor_dist_time:
                        mejor_dist_time = dist_actual
                        mejor_sol_time = sol_time

            if mejor_sol_time:
                sol_opt, dist_opt = local_search(mejor_sol_time, self.nodes_full, self.capacity, self.dist_matrix)
                if dist_opt < self.best_dist:
                    self.best_dist = dist_opt
                    self.best_routes = sol_opt

            # 3. Actualizar feromonas
            if self.best_routes is not None:
                self.global_update(self.best_routes, self.best_dist, self.pheromone_time)
            
            self.history_dist.append(self.best_dist)
            
    def build_solution(self, num_vehiculos_activos, pheromone_matrix): 
        # Clientes van desde el índice min_v hasta el final
        clientes_sin_visitar = list(range(self.min_v, self.min_v + self.n - 1)) 
        
        # Limitamos los depósitos disponibles a los vehículos permitidos en este ciclo
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
                
            if not nodos_factibles: break # Hormiga atascada
                
            next_node = self.select_next(curr, nodos_factibles, current_time, pheromone_matrix)
            
            if next_node < self.min_v: # Si eligió un depósito (cambio de vehículo)
                current_time, current_load = 0, 0
                depositos_disponibles.remove(next_node)
            else: # Si eligió un cliente
                arrival_time = current_time + self.dist_matrix_exp[curr][next_node]
                current_time = max(arrival_time, self.windows_exp[next_node][0]) + self.service_time_exp[next_node]
                current_load += self.demands_exp[next_node]
                clientes_sin_visitar.remove(next_node)
                
            tour_gigante.append(next_node)
            
            # Actualización local
            pheromone_matrix[curr][next_node] = (1 - phi) * pheromone_matrix[curr][next_node] + phi * (1 / (self.n * self.n))
            curr = next_node
            
        # Transformar de vuelta al formato original para evaluar
        rutas_formateadas = self.decodificar_tour(tour_gigante)
        unvisited_original = [nodo - self.min_v + 1 for nodo in clientes_sin_visitar]
        
        return rutas_formateadas, unvisited_original

    def select_next(self, i, feasible, current_time, pheromone_matrix):
        scores = []
        for j in feasible:
            arrival_time = current_time + self.dist_matrix_exp[i][j]
            wait_time = max(0, self.windows_exp[j][0] - arrival_time)
            delivery_urgency = self.windows_exp[j][1] - arrival_time

            eta = 1.0 / (self.dist_matrix_exp[i][j]**2 + wait_time + delivery_urgency + 0.0001) 
            tau = pheromone_matrix[i][j]
            scores.append(tau * (eta ** beta) * (self.IN_exp[j]**2)) 

        if random.random() < q0:
            return feasible[np.argmax(scores)]
        else:
            total_score = sum(scores)
            if total_score == 0: return random.choice(feasible)
            probs = [s / total_score for s in scores]
            return np.random.choice(feasible, p=probs)
    
    def global_update(self, best_routes, best_dist, pheromone_matrix):
        pheromone_matrix *= (1 - rho)
        deposit = 1.0 / best_dist
        
        deposito_actual = 0
        for route in best_routes:
            if len(route) <= 2: continue # Ignorar rutas vacías [0, 0]
            
            # Feromona de depósito a primer cliente
            primer_cliente = route[1] + self.min_v - 1
            pheromone_matrix[deposito_actual][primer_cliente] += rho * deposit
            
            # Feromona entre clientes
            for i in range(1, len(route) - 2):
                u = route[i] + self.min_v - 1
                v = route[i+1] + self.min_v - 1
                pheromone_matrix[u][v] += rho * deposit
            
            # Feromona de último cliente a siguiente depósito
            ultimo_cliente = route[-2] + self.min_v - 1
            siguiente_deposito = deposito_actual + 1 if deposito_actual + 1 < self.min_v else deposito_actual
            pheromone_matrix[ultimo_cliente][siguiente_deposito] += rho * deposit
            
            deposito_actual += 1
        
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
    plt.figure(figsize=(12, 8))
    plt.scatter(nodes[1:, 0], nodes[1:, 1], c='blue', s=30, label='Clientes')
    plt.scatter(nodes[0, 0], nodes[0, 1], c='red', marker='s', s=150, label='Deposito')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(best_routes)))
    for idx, tour in enumerate(best_routes):
        route = nodes[tour]
        plt.plot(route[:, 0], route[:, 1], color=colors[idx], alpha=0.7, linewidth=2)
        
    plt.title(f"Solucion MACS-VRPTW para la instancia\nDistancia: {best_dist:.2f} | Vehiculos: {len(best_routes)}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Grafica de convergencia - Distancia total
def plot_convergence(history_dist):
    plt.figure(figsize=(10, 5))
    plt.plot(history_dist, color='green', linewidth=2)
    plt.title("Convergencia del Algoritmo MACS-VRPTW")
    plt.xlabel("Iteracion")
    plt.ylabel("Mejor Distancia Total")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Generar una instancia del solver
solver = MACS_VRPTW(cargar_instancia('c103.txt'), 200)
solver.run_macs(iterations=100)

print_detailed_routes(solver.best_routes, solver.best_dist, solver.min_v, solver.demands, solver.dist_matrix) # Imprimir rutas finales
plot_convergence(solver.history_dist) # Grafica de convergencia
plot_final_solution(solver.nodes, solver.best_routes, solver.best_dist) # Grafico de las rutas generadas
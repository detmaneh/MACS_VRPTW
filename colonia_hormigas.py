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
def local_search(solucion, nodos, capacity, dist_matrix):
    """
    Intenta mover un cliente de una ruta a otra posición en una ruta distinta.
    """
    nueva_sol = [r[:] for r in solucion]
    mejoro = True
    
    while mejoro:
        mejoro = False
        for r1_idx in range(len(nueva_sol)):
            for cliente_idx in range(1, len(nueva_sol[r1_idx]) - 1):
                cliente = nueva_sol[r1_idx][cliente_idx]
                
                # Intentar mover el 'cliente' a todas las posiciones de otras rutas
                for r2_idx in range(len(nueva_sol)):
                    if r1_idx == r2_idx: continue
                    
                    for pos in range(1, len(nueva_sol[r2_idx])):
                        nueva_r1 = nueva_sol[r1_idx][:cliente_idx] + nueva_sol[r1_idx][cliente_idx+1:]
                        nueva_r2 = nueva_sol[r2_idx][:pos] + [cliente] + nueva_sol[r2_idx][pos:]
                        
                        # Validar si el cambio es factible en ambas rutas
                        v1, d1, _ = validar_ruta(nueva_r1, nodos, capacity, dist_matrix)
                        v2, d2, _ = validar_ruta(nueva_r2, nodos, capacity, dist_matrix)
                        
                        if v1 and v2:
                            # Si la distancia total de estas dos rutas baja, aceptamos el cambio
                            dist_orig = calcular_distancia_total([nueva_sol[r1_idx], nueva_sol[r2_idx]], dist_matrix)
                            dist_nueva = d1 + d2
                            
                            if dist_nueva < dist_orig - 0.001:
                                nueva_sol[r1_idx] = nueva_r1
                                nueva_sol[r2_idx] = nueva_r2
                                mejoro = True
                                break
                    if mejoro: break
            if mejoro: break
    return [r for r in nueva_sol if len(r) > 2] # Eliminar rutas vacías

# Parametros
phi = 0.1 # Evaporacion local
rho = 0.3 # Evaporacion global -> Rango recomendado 0.1-0.5
beta = 3 # Peso de la visibilidad -> Rango recomendado 2-5
q0 = 0.9 # Probabilidad de eleccion determinista

class MACS_VRPTW():
    def __init__(self, datos, vehicle_capacity, num_ants=10): # Inicializacion, 10 Hormigas por ciclo
        self.nodes_full = datos
        self.nodes = datos[:, 1:3] # Coordenadas [x, y]
        self.capacity = vehicle_capacity # Capacidad del vehiculo
        self.demands = datos[:, 3] # Demanda del cliente
        self.service_time = datos[:, 6] # Tiempo de servicio
        self.n = len(self.nodes)
        self.windows = datos[:, 4:6] # Ventanas de tiempo [Ready time, Due date]
        self.dist_matrix = calc_dist(self.nodes) # Matriz de distancia entre nodos
        self.num_ants = num_ants 

        self.best_routes = None # Rutas
        self.min_v = 25 # Numero de vehiculos
        self.best_dist = float('inf') # Distancia total
        self.history_dist = [] # Historial de las distancias minimizadas por solucion

        self.IN = np.ones(self.n) # Registro del # de veces que un nodo no ha sido incluido en la ruta
        self._actualizar_matrices_expandidas() # Inicializar matrices expandidas
        
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
    
    def run_macs(self, iterations=20, callback=None):
        for i in range(iterations):
            # 1. Colonia ACS-VEI: Intenta reducir el # de vehiculos actuales
            mejor_unvisited_vei = float('inf')
            mejor_sol_vei, mejor_sobrantes_vei = None, None
            for ant in range(self.num_ants):
                # VEI opera con (vehículos actuales - 1)
                sol_vei, unvisited = self.build_solution(self.min_v - 1, self.pheromone_vei)
                if len(unvisited) < mejor_unvisited_vei:
                    mejor_unvisited_vei = len(unvisited)
                    mejor_sol_vei, mejor_sobrantes_vei = sol_vei, unvisited
            visitados = {nodo for ruta in mejor_sol_vei for nodo in ruta}
            for nodo in range(1, self.n): # Omite el deposito [0]
                if nodo in visitados:
                    self.IN[nodo] = 1 # Nodo fue visitado - Valor se mantiene
                else:
                    self.IN[nodo] += 1 # Nodo olvidado - Aumenta su prioridad

            self.IN_exp[self.min_v:] = self.IN[1:] 

            if mejor_sobrantes_vei:
                mejor_sol_vei, mejor_sobrantes_vei = insertion_procedure(mejor_sol_vei, mejor_sobrantes_vei, self.nodes_full, self.capacity, self.dist_matrix)
            if not mejor_sobrantes_vei:
                self.min_v -= 1 # Disminuir la cantidad de vehiculos a v-1
                self.IN = np.ones(self.n) # Reiniciar el registro IN
                self.best_routes = mejor_sol_vei # Ruta generada en VEI = Mejor ruta generada al momento
                self.best_dist = calcular_distancia_total(mejor_sol_vei, self.dist_matrix) # Calcular la distancia total de la solucion
                self._actualizar_matrices_expandidas() # Regenerar matrices con el nuevo límite de vehículos
                print(f"Iteración {i}: ¡Vehículos reducidos a {self.min_v}!")

            # 2. Colonia ACS-TIME: Optimizar la distancia para un # fijo de vehiculos
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
                if dist_opt < self.best_dist: # Se optimizo la distancia
                    self.best_dist = dist_opt
                    self.best_routes = sol_opt

            # 3. Actualizar feromonas (Global)
            if self.best_routes is not None:
                self.global_update(self.best_routes, self.best_dist, self.pheromone_time)

            self.history_dist.append(self.best_dist) # Agregar distancia total encontrada en la solucion al historial
            if callback:
                callback(i, self.min_v, self.best_dist)
    
    def build_solution(self, num_vehiculos_activos, pheromone_matrix): # Construccion de rutas -> Logica de la hormiga
        clientes_sin_visitar = list(range(self.min_v, self.min_v + self.n - 1)) # Clientes van desde el índice min_v hasta el final
        depositos_disponibles = list(range(num_vehiculos_activos)) # Limitamos los depósitos disponibles a los vehículos permitidos en este ciclo
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

                # Actualizacion local de feromona
            pheromone_matrix[curr][next_node] = (1 - phi) * pheromone_matrix[curr][next_node] + phi * (1 / (self.n * self.n))
            curr = next_node

        # Transformar de vuelta al formato original para evaluar
        rutas_formateadas = self.decodificar_tour(tour_gigante)
        unvisited_original = [nodo - self.min_v + 1 for nodo in clientes_sin_visitar]
        return rutas_formateadas, unvisited_original
    
    def select_next(self, i, feasible, current_time, pheromone_matrix):
        scores = []
        for j in feasible:
            # Calcular tiempos según el paper (Pág. 11) arrival_time incluye el viaje desde i hasta j
            arrival_time = current_time + self.dist_matrix_exp[i][j] 
            # delivery_time es cuando inicia el servicio (respetando el inicio de ventana)
            delivery_time = max(arrival_time, self.windows_exp[j][0]) #[cite: 274]
            
            # Calcular delta_time (tiempo transcurrido) [cite: 275]
            delta_time = delivery_time - current_time 
            
            # Calcular la urgencia temporal (distance_ij) [cite: 276, 278]
            # Considera cuánto tiempo sobra antes de que cierre la ventana del cliente
            urgency = delta_time * (self.windows_exp[j][1] - current_time) #[cite: 278]
            
            # Evitar valores cero y aplicar el vector IN para ACS-VEI [cite: 280, 281]
            dist_final = max(1.0, urgency - self.IN_exp[j]) #[cite: 280]
            eta = 1.0 / dist_final #[cite: 281]
            
            tau = pheromone_matrix[i][j]
            scores.append(tau * (eta ** beta))
        # ACS (Gambardella & Dorigo, 1996) 
        if random.random() < self.q0:
            return feasible[np.argmax(scores)]  # Explotación [cite: 94]
        else:
            total_score = sum(scores)
            if total_score == 0: return random.choice(feasible)
            probs = [s / total_score for s in scores]
            return np.random.choice(feasible, p=probs)  # Exploración [cite: 94]
    
    def global_update(self, best_routes, best_dist, pheromone_matrix):
            # Evaporacion y refuerzo de la mejor hormiga
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

if __name__ == "__main__":
    # Generar una instancia del solver (Solo se ejecuta si corres este archivo directamente)
    solver = MACS_VRPTW(cargar_instancia('c103.txt'), 200)
    solver.run_macs(iterations=100)

    print_detailed_routes(solver.best_routes, solver.best_dist, solver.min_v, solver.demands, solver.dist_matrix) 
    plot_convergence(solver.history_dist) 
    plot_final_solution(solver.nodes, solver.best_routes, solver.best_dist)
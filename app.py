import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# IMPORTANTE: Ahora importamos desde vary.py (el cerebro de dos colonias)
from colonia_hormigas import MACS_VRPTW, plot_final_solution

# --- Lector de archivos en memoria para Streamlit ---
def cargar_instancia_en_memoria(uploaded_file):
    datos = []
    lineas = uploaded_file.getvalue().decode("utf-8").splitlines()
    for linea in lineas:
        partes = re.findall(r'\d+', linea)
        if len(partes) >= 7:
            datos.append([int(x) for x in partes[:7]])
    return np.array(datos)

# --- Configuración de la Página ---
st.set_page_config(page_title="MACS-VRPTW Multi-Corrida", layout="wide")

st.title("MACS-VRPTW: Análisis de Múltiples Corridas")
st.markdown("Sube una instancia (ej. `c103.txt`) y ejecuta el algoritmo varias veces para encontrar la mejor solución entre múltiples intentos")

# --- Interfaz de Controles ---
col_param1, col_param2, col_param3 = st.columns([2, 1, 1])

with col_param1:
    archivo_subido = st.file_uploader("Sube tu instancia (.txt)", type=['txt'])
    
with col_param2:
    # AQUÍ ESTÁ LA LÓGICA DE LAS CORRIDAS (El Jefe)
    num_corridas = st.number_input("Número de corridas:", min_value=2, max_value=20, value=5, step=1)
    iteraciones = st.number_input("Iteraciones por corrida:", min_value=10, max_value=500, value=50, step=10)
    
with col_param3:
    st.write("")
    st.write("")
    ejecutar = st.button("Iniciar Simulación Múltiple", use_container_width=True, type="primary", disabled=archivo_subido is None)

st.divider()

# --- Lógica de Procesamiento ---
if ejecutar and archivo_subido:
    st.subheader(f"Procesando {num_corridas} corridas de {archivo_subido.name}...")
    
    # Cargar los datos una sola vez
    datos = cargar_instancia_en_memoria(archivo_subido)
    resultados = []
    
    # --- Contenedores de Interfaz (Se actualizan en vivo) ---
    estado_general = st.empty()
    progress_bar_general = st.progress(0)
    
    st.markdown("#### Progreso...")
    col_rt1, col_rt2, col_rt3 = st.columns(3)
    rt_iter = col_rt1.empty()
    rt_v = col_rt2.empty()
    rt_d = col_rt3.empty()
    rt_barra_iter = st.progress(0)
    
    # El Bucle que manda llamar a vary.py múltiples veces
    for corrida in range(int(num_corridas)):
        nombre_corrida = f"Corrida {corrida + 1}"
        estado_general.info(f"Calculando **{nombre_corrida}**... ({corrida + 1}/{num_corridas})")
        
        # --- Función Callback para actualizar Streamlit en cada iteración ---
        def actualizar_ui_en_vivo(iteracion, min_v, best_dist):
            rt_iter.metric(f"Iteración ({nombre_corrida})", f"{iteracion + 1} / {iteraciones}")
            rt_v.metric("Flota (Minimizando)", min_v)
            
            dist_str = f"{best_dist:.2f}" if best_dist != float('inf') else "Explorando rutas..."
            rt_d.metric("Distancia (Optimizando)", dist_str)
            
            rt_barra_iter.progress((iteracion + 1) / iteraciones)

        # Inicializar un nuevo solver de vary.py para esta corrida específica
        solver = MACS_VRPTW(datos, 200)
        
        # Ejecutar
        solver.run_macs(iterations=iteraciones, callback=actualizar_ui_en_vivo)
        
        # Guardar los resultados al terminar
        resultados.append({
            'Corrida': nombre_corrida,
            'Vehículos': solver.min_v,
            'Distancia': solver.best_dist,
            'Historial': solver.history_dist,
            'Nodos': solver.nodes,
            'Rutas': solver.best_routes,
            'Demandas': solver.demands,
            'Matriz_Dist': solver.dist_matrix
        })
            
        progress_bar_general.progress((corrida + 1) / int(num_corridas))

    estado_general.info(f"**{num_corridas} Corridas Calculadas**")

    st.divider()

    # --- Dashboard Visual ---
    if resultados:
        st.markdown("## Análisis de Resultados")
        
        # Ordenar: 1ro Menos Vehículos, 2do Menor Distancia
        resultados_ordenados = sorted(resultados, key=lambda x: (x['Vehículos'], x['Distancia']))
        mejor_resultado = resultados_ordenados[0]
        
        # El Podio
        st.success(f"**La Mejor Solución:** La `{mejor_resultado['Corrida']}` encontró la configuración más eficiente.")
        
        col_ganador1, col_ganador2, col_ganador3 = st.columns(3)
        col_ganador1.metric("Mejor Intento", mejor_resultado['Corrida'])
        col_ganador2.metric("Mejor Flota", mejor_resultado['Vehículos'])
        col_ganador3.metric("Menor Distancia", f"{mejor_resultado['Distancia']:.2f}")
        
        st.write("")
        
        tab_tabla, tab_convergencia, tab_mapa, tab_rutas = st.tabs([
            "Tabla Comparativa", 
            "Curvas de Aprendizaje", 
            "Mapa Logístico (Mejor)",
            "Desglose de Rutas (Mejor)"
        ])
        
        with tab_tabla:
            df = pd.DataFrame(resultados_ordenados)[['Corrida', 'Vehículos', 'Distancia']]
            df['Distancia'] = df['Distancia'].round(2)
            df.index = np.arange(1, len(df) + 1) 
            
            def highlight_first_row(s):
                return ['background-color: #d4edda; color: #155724; font-weight: bold' if s.name == 1 else '' for v in s]

            st.dataframe(df.style.apply(highlight_first_row, axis=1), use_container_width=True)
            
        with tab_convergencia:
            # Aquí construimos la gráfica de convergencia superpuesta directamente
            fig_conv, ax_conv = plt.subplots(figsize=(10, 5))
            
            for res in resultados:
                if res['Corrida'] == mejor_resultado['Corrida']:
                    ax_conv.plot(res['Historial'], linewidth=3.5, color='#D62828', label=f"🏆 {res['Corrida']}", zorder=5)
                else:
                    ax_conv.plot(res['Historial'], linewidth=1.5, color='gray', alpha=0.4, label=res['Corrida'])
                    
            ax_conv.set_title(f"Comparativa de {num_corridas} Corridas Independientes", fontsize=14, fontweight='bold', pad=15)
            ax_conv.set_xlabel("Iteración", fontsize=11)
            ax_conv.set_ylabel("Distancia Total", fontsize=11)
            ax_conv.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_conv.grid(True, linestyle='--', alpha=0.5)
            ax_conv.spines['top'].set_visible(False)
            ax_conv.spines['right'].set_visible(False)
            
            st.pyplot(fig_conv)
            
        with tab_mapa:
            st.markdown(f"**Visualizando rutas de la:** `{mejor_resultado['Corrida']}`")
            # Llamamos al graficador de vary.py y le pasamos el objeto fig a Streamlit
            fig_mapa = plot_final_solution(mejor_resultado['Nodos'], mejor_resultado['Rutas'], mejor_resultado['Distancia'])
            st.pyplot(fig_mapa)
            
        with tab_rutas:
            st.markdown(f"## Detalle Operativo: `{mejor_resultado['Corrida']}`")
            if mejor_resultado['Rutas']:
                for i, tour in enumerate(mejor_resultado['Rutas']):
                    load = sum(mejor_resultado['Demandas'][n] for n in tour)
                    d_ruta = sum(mejor_resultado['Matriz_Dist'][tour[j], tour[j+1]] for j in range(len(tour)-1))
                    ruta_str = " ➔ ".join(map(str, tour))
                    st.markdown(f"**Vehículo {i+1}** &nbsp;|&nbsp; Carga: `{load:.0f}/200` &nbsp;|&nbsp; Distancia: `{d_ruta:.2f}` &nbsp;|&nbsp; **Ruta:** {ruta_str}")
            else:
                st.warning("No se logró encontrar una ruta factible completa en estas iteraciones.")
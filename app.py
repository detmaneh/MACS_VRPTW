import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from colonia_hormigas import MACS_VRPTW, cargar_instancia

def crear_figura_mapa(nodes, best_routes, best_dist):
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.scatter(nodes[1:, 0], nodes[1:, 1], c='#2E86AB', s=60, label='Clientes', edgecolors='white', zorder=2)
    ax.scatter(nodes[0, 0], nodes[0, 1], c='#D62828', marker='s', s=250, label='Depósito', edgecolors='black', zorder=3)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(best_routes)))
    for idx, tour in enumerate(best_routes):
        route = nodes[tour]
        ax.plot(route[:, 0], route[:, 1], color=colors[idx], alpha=0.8, linewidth=2.5, zorder=1)
        
    ax.set_title(f"Distribución Geográfica de {len(best_routes)} Rutas", fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

def crear_figura_convergencia(history_dist):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_dist, color='#4CAF50', linewidth=3)
    ax.set_title("Evolución de la Distancia (Colonia ACS-TIME)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Iteración", fontsize=11)
    ax.set_ylabel("Mejor Distancia Total", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.fill_between(range(len(history_dist)), history_dist, color='#4CAF50', alpha=0.1)
    return fig

# --- Configuración de la página ---
st.set_page_config(page_title="MACS-VRPTW Live", layout="wide")

st.title("MACS-VRPTW: Múltiples Colonias de Hormigas")
st.markdown("Optimización de rutas logísticas con ventanas de tiempo (Instancia Solomon C103).")

# Configuración
col_param1, col_param2, col_param3 = st.columns([1, 1, 1])

with col_param1:
    iteraciones = st.number_input("Iteraciones (Ciclos de búsqueda):", min_value=10, max_value=500, value=100, step=10)
with col_param2:
    hormigas = st.number_input("Hormigas por ciclo (Opcional):", min_value=5, max_value=50, value=10, step=5)
with col_param3:
    st.write("")
    st.write("")
    ejecutar = st.button("Iniciar Optimización en Vivo", use_container_width=True, type="primary")

st.divider()

if ejecutar:
    st.subheader("Procesando...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col_metricas1, col_metricas2 = st.columns(2)
    metric_vehiculos = col_metricas1.empty()
    metric_distancia = col_metricas2.empty()

    def actualizar_ui(iteracion, min_v, best_dist):
        progreso = (iteracion + 1) / iteraciones
        progress_bar.progress(progreso)
        status_text.caption(f"Calculando iteración **{iteracion + 1}** de **{iteraciones}**...")
        
        metric_vehiculos.metric("Flota de Vehículos (Minimizando)", min_v)
        if best_dist != float('inf'):
            metric_distancia.metric("Distancia Total (Optimizando)", f"{best_dist:.2f}")

    datos_c103 = cargar_instancia('c103.txt')
    solver = MACS_VRPTW(datos_c103, 200, num_ants=hormigas) 
    solver.run_macs(iterations=iteraciones, callback=actualizar_ui)
        
    st.success("¡Optimización finalizada!")
    
    # --- RESULTADOS ORGANIZADOS EN PESTAÑAS (TABS) ---
    st.markdown("#Análisis de Resultados")
    tab1, tab2, tab3 = st.tabs(["Mapa Logístico", "Curva de Convergencia", "Desglose de Rutas"])
    
    with tab1:
        st.pyplot(crear_figura_mapa(solver.nodes, solver.best_routes, solver.best_dist))
        
    with tab2:
        st.pyplot(crear_figura_convergencia(solver.history_dist))
        
    with tab3:
        st.markdown("#### Detalle Operativo por Vehículo")
        st.write("") # Un pequeño espacio para respirar
        
        for i, tour in enumerate(solver.best_routes):
            load = sum(solver.demands[n] for n in tour)
            d_ruta = sum(solver.dist_matrix[tour[j], tour[j+1]] for j in range(len(tour)-1))
            ruta_str = " ➔ ".join(map(str, tour))
            
            # Todo en una sola línea limpia usando Markdown y resaltando los números
            st.markdown(f"**Vehículo {i+1}** &nbsp;|&nbsp; Carga: `{load:.0f}/200` &nbsp;|&nbsp; Distancia: `{d_ruta:.2f}` &nbsp;|&nbsp; **Ruta:** {ruta_str}")
        
        st.divider() # Una línea decorativa al final
else:
    st.info("Ajusta los parámetros en la parte superior y haz clic en 'Iniciar Optimización en Vivo' para comenzar.")
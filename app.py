import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from colonia_hormigas import MACS_VRPTW, plot_final_solution

BENCHMARKS = {
    "C101": {"vehicles": 10, "distance": 828.94},
    "C102": {"vehicles": 10, "distance": 828.94},
    "C103": {"vehicles": 10, "distance": 828.94},
    "R101": {"vehicles": 19, "distance": 1645.79},
    "RC101": {"vehicles": 14, "distance": 1619.80},
}

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
    num_corridas = st.number_input("Número de corridas:", min_value=2, max_value=20, value=5, step=1)
    iteraciones = st.number_input("Iteraciones por corrida:", min_value=10, max_value=500, value=50, step=10)
    vehiculos_iniciales = st.number_input("Vehículos iniciales:", min_value=5, max_value=50, value=25, step=1)
    
with col_param3:
    st.write("")
    st.write("")
    ejecutar = st.button("Iniciar Simulación Múltiple", use_container_width=True, type="primary", disabled=archivo_subido is None)

st.divider()

if ejecutar and archivo_subido:
    st.subheader(f"Procesando {num_corridas} corridas de {archivo_subido.name}...")
    
    # Cargar los datos una sola vez
    datos = cargar_instancia_en_memoria(archivo_subido)
    resultados = []
    
    # --- DICCIONARIO PARA GUARDAR EL GLOBAL ---
    record_global = {'v': float('inf'), 'd': float('inf')}
    
    # --- Contenedores de Interfaz del Global ---
    st.markdown("### Mejor Solución Encontrada Hasta Ahora")
    col_gb1, col_gb2 = st.columns(2)
    gb_v = col_gb1.empty()
    gb_d = col_gb2.empty()
    
    # Inicializar el texto
    gb_v.metric("Vehiculos Encontrados", "Buscando...")
    gb_d.metric("Distancia Encontrada", "Buscando...")
    
    st.divider()
    
    estado_general = st.empty()
    progress_bar_general = st.progress(0)
    
    st.markdown("#### Progreso de la Corrida Actual")
    col_rt1, col_rt2, col_rt3 = st.columns(3)
    rt_iter = col_rt1.empty()
    rt_v = col_rt2.empty()
    rt_d = col_rt3.empty()
    rt_barra_iter = st.progress(0)
    
    # llamar al algoritmo múltiples veces
    for corrida in range(int(num_corridas)):
        nombre_corrida = f"Corrida {corrida + 1}"
        estado_general.info(f"Calculando **{nombre_corrida}**... ({corrida + 1}/{num_corridas})")
        
        # --- Función Callback para actualizar Streamlit en cada iteración ---
        def actualizar_ui_en_vivo(iteracion, min_v, best_dist):
            # 1. Actualizar el progreso de la corrida actual
            rt_iter.metric(f"Iteración ({nombre_corrida})", f"{iteracion + 1} / {iteraciones}")
            rt_v.metric("Vehiculos (Actual)", min_v)
            
            dist_str = f"{best_dist:.2f}" if best_dist != float('inf') else "Explorando rutas..."
            rt_d.metric("Distancia (Actual)", dist_str)
            
            rt_barra_iter.progress((iteracion + 1) / iteraciones)
            
            # 2. Lógica del GLOBAL
            # Si encontramos menos vehículos, o mismos vehículos pero mejor distancia:
            if min_v < record_global['v'] or (min_v == record_global['v'] and best_dist < record_global['d']):
                record_global['v'] = min_v
                record_global['d'] = best_dist
                
                # Actualizar los contenedores inmediatamente
                gb_v.metric("Vehiculos Encontrados", record_global['v'])
                gb_d.metric("Distancia Encontrada", f"{record_global['d']:.2f}")

        # Inicializar un nuevo solver para esta corrida específica
        solver = MACS_VRPTW(datos, 200, initial_vehicles=vehiculos_iniciales)
        
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

    estado_general.success(f"**{num_corridas} Corridas Calculadas**")

    st.divider()

    if resultados:
        st.markdown("## Análisis de Resultados")
        
        # Ordenar: 1ro Menos Vehículos, 2do Menor Distancia
        resultados_ordenados = sorted(resultados, key=lambda x: (x['Vehículos'], x['Distancia']))
        mejor_resultado = resultados_ordenados[0]
        instance_name = archivo_subido.name.split('.')[0].upper()
        benchmark = BENCHMARKS.get(instance_name, None)
        
        if benchmark:
            gap_vehiculos = mejor_resultado['Vehículos'] - benchmark['vehicles']
            gap_distancia = (
                (mejor_resultado['Distancia'] - benchmark['distance']) 
                / benchmark['distance']
            ) * 100
            
        st.success(f"**La Mejor Solución Final:** La más eficiente es la corrida `{mejor_resultado['Corrida']}` .")
        
        col_ganador1, col_ganador2, col_ganador3 = st.columns(3)
        col_ganador1.metric("Mejor Intento", mejor_resultado['Corrida'])
        col_ganador2.metric("Vehiculos Encontrados", mejor_resultado['Vehículos'])
        col_ganador3.metric("Distancia Encontrada", f"{mejor_resultado['Distancia']:.2f}")
        
        if benchmark:
            st.markdown("## Comparación")

            col_b1, col_b2, col_b3 = st.columns(3)

            col_b1.metric(
                "Vehículos (Lectura vs Mejor Corrida)",
                f"{benchmark['vehicles']} vs {mejor_resultado['Vehículos']}",
                delta=gap_vehiculos,
                delta_color="inverse"
            )

            col_b2.metric(
                "Distancia (Lectura vs Mejor Corrida)",
                f"{benchmark['distance']} vs {mejor_resultado['Distancia']:.2f}",
                delta=f"{gap_distancia:.2f}%",
                delta_color="inverse"
            )

            if gap_distancia < 10:
                st.success("Excelente")
            elif gap_distancia < 30:
                st.warning("Aceptable")
            else:
                st.error("Lejos del benchmark")
                
            df_benchmark = pd.DataFrame([{
                "Instancia": instance_name,
                "Vehículos Lectura": benchmark["vehicles"],
                "Vehículos Tú": mejor_resultado["Vehículos"],
                "Gap Vehículos": gap_vehiculos,
                "Distancia Lectura": benchmark["distance"],
                "Distancia Tú": round(mejor_resultado["Distancia"], 2),
                "Gap (%)": round(gap_distancia, 2)
            }])

            st.dataframe(df_benchmark, use_container_width=True)
        
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
#LIBRERIAS NECESARIAS
import numpy as np                  # numeros aleatorios y operaciones con arreglos
import networkx as nx               # construccion de los grafos
import matplotlib.pyplot as plt     # graficas
import matplotlib.ticker as mtick   # para formatear el eje Y como porcentaje
import math                         # para elegir "pasos bonitos" en los ejes Y
import time                         # medir tiempo de ejecucion
from bisect import bisect_right     # busqueda rapida del siguiente nodo en la fila acumulada de P

#nuestra semilla aleatoria para reproducibilidad
np.random.seed(12082026)

#PARAMETROS GLOBALES
N_NODOS = 25        # numero de nodos en cada grafo
T_PASOS = 10000     # numero de pasos que camina cada caminante
N_SIM = 1000        # numero de caminantes (simulaciones) independientes por grafo
EPSILON = 0.05      # umbral para decidir que la caminata ya "se mezclo"
EPSILON2 = 0.25     # segundo umbral (el que usa el libro), solo para COMPARAR visualmente
                    # en las graficas de d(t); no se usa para calcular ningun t_mezcla
NODO_INICIAL = 0    # todos los caminantes arrancan en este nodo


#FUNCION PARA GENERAR LOS 5 GRAFOS (n = 25 nodos, etiquetados 0..24)
def generar_grafos(n=N_NODOS):
    grafos = {}
    grafos["Camino"] = nx.path_graph(n)          # nodos conectados en linea: 0-1-2-...-24
    grafos["Ciclo"] = nx.cycle_graph(n)          # como el camino, pero el ultimo nodo se conecta con el 0
    grafos["Estrella"] = nx.star_graph(n - 1)    # 1 nodo central conectado a los otros 24 (star_graph(k) crea k+1 nodos)
    grafos["Completo"] = nx.complete_graph(n)    # todos los nodos conectados entre si

    G_malla = nx.grid_2d_graph(5, 5)             # rejilla 5x5 = 25 nodos, etiquetados como (fila, col)
    G_malla = nx.convert_node_labels_to_integers(G_malla, ordering="sorted")  # renombramos a 0..24
    grafos["Malla"] = G_malla
    return grafos


#FUNCION QUE CONSTRUYE LA MATRIZ DE TRANSICION P DEL GRAFO
def construir_matriz_transicion(G, n=N_NODOS):
    # P[i][j] = 1/deg(i) si existe arista entre i y j, 0 en otro caso
    A = nx.to_numpy_array(G, nodelist=range(n))   # MATRIZ DE ADYACENCIA: 1 si hay arista entre i y j, 0 si no
    grados = A.sum(axis=1)                        # grado de cada nodo (suma de su fila)
    P = A / grados[:, None]                        # dividimos cada fila entre su grado -> probabilidades (1/grado(i))
    return P


#FUNCION QUE SIMULA N_SIM CAMINATAS CON UN FOR (un caminante a la vez, un paso a la vez)
# Devuelve la matriz de trayectorias: trayectorias[i][t] = nodo donde esta el caminante i en el paso t
def simular_caminatas_con_P(P, n=N_NODOS, n_sim=N_SIM, t_pasos=T_PASOS, nodo_inicial=NODO_INICIAL):
    cumP = np.cumsum(P, axis=1)   # acumulado de probabilidades por fila (muestreo por transformada inversa)
    cumP[:, -1] = 1.0             # evitamos que el error de punto flotante deje la ultima prob. acumulada en 0.99999...
    cumP_lista = cumP.tolist()    # listas de Python: bisect sobre listas es mucho mas rapido que sobre arreglos de numpy

    trayectorias = np.zeros((n_sim, t_pasos + 1), dtype=np.int16)

    for i in range(n_sim):                                # FOR 1: un caminante a la vez
        aleatorios = np.random.uniform(0, 1, t_pasos).tolist()   # un uniforme por paso de este caminante
        pos = nodo_inicial                                 # el caminante arranca en el nodo inicial
        camino = [pos]                                     # X_0 = nodo inicial
        for u in aleatorios:                               # FOR 2: un paso a la vez
            # siguiente nodo = primer nodo cuya prob. acumulada supera u
            pos = bisect_right(cumP_lista[pos], u)
            camino.append(pos)
        trayectorias[i] = camino                           # guardamos X_0, X_1, ..., X_T de este caminante

    return trayectorias


#FUNCION: H[t][i] = cuantos caminantes estan en el nodo i en el paso t (se obtiene a partir de las trayectorias)
def contar_visitas(trayectorias, n=N_NODOS):
    n_pasos_total = trayectorias.shape[1]          # T + 1 (incluye t = 0)
    H = np.zeros((n_pasos_total, n), dtype=int)
    for t in range(n_pasos_total):
        H[t] = np.bincount(trayectorias[:, t], minlength=n)
    return H


#FUNCION: METRICAS DE EXPLORACION (las del PDF), calculadas caminante por caminante
#   V(t)    = |{X_0, ..., X_t}|           vertices distintos visitados hasta el paso t
#   C(t)    = V(t) / N                    fraccion de cobertura
#   T_cov   = min{t : V(t) = N}           tiempo de cobertura
#   tau_v   = min{t >= 0 : X_t = v}       tiempo de primera visita al vertice v
#   R_0(T)  = sum_{t=1}^{T} 1{X_t = 0}    numero de retornos al nodo inicial
def calcular_metricas_exploracion(trayectorias, n=N_NODOS, nodo_inicial=NODO_INICIAL):
    n_sim, largo = trayectorias.shape
    t_valores = np.arange(largo)                    # t = 0, 1, ..., T

    tau = np.full((n_sim, n), np.nan)               # tau[i][v] = primera visita de v por el caminante i (nan si nunca lo visito)
    t_cobertura = np.full(n_sim, np.nan)            # T_cov de cada caminante (nan si no cubrio el grafo en T pasos)
    retornos = np.zeros(n_sim, dtype=int)           # R_0(T) de cada caminante
    V_suma = np.zeros(largo)                        # suma de V^(i)(t) sobre los caminantes, para el promedio

    for i in range(n_sim):
        x = trayectorias[i]

        # np.unique con return_index da, para cada nodo visitado, el indice de su PRIMERA aparicion = tau_v
        nodos_visitados, primeras_visitas = np.unique(x, return_index=True)
        tau[i, nodos_visitados] = primeras_visitas

        # V^(i)(t) = cuantos nodos tienen tau <= t  (contamos con searchsorted sobre los tau ordenados)
        V_suma += np.searchsorted(np.sort(primeras_visitas), t_valores, side="right")

        # T_cov = el mayor de los tau, pero solo si visito los n nodos
        if nodos_visitados.size == n:
            t_cobertura[i] = primeras_visitas.max()

        # R_0(T): cuantas veces estuvo en el nodo inicial desde t = 1 hasta T (sin contar t = 0)
        retornos[i] = np.count_nonzero(x[1:] == nodo_inicial)

    V_prom = V_suma / n_sim            # V barra (t): promedio sobre las simulaciones
    C_prom = V_prom / n                # fraccion de cobertura promedio
    return V_prom, C_prom, t_cobertura, tau, retornos


#FUNCION: DISTRIBUCION ESTACIONARIA TEORICA
def pi_teorica(G, n=N_NODOS):
    grados = np.array([G.degree(i) for i in range(n)], dtype=float)  # grado de cada nodo
    return grados / grados.sum()   # pi(i) = deg(i) / (2|E|), ya que sum(deg) = 2|E|


#FUNCION: DISTANCIA DE VARIACION TOTAL d(t) Y TIEMPO DE MEZCLA
def calcular_dtv_y_tmezcla(H, pi, n_sim=N_SIM, epsilon=EPSILON):
    mu_t = H / n_sim                               # distribucion empirica en cada paso (normalizamos por num. de caminantes)
    d_t = 0.5 * np.sum(np.abs(mu_t - pi), axis=1)  # d(t) = 1/2 * suma |mu_t(i) - pi(i)| para cada paso t

    bajo_umbral = np.where(d_t < epsilon)[0]        # pasos donde d(t) ya esta por debajo de epsilon
    t_mezcla = int(bajo_umbral[0]) if bajo_umbral.size > 0 else None  # el primero de esos pasos = tiempo de mezcla
    return d_t, t_mezcla


#FUNCION: TIEMPO DE MEZCLA DE CESARO (formula 6.18 del libro)
#   nu_x^t := (1/t) * sum_{s=1}^{t} P^s(x, .)
# En vez de comparar contra pi la distribucion de UN SOLO paso t (mu_t),
# comparamos contra pi el PROMEDIO de las distribuciones de los pasos 1, 2, ..., t.
# Esto "suaviza" la oscilacion de los grafos periodicos (bipartitos), porque promedia
# pasos pares e impares juntos, en vez de mirar un paso aislado.
def calcular_cesaro_dtv_y_tmezcla(H, pi, n_sim=N_SIM, epsilon=EPSILON):
    mu_t = H / n_sim   # mu_t[t] = distribucion empirica en el paso t (t = 0, 1, ..., T_PASOS)

    # suma_acumulada[k] = mu_t[1] + mu_t[2] + ... + mu_t[k+1]   (suma acumulada empezando en s=1, no en s=0)
    suma_acumulada = np.cumsum(mu_t[1:], axis=0)

    # dividimos cada suma acumulada entre el numero de terminos que lleva sumados (t = 1, 2, ..., T_PASOS)
    t_valores = np.arange(1, mu_t.shape[0]).reshape(-1, 1)
    nu_t = suma_acumulada / t_valores   # nu_t[k] = promedio de Cesaro para t = k+1 (formula 6.18)

    # misma formula de TVD de siempre, pero comparando nu_t (el promedio) contra pi, en vez de mu_t
    d_cesaro = 0.5 * np.sum(np.abs(nu_t - pi), axis=1)

    bajo_umbral = np.where(d_cesaro < epsilon)[0]
    # bajo_umbral son indices dentro de d_cesaro, que empieza en t=1 (no en t=0) -> sumamos 1 para el t real
    t_mezcla_cesaro = int(bajo_umbral[0]) + 1 if bajo_umbral.size > 0 else None

    return d_cesaro, t_mezcla_cesaro


#FUNCIONES PARA GRAFICAR

#FUNCION: elige un "paso bonito" para las marcas del eje Y (1, 2 o 5 x 10^k),
# apuntando a mostrar aproximadamente n_marcas marcas en total.
def elegir_paso_bonito(valor_max, n_marcas=5):
    if valor_max <= 0:
        return 0.01
    paso_crudo = valor_max / n_marcas
    exponente = math.floor(math.log10(paso_crudo))
    base = paso_crudo / (10 ** exponente)
    if base <= 1.5:
        paso_bonito = 1
    elif base <= 3.5:
        paso_bonito = 2
    elif base <= 7.5:
        paso_bonito = 5
    else:
        paso_bonito = 10
    return paso_bonito * (10 ** exponente)


# formateador manual: convierte 0.05 -> "5%", pero 0.0375 -> "3.75%"
def formato_porcentaje(x, pos):
    valor = x * 100
    texto = f"{valor:.2f}".rstrip("0").rstrip(".")   # quita ceros y el punto si sobran
    if texto == "" or texto == "-":                   # caso especial: x=0 -> "0.00" -> "" tras strip
        texto = "0"
    return texto + "%"


# FUNCION: crea la figura de 5 subgraficas (3 arriba, 2 abajo centradas) que se usa en varias ventanas
def crear_figura_5_grafos():
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.6)
    ejes = {
        "Camino":   fig.add_subplot(gs[0, 0:2]),
        "Ciclo":    fig.add_subplot(gs[0, 2:4]),
        "Estrella": fig.add_subplot(gs[0, 4:6]),
        "Completo": fig.add_subplot(gs[1, 1:3]),   # fila de abajo, centrada
        "Malla":    fig.add_subplot(gs[1, 3:5]),
    }
    return fig, ejes


def fmt_num(v):
    # formatea un numero que puede ser None/nan (por si algun grafo no alcanzo la meta en T pasos)
    return "no alcanzado" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


#====================================================================
# EJECUCION PRINCIPAL
# Flujo: 1) construir grafo -> 2) obtener matriz P -> 3) simular con P (con for)
#        -> 4) metricas de exploracion (V, C, T_cov, tau_v, R_0)
#        -> 5) estimar pi -> 6) d(t) y tiempo de mezcla (TVD y Cesaro)
#====================================================================
print(f"Simulando caminatas aleatorias sobre 5 topologias de grafo (n={N_NODOS} nodos).")
print(f"Cada grafo corre {N_SIM} caminantes independientes de {T_PASOS} pasos, saliendo del nodo {NODO_INICIAL}.\n")

grafos = generar_grafos()
resultados = {}   # tipo -> diccionario con todo lo calculado para ese grafo

inicio = time.time()

for tipo, G in grafos.items():
    t0 = time.time()

    P = construir_matriz_transicion(G)                                  # 2) matriz de transicion
    trayectorias = simular_caminatas_con_P(P)                           # 3) simulacion con for
    H = contar_visitas(trayectorias)                                    #    H[t][i] a partir de las trayectorias
    V_prom, C_prom, t_cob, tau, retornos = calcular_metricas_exploracion(trayectorias)   # 4) metricas de exploracion
    pi = pi_teorica(G)                                                  # 5) distribucion estacionaria
    d_t, t_mezcla = calcular_dtv_y_tmezcla(H, pi)                       # 6) TVD y tiempo de mezcla
    d_cesaro, t_mezcla_cesaro = calcular_cesaro_dtv_y_tmezcla(H, pi)    #    Cesaro y tiempo de mezcla

    resultados[tipo] = {
        "G": G, "H": H, "pi": pi,
        "V_prom": V_prom, "C_prom": C_prom,
        "t_cob": t_cob, "tau": tau, "retornos": retornos,
        "d_t": d_t, "t_mezcla": t_mezcla,
        "d_cesaro": d_cesaro, "t_mezcla_cesaro": t_mezcla_cesaro,
        "tiempo": time.time() - t0,
    }

    r = resultados[tipo]
    n_cubrieron = int(np.sum(~np.isnan(t_cob)))
    t_mezcla_str = str(t_mezcla) if t_mezcla is not None else f"> {T_PASOS}"
    t_mezcla_ces_str = str(t_mezcla_cesaro) if t_mezcla_cesaro is not None else f"> {T_PASOS}"
    print(f"{tipo:9s} | |E|={G.number_of_edges():3d} | t_mezcla(TVD)={t_mezcla_str:>6s} | t_mezcla(Cesaro)={t_mezcla_ces_str:>6s}"
          f" | T_cov: media={fmt_num(np.nanmean(t_cob) if n_cubrieron else None)}, "
          f"desv={fmt_num(np.nanstd(t_cob) if n_cubrieron else None)} ({n_cubrieron}/{N_SIM} cubrieron)"
          f" | R_0: media={retornos.mean():.1f}, desv={retornos.std():.1f}"
          f" | tiempo={r['tiempo']:.1f} s")

fin = time.time()
print(f"\nTiempo total de simulacion: {fin - inicio:.2f} s")

tipos_ordenados = ["Camino", "Ciclo", "Estrella", "Completo", "Malla"]


#--------------------------------------
# VENTANA 1: frecuencia de visitas por nodo (datos crudos de la simulacion)
#--------------------------------------
frecuencias = {t: r["H"].sum(axis=0) / r["H"].sum() for t, r in resultados.items()}

grafos_escala_chica = [t for t in frecuencias if t != "Estrella"]
max_chico = max(frecuencias[t].max() for t in grafos_escala_chica)
paso_chico = elegir_paso_bonito(max_chico)
paso_estrella = elegir_paso_bonito(frecuencias["Estrella"].max())

fig, ejes = crear_figura_5_grafos()
for tipo in tipos_ordenados:
    ax = ejes[tipo]
    ax.bar(range(N_NODOS), frecuencias[tipo], color="lightcoral", edgecolor="black")
    ax.set_title(tipo)
    ax.set_xlabel("Nodo")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(formato_porcentaje))
    ax.tick_params(axis="y", labelsize=8)
    if tipo == "Estrella":
        ax.yaxis.set_major_locator(mtick.MultipleLocator(paso_estrella))
    else:
        ax.yaxis.set_major_locator(mtick.MultipleLocator(paso_chico))
        ax.set_ylim(0, max_chico * 1.1)   # mismo rango para los 4 grafos de escala chica
ejes["Camino"].set_ylabel("Frecuencia")
ejes["Completo"].set_ylabel("Frecuencia")
fig.suptitle("Frecuencia de visitas por nodo (datos de la simulación)")
plt.show()


#--------------------------------------
# VENTANA 2 (a): d(t) vs pasos para las 5 topologias, sobrepuestas (DVT normal)
#--------------------------------------
plt.figure(figsize=(10, 6))
for tipo in tipos_ordenados:
    plt.plot(resultados[tipo]["d_t"], label=tipo, linewidth=1.5)

plt.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1, label=f"epsilon = {EPSILON}")
plt.axhline(y=EPSILON2, color="gray", linestyle="--", linewidth=1, label=f"epsilon 2 = {EPSILON2}")

plt.xlabel("Paso (t)")
plt.ylabel("Distancia de variación total d(t)")
plt.title("d(t) vs Pasos")
plt.grid(alpha=0.3)
plt.xscale("log")

ax = plt.gca()
ax.text(-0.01, EPSILON, r"$\epsilon$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=10, color="gray", clip_on=False)
ax.text(-0.01, EPSILON2, r"$\epsilon_2$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=10, color="gray", clip_on=False)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.tight_layout()
plt.show()


#--------------------------------------
# VENTANA 2b: d_cesaro(t) vs pasos para las 5 topologias, sobrepuestas
#--------------------------------------
plt.figure(figsize=(10, 6))
for tipo in tipos_ordenados:
    plt.plot(range(1, T_PASOS + 1), resultados[tipo]["d_cesaro"], label=tipo, linewidth=1.5)

plt.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1, label=f"epsilon = {EPSILON}")
plt.axhline(y=EPSILON2, color="gray", linestyle="--", linewidth=1, label=f"epsilon 2 = {EPSILON2}")

plt.xlabel("Paso (t)")
plt.ylabel("Distancia de Cesaro  d_cesaro(t)")
plt.title("d_cesaro(t) vs Pasos")
plt.grid(alpha=0.3)
plt.xscale("log")

ax = plt.gca()
ax.text(-0.01, EPSILON, r"$\epsilon$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=10, color="gray", clip_on=False)
ax.text(-0.01, EPSILON2, r"$\epsilon_2$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=10, color="gray", clip_on=False)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.tight_layout()
plt.show()


#--------------------------------------
# VENTANA 3: comparacion en barras agrupadas -> t_mezcla (TVD) vs t_mezcla (Cesaro)
#--------------------------------------
tipos = list(resultados.keys())

valores_tvd = [resultados[t]["t_mezcla"] for t in tipos]          # puede tener None si no convergio
valores_ces = [resultados[t]["t_mezcla_cesaro"] for t in tipos]   # puede tener None si no convergio

alturas_tvd = [v if v is not None else 0 for v in valores_tvd]    # 0 = no se dibuja la barra
alturas_ces = [v if v is not None else 0 for v in valores_ces]

x = np.arange(len(tipos))
ancho = 0.35

plt.figure(figsize=(10, 6))
barras_tvd = plt.bar(x - ancho/2, alturas_tvd, ancho, label="t_mezcla (DVT)", color="#C44E52")
barras_ces = plt.bar(x + ancho/2, alturas_ces, ancho, label="t_mezcla (Cesaro)", color="#55A868")

plt.xlabel("Tipo de grafo")
plt.ylabel("Pasos para llegar al tiempo de mezcla")
plt.title(f"Comparación: Tiempo de mezcla DVT vs Cesaro (epsilon = {EPSILON})")
plt.xticks(x, tipos)
plt.legend()
plt.grid(axis="y", alpha=0.3)

for barra, valor in zip(barras_tvd, valores_tvd):
    etiqueta = str(valor) if valor is not None else "no\nalcanzado"
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
             etiqueta, ha="center", va="bottom", fontsize=6)
for barra, valor in zip(barras_ces, valores_ces):
    etiqueta = str(valor) if valor is not None else "no\nalcanzado"
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
             etiqueta, ha="center", va="bottom", fontsize=6)
plt.show()


#--------------------------------------
# VENTANA 4 (NUEVA): cobertura del grafo -> V barra(t) y fraccion de cobertura C(t)
# El eje izquierdo es la fraccion C(t) = V(t)/N y el derecho es V barra(t) (nodos distintos visitados).
#--------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for tipo in tipos_ordenados:
    ax.plot(range(1, T_PASOS + 1), resultados[tipo]["C_prom"][1:], label=tipo, linewidth=1.5)

ax.set_xscale("log")
ax.set_xlabel("Paso (t)")
ax.set_ylabel("Fracción de cobertura promedio  C(t) = V(t) / N")
ax.set_title("Cobertura del grafo vs Pasos (promedio sobre las simulaciones)")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(formato_porcentaje))
ax.grid(alpha=0.3)
ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1), borderaxespad=0)

# eje derecho: el mismo dato expresado como numero de nodos distintos visitados
ax_der = ax.secondary_yaxis("right", functions=(lambda c: c * N_NODOS, lambda v: v / N_NODOS))
ax_der.set_ylabel("Nodos distintos visitados  V(t)")
plt.tight_layout()
plt.show()


#--------------------------------------
# VENTANA 5 (NUEVA): tiempo de cobertura T_cov (media y desviacion estandar por grafo)
#--------------------------------------
medias_cov = [np.nanmean(resultados[t]["t_cob"]) if np.any(~np.isnan(resultados[t]["t_cob"])) else 0 for t in tipos_ordenados]
desv_cov = [np.nanstd(resultados[t]["t_cob"]) if np.any(~np.isnan(resultados[t]["t_cob"])) else 0 for t in tipos_ordenados]

plt.figure(figsize=(10, 6))
barras = plt.bar(tipos_ordenados, medias_cov, yerr=desv_cov, capsize=5,
                 color="#4C72B0", edgecolor="black")
for barra, media in zip(barras, medias_cov):
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
             f"{media:.0f}", ha="center", va="bottom", fontsize=9)
plt.xlabel("Tipo de grafo")
plt.ylabel("Tiempo de cobertura T_cov (pasos)")
plt.title(f"Tiempo de cobertura desde el nodo {NODO_INICIAL} (media ± desviación estándar)")
plt.grid(axis="y", alpha=0.3)
plt.show()


#--------------------------------------
# VENTANA 6 (NUEVA): tiempo promedio de primera visita por nodo (tau barra_v)
# Es el "mapa" de que tan accesible es cada nodo desde el nodo inicial.
#--------------------------------------
fig, ejes = crear_figura_5_grafos()
for tipo in tipos_ordenados:
    ax = ejes[tipo]
    tau_prom = np.nanmean(resultados[tipo]["tau"], axis=0)   # promedio sobre las simulaciones (ignora nan)
    ax.bar(range(N_NODOS), tau_prom, color="steelblue", edgecolor="black")
    ax.set_title(tipo)
    ax.set_xlabel("Nodo")
    ax.tick_params(axis="y", labelsize=8)
ejes["Camino"].set_ylabel("Primera visita promedio")
ejes["Completo"].set_ylabel("Primera visita promedio")
fig.suptitle(f"Tiempo promedio de primera visita por nodo (desde el nodo {NODO_INICIAL})")
plt.show()


#--------------------------------------
# VENTANA 7 (NUEVA): numero de retornos al nodo inicial R_0(T)
# Se compara con el valor esperado T * pi(0), que es lo que predice la distribucion estacionaria.
#--------------------------------------
medias_ret = [resultados[t]["retornos"].mean() for t in tipos_ordenados]
desv_ret = [resultados[t]["retornos"].std() for t in tipos_ordenados]
esperados = [T_PASOS * resultados[t]["pi"][NODO_INICIAL] for t in tipos_ordenados]

x = np.arange(len(tipos_ordenados))
plt.figure(figsize=(10, 6))
plt.bar(x, medias_ret, yerr=desv_ret, capsize=5, color="#DD8452", edgecolor="black",
        label="Simulación (media ± desv.)")
plt.scatter(x, esperados, color="black", marker="D", zorder=3, label=r"Esperado  $T\,\pi(0)$")
plt.xticks(x, tipos_ordenados)
plt.xlabel("Tipo de grafo")
plt.ylabel(f"Retornos al nodo {NODO_INICIAL}  R_0(T)")
plt.title(f"Número de retornos al nodo inicial en T = {T_PASOS} pasos")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()
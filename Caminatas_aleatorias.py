#LIBRERIAS NECESARIAS
import numpy as np                  # numeros aleatorios y operaciones con arreglos
import networkx as nx               # construccion de los grafos
import matplotlib.pyplot as plt     # graficas
import matplotlib.ticker as mtick   # para formatear el eje Y como porcentaje
from matplotlib.lines import Line2D # para el cuadro de tiempos de mezcla (leyenda sin dibujo)
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
LLEVAR_A_CERO_TRAS_MEZCLA = True   # en d(t) y d_cesaro(t), despues de t_mezcla la curva se dibuja en 0
                                   # (limite teorico; la curva simulada queda en un piso de ruido)


#FUNCION PARA GENERAR LOS 4 GRAFOS (n = 25 nodos, etiquetados 0..24)
def generar_grafos(n=N_NODOS):
    grafos = {}
    grafos["Camino"] = nx.path_graph(n)          # nodos conectados en linea: 0-1-2-...-24
    grafos["Ciclo"] = nx.cycle_graph(n)          # como el camino, pero el ultimo nodo se conecta con el 0
    grafos["Estrella"] = nx.star_graph(n - 1)    # 1 nodo central conectado a los otros 24 (star_graph(k) crea k+1 nodos)
    grafos["Completo"] = nx.complete_graph(n)    # todos los nodos conectados entre si
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


#FUNCION: METRICAS DE EXPLORACION, calculadas caminante por caminante
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


# FUNCION: crea la figura de 4 subgraficas (cuadricula de 2x2) que se usa en varias ventanas
def crear_figura_4_grafos():
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)
    ejes = {
        "Camino":   fig.add_subplot(gs[0, 0]),
        "Ciclo":    fig.add_subplot(gs[0, 1]),
        "Estrella": fig.add_subplot(gs[1, 0]),
        "Completo": fig.add_subplot(gs[1, 1]),
    }
    return fig, ejes


def fmt_num(v):
    # formatea un numero que puede ser None/nan (por si algun grafo no alcanzo la meta en T pasos)
    return "no alcanzado" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


# FUNCION: dibuja, debajo de la leyenda, un segundo cuadro con los tiempos de mezcla.
# Es otra leyenda (misma tipografia) con mode="expand", para que tenga el MISMO ANCHO que la de arriba.
# El alto depende de cuantas lineas tenga.
def cuadro_tiempos_bajo_leyenda(ax, leyenda, resultados, tipos, clave, titulo, separacion=0.03):
    ax.add_artist(leyenda)            # conserva la leyenda original al crear la segunda
    ax.figure.canvas.draw()           # necesario para conocer el tamano real de la leyenda ya dibujada
    caja = leyenda.get_window_extent().transformed(ax.transAxes.inverted())

    etiquetas = []
    for tipo in tipos:
        valor = resultados[tipo][clave]
        etiquetas.append(f"{tipo} = {valor if valor is not None else 'no converge'}")
    vacios = [Line2D([], [], linestyle="none") for _ in etiquetas]   # sin dibujo, solo texto

    cuadro = ax.legend(vacios, etiquetas, title=titulo, loc="upper left",
                       bbox_to_anchor=(caja.x0, caja.y0 - separacion, caja.width, 0),   # mismo x y mismo ancho que la leyenda
                       mode="expand", borderaxespad=0, handlelength=0, handletextpad=0,
                       alignment="left")                     # titulo pegado a la izquierda del cuadro
    cuadro.get_title().set_multialignment("left")            # cada renglon del titulo tambien alineado a la izquierda


#====================================================================
# EJECUCION PRINCIPAL
# Flujo: 1) construir grafo -> 2) obtener matriz P -> 3) simular con P (con for)
#        -> 4) metricas de exploracion (V, C, T_cov, tau_v, R_0)
#        -> 5) estimar pi -> 6) d(t) y tiempo de mezcla (TVD y Cesaro)
#====================================================================
print(f"Simulando caminatas aleatorias sobre 4 topologias de grafo (n={N_NODOS} nodos).")
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

tipos_ordenados = ["Camino", "Ciclo", "Estrella", "Completo"]


#--------------------------------------
# TAMANO DE IMPRESION DE LAS GRAFICAS
#   Grupo A (ventanas 1, 6 y 9): misma figura y mismo recuadro de grafica.
#   Grupo B (ventanas 2, 2b y 3): misma figura y mismo recuadro; son mas anchas porque llevan la leyenda afuera.
#   figsize = (ancho, alto) en pulgadas. Los margenes son fracciones de la figura (0 = borde izq/abajo, 1 = borde der/arriba).
#   Ya no se usa plt.tight_layout() en estas ventanas: acomoda cada figura distinto y los recuadros no coincidirian.
#--------------------------------------
TAMANO_GRUPO_A = (10, 6)
MARGENES_GRUPO_A = dict(left=0.10, right=0.97, bottom=0.12, top=0.91)

TAMANO_GRUPO_B = (11, 6)
MARGENES_GRUPO_B = dict(left=0.08, right=0.74, bottom=0.10, top=0.92)

#--------------------------------------
# ESTILO GENERAL DE LAS GRAFICAS (texto un poco mas grande)
# Aplica a todas las ventanas de abajo. Si quieres el texto mas grande o mas chico, cambia solo estos numeros.
#--------------------------------------
plt.rcParams.update({
    "axes.labelsize": 14,      # nombres de los ejes
    "axes.titlesize": 15,      # titulo de cada grafica
    "xtick.labelsize": 12,     # numeros del eje X
    "ytick.labelsize": 12,     # numeros del eje Y
    "legend.fontsize": 12,     # leyenda
    "legend.title_fontsize": 12,
})


#--------------------------------------
# VENTANA 1: frecuencia de visitas por nodo (datos crudos de la simulacion)
#   Poligono de frecuencia: se unen con rectas las puntas de las barras, y los 4 grafos van en una sola grafica.
#   El eje Y es logaritmico (como en la de primera visita) porque el centro de la Estrella (nodo 0) recibe ~50%
#   de las visitas y sus hojas ~2%: asi caben todos los grafos en la misma escala sin recortar nada.
#--------------------------------------
frecuencias = {t: r["H"].sum(axis=0) / r["H"].sum() for t, r in resultados.items()}

fig, ax = plt.subplots(figsize=TAMANO_GRUPO_A)
for tipo in tipos_ordenados:
    ax.plot(range(N_NODOS), frecuencias[tipo], marker="o", markersize=6, linewidth=2.5, label=tipo)

ax.set_yscale("log")
# marcas "redondas" del eje Y (en fraccion) que caen dentro del rango de los datos
todas = np.concatenate([frecuencias[t] for t in tipos_ordenados])
candidatas = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
marcas_y = [c for c in candidatas if todas.min() * 0.8 <= c <= todas.max() * 1.3]
ax.set_yticks(marcas_y)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(formato_porcentaje))
ax.yaxis.set_minor_formatter(mtick.NullFormatter())          # sin etiquetas en las marcas menores
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.set_xlabel("Nodo", fontsize=16)
ax.set_ylabel("Frecuencia de visitas", fontsize=16)
ax.set_title("Frecuencia de visitas por nodo (datos de la simulación)", fontsize=18)
ax.tick_params(axis="both", labelsize=14)                   # tamaño de los numeros de los ejes
ax.grid(alpha=0.3)
ax.legend(loc="upper right", ncol=1, fontsize=14)          # leyenda dentro de la grafica, en una sola columna
fig.subplots_adjust(**MARGENES_GRUPO_A)   # mismo recuadro que las ventanas 6 y 9
plt.show()



#--------------------------------------
# VENTANA 2: d(t) vs pasos para las 4 topologias, sobrepuestas (DVT normal)
# - Si LLEVAR_A_CERO_TRAS_MEZCLA = True, desde t_mezcla la curva se dibuja en 0 (limite teorico).
# - Debajo de la leyenda va un cuadro con el tiempo de mezcla de cada grafo.
#--------------------------------------
fig, ax = plt.subplots(figsize=TAMANO_GRUPO_B)
for tipo in tipos_ordenados:
    d_graf = resultados[tipo]["d_t"]                 # d_t[t] corresponde al paso t (empieza en t = 0)
    t_mez = resultados[tipo]["t_mezcla"]
    pasos = np.arange(len(d_graf))
    if LLEVAR_A_CERO_TRAS_MEZCLA and t_mez is not None:
        # la curva llega a su valor REAL en t_mezcla (ya debajo de epsilon) y justo AHI cae verticalmente a 0;
        # para eso se repite el punto t_mezcla: una vez con su valor real y otra con 0
        x_graf = np.concatenate((pasos[:t_mez + 1], [t_mez], pasos[t_mez + 1:]))
        y_graf = np.concatenate((d_graf[:t_mez + 1], [0.0], np.zeros(len(pasos) - t_mez - 1)))
    else:
        x_graf, y_graf = pasos, d_graf               # no convergio: se dibuja tal cual
    ax.plot(x_graf, y_graf, label=tipo, linewidth=2)

ax.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1.3, label=f"epsilon = {EPSILON}")
ax.axhline(y=EPSILON2, color="gray", linestyle="--", linewidth=1.3, label=f"epsilon 2 = {EPSILON2}")

ax.set_xlabel("Paso (t)")
ax.set_ylabel("Distancia de variación total d(t)")
ax.set_title("d(t) vs Pasos")
ax.grid(alpha=0.3)
ax.set_xscale("log")

ax.text(-0.01, EPSILON, r"$\epsilon$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=13, color="gray", clip_on=False)
ax.text(-0.01, EPSILON2, r"$\epsilon_2$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=13, color="gray", clip_on=False)

leyenda = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, handlelength=3.5)
fig.subplots_adjust(**MARGENES_GRUPO_B)   # deja espacio a la derecha para leyenda y cuadro (igual que 2b y 3)
cuadro_tiempos_bajo_leyenda(ax, leyenda, resultados, tipos_ordenados, "t_mezcla",
    "Tiempo de mezcla\n(pasos)")
plt.show()


#--------------------------------------
# VENTANA 2b: d_cesaro(t) vs pasos para las 4 topologias, sobrepuestas
#--------------------------------------
fig, ax = plt.subplots(figsize=TAMANO_GRUPO_B)
for tipo in tipos_ordenados:
    ax.plot(range(1, T_PASOS + 1), resultados[tipo]["d_cesaro"], label=tipo, linewidth=2)

ax.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1.3, label=f"epsilon = {EPSILON}")
ax.axhline(y=EPSILON2, color="gray", linestyle="--", linewidth=1.3, label=f"epsilon 2 = {EPSILON2}")

ax.set_xlabel("Paso (t)")
ax.set_ylabel("Distancia de Cesaro  d_cesaro(t)")
ax.set_title("d_cesaro(t) vs Pasos")
ax.grid(alpha=0.3)
ax.set_xscale("log")

ax.text(-0.01, EPSILON, r"$\epsilon$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=13, color="gray", clip_on=False)
ax.text(-0.01, EPSILON2, r"$\epsilon_2$", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=13, color="gray", clip_on=False)

leyenda = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, handlelength=3.5)
fig.subplots_adjust(**MARGENES_GRUPO_B)   # igual que 2 y 3
cuadro_tiempos_bajo_leyenda(ax, leyenda, resultados, tipos_ordenados, "t_mezcla_cesaro",
                            "Tiempo de mezcla\nCesàro (pasos)")
plt.show()


#--------------------------------------
# VENTANA 3: cobertura del grafo -> V barra(t) y fraccion de cobertura C(t)
# El eje izquierdo es la fraccion C(t) = V(t)/N y el derecho es V barra(t) (nodos distintos visitados).
#--------------------------------------
fig, ax = plt.subplots(figsize=TAMANO_GRUPO_B)
for tipo in tipos_ordenados:
    ax.plot(range(1, T_PASOS + 1), resultados[tipo]["C_prom"][1:], label=tipo, linewidth=2)

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
fig.subplots_adjust(**MARGENES_GRUPO_B)   # mismo recuadro que las ventanas 2 y 2b
plt.show()


#--------------------------------------
# VENTANA 5: poligono de frecuencia del tiempo de cobertura T_cov, los 4 grafos en una sola grafica
# Se agrupan los T_cov de los caminantes en clases, se calcula la frecuencia relativa de cada clase y se unen
# con rectas los puntos medios. El eje X es logaritmico (clases de igual ancho en escala log) porque los tiempos
# van de ~90 pasos (Completo) a varios miles (cola del Camino). La media de cada grafo se marca con una
# linea vertical punteada del mismo color.
#--------------------------------------
N_CLASES = 30

t_cob_todos = {}
for tipo in tipos_ordenados:
    v = resultados[tipo]["t_cob"]
    t_cob_todos[tipo] = v[~np.isnan(v)]              # solo los caminantes que si cubrieron el grafo

todos = np.concatenate(list(t_cob_todos.values()))
bordes = np.logspace(np.log10(todos.min()), np.log10(todos.max()), N_CLASES + 1)   # limites de las clases (comunes a los 4 grafos)
marcas = np.sqrt(bordes[:-1] * bordes[1:])           # punto medio de cada clase (en escala log)
factor = bordes[1] / bordes[0]                       # cuanto crece cada clase
x_pol = np.concatenate(([marcas[0] / factor], marcas, [marcas[-1] * factor]))   # cerramos el poligono con una clase vacia en cada extremo

fig, ax = plt.subplots(figsize=(10, 6))
for k, tipo in enumerate(tipos_ordenados):
    datos = t_cob_todos[tipo]
    if datos.size == 0:
        continue
    conteo, _ = np.histogram(datos, bins=bordes)
    frec_rel = conteo / datos.size                   # frecuencia relativa de cada clase
    y_pol = np.concatenate(([0], frec_rel, [0]))
    ax.plot(x_pol, y_pol, marker="o", markersize=4, linewidth=2, color=f"C{k}",
            label=f"{tipo} (media = {datos.mean():.0f})")
    ax.axvline(datos.mean(), color=f"C{k}", linestyle="--", linewidth=1.3)   # media de T_cov

ax.set_xscale("log")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mtick.ScalarFormatter())   # muestra 100, 1000 en vez de 10^2, 10^3
ax.yaxis.set_major_formatter(mtick.FuncFormatter(formato_porcentaje))
ax.set_xlabel("Tiempo de cobertura T_cov (pasos)")
ax.set_ylabel("Frecuencia relativa")
ax.set_title(f"Tiempo de cobertura desde el nodo {NODO_INICIAL} ({N_SIM} simulaciones por grafo)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()


#--------------------------------------
# VENTANA 6: tiempo promedio de primera visita por nodo, los 4 grafos en una sola grafica
# Es lo mismo que las 4 graficas de barras, pero unimos con rectas la punta de cada barra (poligono)
# y dibujamos los 4 grafos juntos. En el nodo inicial el valor es 0 (ahi empieza el caminante).
#--------------------------------------
nodos = np.arange(N_NODOS)

fig, ax = plt.subplots(figsize=TAMANO_GRUPO_A)
for tipo in tipos_ordenados:
    tau_prom = np.nanmean(resultados[tipo]["tau"], axis=0)   # promedio sobre las simulaciones (ignora nan)
    ax.plot(nodos, tau_prom, marker="o", markersize=5, linewidth=2, label=tipo)

ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.set_xlabel("Nodo")
ax.set_ylabel("Primera visita promedio (pasos)")
ax.set_title(f"Tiempo promedio de primera visita por nodo (desde el nodo {NODO_INICIAL})")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.subplots_adjust(**MARGENES_GRUPO_A)   # mismo recuadro que las ventanas 1 y 9
plt.show()



#--------------------------------------
# VENTANA 9: poligono de frecuencia del tiempo promedio de retorno al nodo inicial
# Para cada caminante se calcula su tiempo promedio de retorno T / R_0(T); se agrupan en clases, se calcula
# la frecuencia relativa de cada clase y se unen con rectas los puntos medios (igual que en la ventana 5).
# La linea vertical punteada de cada color es el valor teorico 1/pi(0).
# La Estrella NO se dibuja como poligono: ahi TODOS los caminantes tardan exactamente 2 pasos
# (centro -> hoja -> centro), asi que seria un solo pico de 100% lejos de los demas grafos.
# Se indica con un texto dentro de la grafica.
#--------------------------------------
N_CLASES_RET = 30

t_ret_todos = {}
for tipo in tipos_ordenados:
    ret = resultados[tipo]["retornos"].astype(float)
    t_ret_todos[tipo] = T_PASOS / ret[ret > 0]          # tiempo promedio de retorno de cada caminante

tipos_poligono = [t for t in tipos_ordenados if t != "Estrella"]
todos_ret = np.concatenate([t_ret_todos[t] for t in tipos_poligono])
bordes_ret = np.logspace(np.log10(todos_ret.min()), np.log10(todos_ret.max()), N_CLASES_RET + 1)  # clases comunes
marcas_ret = np.sqrt(bordes_ret[:-1] * bordes_ret[1:])     # punto medio de cada clase (en escala log)
factor_ret = bordes_ret[1] / bordes_ret[0]
x_ret = np.concatenate(([marcas_ret[0] / factor_ret], marcas_ret, [marcas_ret[-1] * factor_ret]))  # cerramos el poligono

fig, ax = plt.subplots(figsize=TAMANO_GRUPO_A)
maximo = 0
for k, tipo in enumerate(tipos_ordenados):
    if tipo not in tipos_poligono:
        continue
    datos = t_ret_todos[tipo]
    conteo, _ = np.histogram(datos, bins=bordes_ret)
    frec_rel = conteo / datos.size                     # frecuencia relativa de cada clase
    y_ret = np.concatenate(([0], frec_rel, [0]))
    esperado = 1 / resultados[tipo]["pi"][NODO_INICIAL]
    ax.plot(x_ret, y_ret, marker="o", markersize=4, linewidth=2, color=f"C{k}",   # mismo color que en las demas graficas
            label=f"{tipo} (media = {datos.mean():.1f}, teórico = {esperado:.0f})")
    ax.axvline(esperado, color=f"C{k}", linestyle="--", linewidth=1.3)   # valor teorico 1/pi(0)
    maximo = max(maximo, frec_rel.max())

if "Estrella" in t_ret_todos:
    ax.text(0.98, 0.72, f"Estrella: todos los caminantes tardan\n{t_ret_todos['Estrella'].mean():.0f} pasos (teórico = 2)",
            transform=ax.transAxes, ha="right", va="top", fontsize=12, color="C2",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="C2"))

ax.set_xscale("log")
ax.set_ylim(0, maximo * 1.15)                          # espacio arriba para que no se corte el pico
marcas_x = [m for m in [10, 15, 20, 30, 50, 100, 200, 500] if bordes_ret[0] <= m <= bordes_ret[-1] * 1.2]
ax.set_xticks(marcas_x)
ax.xaxis.set_major_formatter(mtick.ScalarFormatter())   # muestra 20, 50, 100 en vez de potencias de 10
ax.xaxis.set_minor_formatter(mtick.NullFormatter())     # sin numeros en las marcas pequenas
ax.yaxis.set_major_formatter(mtick.FuncFormatter(formato_porcentaje))
ax.set_xlabel(f"Tiempo promedio de retorno al nodo {NODO_INICIAL} (pasos)")
ax.set_ylabel("Frecuencia relativa")
ax.set_title(f"Tiempo promedio de retorno al nodo inicial ({N_SIM} simulaciones por grafo)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.subplots_adjust(**MARGENES_GRUPO_A)   # mismo recuadro que las ventanas 1 y 6
plt.show()
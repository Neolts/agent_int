# %%
# Importar las librerías
import numpy as np

# Definición de los estados
def_de_estados = {'A': 0,
                  'B': 1,
                  'C': 2,
                  'D': 3,
                  'E': 4,
                  'F': 5,
                  'G': 6,
                  'H': 7,
                  'I': 8,
                  'J': 9,
                  'K': 10,
                  'L': 11,
                  'M': 12,
                  'N': 13,
                  'O': 14,
                  'P': 15,
                  'Q': 16,
                  'R': 17,
                  'S': 18,
                  'T': 19,}
print(def_de_estados)

# %%
# Definición de las acciones
acciones = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
print(acciones)

# %%
# Definicion de Recompensas
def crear_matriz_R():
       return np.array([[0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],])


# %% [markdown]

# %%
# Configuración de los parámetros gamma y alfa para el Q-Learning
gamma = 1
alpha = 1
# Inicialización de los valores Q
Q = np.array(np.zeros([18,18]))

# %%
# Implementación del proceso de Q-Learning
for i in range(1000):
    estado_actual = np.random.randint(0,18)
    accion_realizable = []
    for j in range(18):
        if crear_matriz_R()[estado_actual, j] > 0:
            accion_realizable.append(j)
    estado_siguiente = np.random.choice(accion_realizable)
    TD = crear_matriz_R()[estado_actual, estado_siguiente] + gamma*Q[estado_siguiente, np.argmax(Q[estado_siguiente,])]- Q[estado_actual, estado_siguiente]
    Q[estado_actual, estado_siguiente] = Q[estado_actual, estado_siguiente] + alpha*TD

# %%
print("Q-Values:")
print(Q.astype(int))

# %%
def RL_algorithm(R, alpha=0.9, gamma=0.8, episodes=100):
    Q = np.zeros((18, 18))

    for _ in range(episodes):
        estado_actual = np.random.randint(0, 18)
        accion_realizable = []
        for j in range(18):
            if R[estado_actual, j] > 0:
                accion_realizable.append(j)
        estado_siguiente = np.random.choice(accion_realizable)
        TD = R[estado_actual, estado_siguiente] + gamma * Q[estado_siguiente, np.argmax(Q[estado_siguiente, ])] - Q[estado_actual, estado_siguiente]
        Q[estado_actual, estado_siguiente] = Q[estado_actual, estado_siguiente] + alpha * TD

    return Q

def ruta(estado_inicial, estado_intermedio, estado_final):
    R = crear_matriz_R()
    Q = RL_algorithm(R)

    estado_actual = estado_inicial
    ruta_optima = [estado_actual]
    while estado_actual != estado_intermedio:
        estado_siguiente = np.argmax(Q[estado_actual])
        ruta_optima.append(estado_siguiente)
        estado_actual = estado_siguiente

    while estado_actual != estado_final:
        estado_siguiente = np.argmax(Q[estado_actual])
        ruta_optima.append(estado_siguiente)
        estado_actual = estado_siguiente

    return ruta_optima

estado_inicial = def_de_estados['A']
estado_intermedio = def_de_estados['G']
estado_final = def_de_estados['L']
ruta_optima = ruta(estado_inicial, estado_intermedio, estado_final)
ruta_nombres = [estado for estado, indice in def_de_estados.items() if indice in ruta_optima]
print("Ruta óptima:", ruta_nombres)





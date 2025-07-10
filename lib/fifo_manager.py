import numpy as np
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import onnxruntime as ort
import time

measures_fifos = {}  # Dicionário global para armazenar as FIFOs

# Dicionário global para armazenar as SubFIFOs
sub_fifo_A = {} 
sub_fifo_B = {}
sub_fifo_C = {}

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_fitted = False

last_subfifos_snapshot = {"A": {}, "B": {}, "C": {}}

#Carregar Modelo
model_path = r"C:\Users\joao.miranda\Documents\POC\POC_jetson_media_pipe\Neural Network [POC]\transformer_model.onnx"
session = ort.InferenceSession(model_path, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name

def initialize_fifos(measure_names, fifo_size):
    """Inicializa uma FIFO separada para cada medida, com tamanho definido externamente."""
    global measures_fifos
    measures_fifos = {name: deque(maxlen=fifo_size) for name in measure_names}

def update_subfifos():
    """
    A cada chamada, puxa:
      - A: frames [-60:-30]
      - B: frames [-45:-15]
      - C: frames [ -30:   ]
    tudo em relação ao final da FIFO principal.
    """
    for measure, dq in measures_fifos.items():
        temp = list(dq)

        # Janela C: últimos 30
        slice_C = temp[-30:]
        sub_fifo_C[measure] = deque(slice_C, maxlen=30)

        # Janela B: de 45 a 15 frames atrás
        slice_B = temp[-45:-15]
        sub_fifo_B[measure] = deque(slice_B, maxlen=30)

        # Janela A: de 60 a 30 frames atrás
        slice_A = temp[-60:-30]
        sub_fifo_A[measure] = deque(slice_A, maxlen=30)


def update_fifos(measures):
    """Atualiza as FIFOs com as novas medidas."""
    for name, value in measures.items():
        if name in measures_fifos:
            measures_fifos[name].append(value)
    
    update_subfifos()

def prepare_subfifo_matrix(n=37, n_measures=21, min_values=10):
    """
    Prepara matrizes A, B e C com shape (n, n_measures), preenchendo com 0 se faltar valor.
    Só começa a inferência se cada subFIFO tiver pelo menos `min_values` valores.
    """
    global scaler_fitted

    subfifo_configs = [
        ("A", sub_fifo_A),
        ("B", sub_fifo_B),
        ("C", sub_fifo_C),
    ]

    measure_names = list(measures_fifos.keys())[:n_measures]

    # Criamos uma matriz grande para conter os 3 blocos de subFIFO (cada um com n linhas)
    big_matrix = np.zeros((n * 3, n_measures), dtype=np.float32)

    for idx, (nome, sub_fifo) in enumerate(subfifo_configs):
        # Verifica se cada medida tem dados suficientes
        for m in measure_names:
            if len(sub_fifo.get(m, [])) < min_values:
                print(f"SubFIFO {nome} ainda com poucos dados: '{m}' tem {len(sub_fifo.get(m, []))}/{min_values} valores.")
                return None

        # Monta a matriz com preenchimento de zeros se faltar valor
        # Preenche o segmento da matriz correspondente a esse subFIFO
        for i in range(-n, 0):
            for j, m in enumerate(measure_names):
                valores = sub_fifo[m]
                valor = valores[i] if len(valores) >= abs(i) else 0.0
                big_matrix[i + n * idx, j] = valor
    
    if not scaler_fitted:
        scaler.fit(big_matrix)
        scaler_fitted = True

    normalized = scaler.transform(big_matrix)

    # Dividimos em 3 blocos e retornamos
    matA = normalized[0:n]
    matB = normalized[n:2*n]
    matC = normalized[2*n:3*n]

    return [matA.astype(np.float32), matB.astype(np.float32), matC.astype(np.float32)]

def infer_emotions_for_subfifos(fifo_matrix):
    """
    Faz a inferência nas 3 subFIFOs (A, B e C) e retorna os resultados para cada uma.
    
    Parâmetros:
        model (tf.keras.Model): Modelo carregado.
        fifo_matrix: Matriz das subFIFOs (A, B, C).
    
    Retorna:
        results: Lista de tuplas (nome_subfifo, classe_predita, confiança) para cada subFIFO (A, B e C).
    """
    results = []

    subfifo_names = ['A', 'B', 'C']  # Nomes das subFIFOs para exibição
    batch = np.stack(fifo_matrix)  # shape (3, 30, 21)
    start = time.perf_counter()
    predictions = session.run(None, {input_name: batch})[0] 
    end = time.perf_counter()
    temp = end - start
    print(f"Tempo de Inferência: {temp*1000:.1f} ms")

    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        confidence = np.max(pred)
        results.append((subfifo_names[i], predicted_class, confidence))

    return results







"""
def check_subfifos_shiftando():
    global last_subfifos_snapshot

    mudou_algo = False

    for nome, sub_fifo in [("A", sub_fifo_A), ("B", sub_fifo_B), ("C", sub_fifo_C)]:
        for medida in sub_fifo:
            atual = list(sub_fifo[medida])
            anterior = last_subfifos_snapshot[nome].get(medida)

            if anterior != atual:
                mudou_algo = True
                break  # basta uma mudança pra saber que está shiftando

    if mudou_algo:
        print("✅ SubFIFOs estão shiftando normalmente.")
    else:
        print("⚠️  SubFIFOs estão paradas (não shiftaram desde a última verificação).")

    # Atualiza snapshot
    for nome, sub_fifo in [("A", sub_fifo_A), ("B", sub_fifo_B), ("C", sub_fifo_C)]:
        last_subfifos_snapshot[nome] = {k: list(v) for k, v in sub_fifo.items()}

def get_fifo_matrix(n=32):

    #Retorna uma matriz 22x33, onde:
    #- 1ª coluna: Nome da medida.
   # - 32 colunas seguintes: Primeiros 32 valores da FIFO dessa medida.
    
    #Retorna:
   #     - matrix (np.array): Matriz NumPy 22x33 com os nomes e valores.
   # Se houver menos de 32 valores em alguma medida, retorna None.

    if len(measures_fifos) < 22:
        print("Erro: Existem menos de 22 medidas disponíveis na FIFO.")
        return None

    matrix_data = []

    for name, fifo in list(measures_fifos.items())[:22]:  # Pegamos no máximo 22 medidas
        if len(fifo) < n:
            print(f"Medida '{name}' tem menos de {n} valores armazenados.")
            return None
        
        row = [name] + list(fifo)[:n]  # Primeiro elemento é o nome, seguido dos 32 valores
        matrix_data.append(row)

    return np.array(matrix_data, dtype=object)

def prepare_data_for_inference(n=45, n_measures=21):  # Alterado para 45

   # Retorna uma matriz no formato (45, 22), onde:
  #  - 45 "frames" (grupos)
  #  - Cada frame tem 22 medidas (valores das 22 medidas naquele instante)
    
  #  Retorna:
  #      - matrix (np.array): Matriz NumPy (45, 22) formatada corretamente.
  #  Se houver menos de 45 valores em alguma medida, retorna None.

    if len(measures_fifos) < n_measures:
        print("Erro: Existem menos de 21 medidas disponíveis na FIFO.")
        return None

    # Criar uma lista para armazenar as medidas
    data = []
    measure_names = list(measures_fifos.keys())[:n_measures]  # Pegamos no máximo 22 medidas

    # Verificar se todas as medidas têm pelo menos 45 valores
    for name in measure_names:
        if len(measures_fifos[name]) < n:
            print(f"Erro: Medida '{name}' tem menos de {n} valores armazenados.")
            return None

    # Transformar os valores da FIFO para o novo formato
    for i in range(-n, 0):  # Agora pegamos 45 "frames"
        frame_values = [measures_fifos[name][i] for name in measure_names]  # Pega o valor i de cada medida
        data.append(frame_values)   

    # Transformar em um array NumPy de shape (45, 22)
    matrix = np.array(data, dtype=np.float32)

    # Aplicar MinMaxScaler para normalizar entre [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_matrix = scaler.fit_transform(matrix)

    return normalized_matrix

def infer_emotion(model, fifo_matrix):
 
  #  Faz a inferência no modelo carregado e retorna a classe predita (0, 1, 2) e a probabilidade.
    
   # Parâmetros:
  #      model (tf.keras.Model): Modelo carregado.
  #      fifo_matrix (np.array): Matriz de entrada para inferência com shape (32, 22).
    
  #  Retorna:
  #      predicted_class (int): Classe predita (0 = Happy, 1 = Neutral, 2 = Others).
  #      confidence (float): Probabilidade associada à classe predita.

    if fifo_matrix is None or fifo_matrix.shape != (10, 21):
        print("Erro: Matriz de entrada inválida para inferência.")
        return None, None

    # Expandir dimensão para bater com a entrada do modelo (batch_size = 1)
    fifo_matrix = np.expand_dims(fifo_matrix, axis=0)

    # Fazer a inferência
    predictions = model.predict(fifo_matrix)

    # Obter a classe com maior probabilidade
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    

    probabilities = predictions.flatten()  # Mantém todas as probabilidades

    for i, prob in enumerate(probabilities):
        print(f"Classe {i}: {prob*100:.2f}%")


    return predicted_class, confidence

def correlate_with_reference(measure_name, reference_vector):

  #  Calcula a correlação cruzada entre um vetor de referência e as subFIFOs A, B e C 
 #   da medida especificada.

#    Parâmetros:
 #       measure_name (str): Nome da medida (ex: 'acel_x').
 #       reference_vector (list ou np.array): Vetor de referência a ser comparado.

  #  Retorna:
   #     Um dicionário com os resultados para cada subFIFO:
   #     {
    #        'A': {'max_corr': valor, 'index': idx},
    #        'B': {'max_corr': valor, 'index': idx},
    #        'C': {'max_corr': valor, 'index': idx}
   #     }

    results = {}
    ref = np.array(reference_vector)
    ref_len = len(ref)

    for label, fifo_dict in zip(['A', 'B', 'C'], [sub_fifo_A, sub_fifo_B, sub_fifo_C]):
        if measure_name not in fifo_dict:
            results[label] = {'max_corr': None, 'index': None}
            continue

        fifo_data = np.array(fifo_dict[measure_name])
        
        if len(fifo_data) < ref_len:
            print(f"[AVISO] SubFIFO {label} da medida '{measure_name}' tem menos dados que o vetor de referência.")
            results[label] = {'max_corr': None, 'index': None}
            continue

        # Cross-correlation manual (deslizando o vetor)
        correlations = []
        for i in range(len(fifo_data) - ref_len + 1):
            window = fifo_data[i:i+ref_len]
            corr = np.dot(window, ref)  # produto interno = correlação direta
            correlations.append(corr)

        max_corr = max(correlations)
        best_index = correlations.index(max_corr)

        results[label] = {'max_corr': max_corr, 'index': best_index}

    return results
"""
# Laboratório Prático — Aula 17: Modelo Sequencial no TensorFlow/Keras

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Fashion-MNIST* (70.000 imagens 28x28 de 10 artigos de vestuário)

---

## 🎯 Objetivo do Laboratório
Construir a primeira Rede Neural Profunda no ecossistema **TensorFlow / Keras**: carregar e normalizar imagens de pixels, estruturar o modelo `Sequential` com camadas `Flatten` e `Dense`, compilar com otimizador `adam` e perda `sparse_categorical_crossentropy`, treinar por 10 épocas e avaliar com `evaluate()`.

---

### Passo 1 — Carregando e Normalizando o Fashion-MNIST

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregamos o dataset oficial de imagens de roupas
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Normalizamos os pixels de 0-255 para a escala 0.0 a 1.0
X_train = X_train / 255.0
X_test = X_test / 255.0

nomes_classes = ['Camiseta', 'Calça', 'Pulôver', 'Vestido', 'Casaco', 
                 'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

print(f"Imagens de Treino: {X_train.shape} (60.000 imagens de 28x28)")
print(f"Imagens de Teste:  {X_test.shape}  (10.000 imagens de 28x28)")
```

---

### Passo 2 — Visualizando Exemplos Reais do Dataset

```python
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(nomes_classes[y_train[i]])
    plt.axis('off')
plt.suptitle("5 Primeiras Imagens do Conjunto de Treino", fontsize=14)
plt.show()
```

---

### Passo 3 — Construindo a Arquitetura da Rede Neural Keras

```python
# 1. Montamos o modelo Sequencial
modelo_keras = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),          # Achata 28x28 em vetor de 784
    tf.keras.layers.Dense(128, activation='relu'),          # Camada oculta com 128 neurônios
    tf.keras.layers.Dense(64, activation='relu'),           # Camada oculta com 64 neurônios
    tf.keras.layers.Dense(10, activation='softmax')         # Camada de saída para 10 classes
])

# 2. Exibimos a arquitetura e quantidade de parâmetros
modelo_keras.summary()
```

---

### Passo 4 — Compilação e Treinamento

```python
# 1. Compilamos o modelo
modelo_keras.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 2. Treinamos a rede por 10 épocas
historico = modelo_keras.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
```

---

### Passo 5 — Avaliação Formal nos Dados de Teste

```python
# 1. Avaliamos a performance no conjunto de teste independente
perda_teste, acuracia_teste = modelo_keras.evaluate(X_test, y_test, verbose=0)

print("--- DESEMPENHO NO TESTE ---")
print(f"Loss no Teste:     {perda_teste:.4f}")
print(f"Acurácia no Teste: {acuracia_teste * 100:.2f}%")
```

---

## 🏆 Desafio Técnico de Validação
Selecione a primeira imagem do teste (`X_test[0:1]`), faça a predição com `modelo_keras.predict()` e use `np.argmax()` para descobrir qual peça de roupa a rede identificou.

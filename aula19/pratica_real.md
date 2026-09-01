# Laboratório Prático — Aula 19: Dropout, EarlyStopping & Persistência (.keras)

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Fashion-MNIST* (Regularização em Redes Profundas)

---

## 🎯 Objetivo do Laboratório
Controlar o sobreajuste (*Overfitting*) em Deep Learning: inserir camadas de **Dropout**, configurar **Callbacks (`EarlyStopping` e `ModelCheckpoint`)** para salvar automaticamente a melhor versão da rede, plotar a comparação de curvas `loss` vs `val_loss`, e recarregar o arquivo `.keras` para inferência.

---

### Passo 1 — Carregando os Dados e Preparando Amostra

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Carregamos o dataset Fashion-MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
```

---

### Passo 2 — Construindo a Rede com Camadas Dropout

```python
# 1. Modelo com Dropout (desativa aleatoriamente 30% das conexões a cada rodada)
modelo_regularizado = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Camada de regularização Dropout
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

modelo_regularizado.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### Passo 3 — Configurando Callbacks Automáticos

```python
# 1. EarlyStopping: interrompe o treino se a perda de validação parar de cair por 4 épocas
callback_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

# 2. ModelCheckpoint: salva em disco o arquivo no ponto ótimo
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='melhor_modelo_vestuario.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
```

---

### Passo 4 — Treinamento com Monitoramento Automático

```python
# 1. Treinamos com os callbacks ativos
historico = modelo_regularizado.fit(
    X_train, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[callback_early_stop, callback_checkpoint],
    verbose=1
)
```

---

### Passo 5 — Gráfico de Curvas de Treino vs Validação

```python
# 1. Plotamos a comparação das curvas
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(historico.history['loss'], label='Treino')
plt.plot(historico.history['val_loss'], label='Validação')
plt.title('Curva de Perda (Loss)')
plt.xlabel('Épocas')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historico.history['accuracy'], label='Treino')
plt.plot(historico.history['val_accuracy'], label='Validação')
plt.title('Curva de Acurácia')
plt.xlabel('Épocas')
plt.legend()
plt.grid(True)

plt.show()
```

---

### Passo 6 — Recarregando o Modelo do Disco (`.keras`)

```python
# 1. Carregamos o modelo salvo do disco para comprovação
modelo_recarregado = tf.keras.models.load_model('melhor_modelo_vestuario.keras')

# 2. Avaliamos nos dados de teste
_, acuracia_recarregada = modelo_recarregado.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Acurácia do Modelo Recarregado do Disco: {acuracia_recarregada * 100:.2f}%")
```

---

## 🏆 Desafio Técnico de Validação
Altere a taxa de `Dropout` para `0.5` (50%) e veja se o modelo consegue treinar sem perder acurácia no teste final.

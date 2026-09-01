# Laboratório Prático — Aula 18: Keras para Regressão & Classificação Multiclasse

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *California Housing Dataset* (Regressão Neural) & *Fashion-MNIST* (Multiclasse)

---

## 🎯 Objetivo do Laboratório
Configurar redes neurais Keras para tarefas de **Regressão Contínua**: definir a camada de saída sem ativação (`Dense(1)`), compilar com `loss='mse'` e métrica `mae`, e plotar o gráfico de dispersão entre Valores Reais e Valores Previstos ($y$ vs $\hat{y}$).

---

### Passo 1 — Carregando e Padronizando Dados de Imóveis

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregamos o dataset de imóveis da Califórnia
X, y = fetch_california_housing(return_X_y=True)

# 2. Divisão e padronização
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

---

### Passo 2 — Construindo a Rede Neural Keras de Regressão

```python
# 1. Na Regressão: 1 neurônio na saída e SEM ativação (linear puro)
modelo_regressao = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1) # Saída contínua
])

# 2. Compilação com Mean Squared Error (MSE) e métrica Mean Absolute Error (MAE)
modelo_regressao.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

modelo_regressao.summary()
```

---

### Passo 3 — Treinando a Rede de Regressão

```python
# 1. Treinamos a rede por 30 épocas
historico_reg = modelo_regressao.fit(
    X_train_s, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.15,
    verbose=1
)
```

---

### Passo 4 — Avaliação e Gráfico Real vs Previsto ($y$ vs $\hat{y}$)

```python
# 1. Predição no conjunto de teste
y_pred_reg = modelo_regressao.predict(X_test_s).flatten()

# 2. Plotamos o gráfico de dispersão Real vs Previsto
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_reg, alpha=0.3, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predição Perfeita')
plt.title("Regressão Neural: Preços Reais vs Previstos (California)")
plt.xlabel("Valor Real ($100.000)")
plt.ylabel("Valor Previsto ($100.000)")
plt.legend()
plt.grid(True)
plt.show()

mae_final = np.mean(np.abs(y_test - y_pred_reg))
print(f"Erro Médio Absoluto (MAE) da Rede Neural: ${mae_final * 100000:.2f} dólares")
```

---

## 🏆 Desafio Técnico de Validação
Adicione uma terceira camada oculta com 16 neurônios (`Dense(16, activation='relu')`) e verifique se o MAE no teste melhora.

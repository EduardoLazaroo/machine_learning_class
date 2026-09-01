# Laboratório Prático — Aula 15: O Neurônio Artificial e Funções de Ativação

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Iris Dataset* (Classificação com Perceptron)

---

## 🎯 Objetivo do Laboratório
Implementar a matemática de um neurônio artificial em Python puro vetorizado ($z = \mathbf{X}\mathbf{w} + b$), comparar na prática as saídas das funções **Sigmoide, ReLU e LeakyReLU**, e treinar o algoritmo **Perceptron** do Scikit-Learn em dados biológicos reais.

---

### Passo 1 — Neurônio Artificial Vetorizado em Python Puro

```python
import numpy as np

# 1. Criamos a função matemática do neurônio
def neuronio_artificial(entradas_X, pesos_w, bias_b):
    # Soma ponderada: z = X . w + b (Produto Matricial)
    z = np.dot(entradas_X, pesos_w) + bias_b
    return z

# 2. Testamos com 3 amostras e 2 atributos
X_teste = np.array([[1.0, 2.0], [0.5, -1.5], [-2.0, 3.0]])
w = np.array([0.8, -0.5])
b = 0.2

valores_z = neuronio_artificial(X_teste, w, b)
print("Soma Ponderada (z) calculada para as 3 amostras:")
print(np.round(valores_z, 4))
```

---

### Passo 2 — Implementando Funções de Ativação

```python
# 1. Implementação matemática das funções
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, z * alpha)

print(f"z = -5.0 | Sigmoide: {sigmoide(-5.0):.4f} | ReLU: {relu(-5.0):.4f} | LeakyReLU: {leaky_relu(-5.0):.4f}")
print(f"z = +3.0 | Sigmoide: {sigmoide(3.0):.4f} | ReLU: {relu(3.0):.4f} | LeakyReLU: {leaky_relu(3.0):.4f}")
```

---

### Passo 3 — Treinando o Perceptron no Dataset Iris Real

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Carregamos o dataset Iris (Apenas 2 classes linearmente separáveis)
X, y = load_iris(return_X_y=True)
filtro_binario = y < 2 # Filtra apenas Setosa (0) e Versicolor (1)
X_bin, y_bin = X[filtro_binario], y[filtro_binario]

X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.25, random_state=42)

# 2. Instanciamos e treinamos o Perceptron
modelo_perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=42)
modelo_perceptron.fit(X_train, y_train)

# 3. Avaliamos a acurácia
acuracia = accuracy_score(y_test, modelo_perceptron.predict(X_test))
print(f"🎯 Acurácia do Perceptron em dados biológicos reais: {acuracia * 100:.2f}%")
print(f"Pesos aprendidos (w): {modelo_perceptron.coef_}")
print(f"Bias aprendido (b):   {modelo_perceptron.intercept_}")
```

---

## 🏆 Desafio Técnico de Validação
Tente rodar o Perceptron para classificar as classes `1` e `2` (Versicolor vs Virginica) e observe como a acurácia cai porque essas duas classes não são perfeitamente separáveis por uma única reta linear.

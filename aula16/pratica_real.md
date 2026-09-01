# Laboratório Prático — Aula 16: Redes Neurais MLP & Curvas de Perda

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Breast Cancer Wisconsin Dataset* (Classificação de Biópsias com Redes Multicamadas)

---

## 🎯 Objetivo do Laboratório
Construir e treinar uma Rede Neural Multicamadas com `MLPClassifier` do Scikit-Learn: estruturar camadas ocultas (`hidden_layer_sizes`), analisar o impacto da taxa de aprendizado e do otimizador (`adam` vs `sgd`), e plotar a **Curva de Aprendizado / Queda da Perda (`loss_curve_`)**.

---

### Passo 1 — Preparando os Dados Médicos Padronizados

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Carregamos o dataset de biópsias
X, y = load_breast_cancer(return_X_y=True)

# 2. Divisão estratificada e padronização (obrigatória para redes neurais!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Treino: {X_train_s.shape[0]} amostras | Teste: {X_test_s.shape[0]} amostras")
```

---

### Passo 2 — Treinando o MLPClassifier com 2 Camadas Ocultas

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 1. Instanciamos a rede neural com 2 camadas ocultas: 64 e 32 neurônios
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)

# 2. Treinamos a rede com Backpropagation
modelo_mlp.fit(X_train_s, y_train)

# 3. Avaliamos a performance no conjunto de teste
y_pred = modelo_mlp.predict(X_test_s)
print("--- RELATÓRIO DA REDE NEURAL MLP NO TESTE ---")
print(classification_report(y_test, y_pred, target_names=['Maligno', 'Benigno']))
```

---

### Passo 3 — Plotando a Curva de Convergência do Erro (`loss_curve_`)

```python
# 1. Plotamos a queda do erro época a época
plt.figure(figsize=(8, 4))
plt.plot(modelo_mlp.loss_curve_, color='crimson', lw=2)
plt.title("Curva de Aprendizado — Queda da Função de Custo (Loss)")
plt.xlabel("Épocas de Treinamento (Iterações)")
plt.ylabel("Loss / Erro")
plt.grid(True)
plt.show()

print(f"Loss final atingida: {modelo_mlp.loss_curve_[-1]:.6f}")
```

---

## 🏆 Desafio Técnico de Validação
Altere o solver para `solver='sgd'` e `learning_rate_init=0.0001` (taxa muito baixa) e observe no gráfico como a curva de perda desce muito mais devagar e não atinge a convergência total em 300 épocas.

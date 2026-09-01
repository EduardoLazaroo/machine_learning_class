# Laboratório Prático — Aula 08: Classificação Binária com Regressão Logística & KNN

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Breast Cancer Wisconsin Dataset* (569 biópsias com 30 características nucleares celulares)

---

## 🎯 Objetivo do Laboratório
Construir um pipeline de diagnóstico médico binário (Maligno vs Benigno): padronizar os dados com `StandardScaler`, treinar a **Regressão Logística** e o **KNN**, gerar a **Matriz de Confusão** gráfica e interpretar o trade-off entre **Precisão** e **Recall**.

---

### Passo 1 — Carregando o Dataset Médico de Câncer de Mama

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# 1. Carregamos o dataset oficial do Scikit-Learn
dados_cancer = load_breast_cancer(as_frame=True)
X = dados_cancer.data
y = dados_cancer.target  # 0 = Maligno, 1 = Benigno

print(f"Dimensão da base: {X.shape[0]} pacientes x {X.shape[1]} atributos")
print(f"Distribuição das classes:\n{pd.Series(y).value_counts()}")
```

---

### Passo 2 — Divisão Estratificada e Padronização de Escala

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Divisão treino/teste mantendo a mesma proporção de doentes/saudáveis
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Padronização dos atributos médicos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### Passo 3 — Treinando Regressão Logística vs KNN ($K=5$)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 1. Treinamos a Regressão Logística
modelo_logistico = LogisticRegression(random_state=42)
modelo_logistico.fit(X_train_scaled, y_train)

# 2. Treinamos o KNN com K=5
modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_train_scaled, y_train)

print("Modelos treinados com sucesso!")
```

---

### Passo 4 — Matriz de Confusão e Relatório Completo de Métricas

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Predições no conjunto de teste
y_pred_log = modelo_logistico.predict(X_test_scaled)

# 2. Exibimos o classification report formal
print("--- RELATÓRIO DE CLASSIFICAÇÃO: REGRESSÃO LOGÍSTICA ---")
print(classification_report(y_test, y_pred_log, target_names=dados_cancer.target_names))

# 3. Plotamos a Matriz de Confusão
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_log, 
    display_labels=dados_cancer.target_names, 
    cmap='Blues', 
    ax=ax
)
plt.title("Matriz de Confusão — Diagnóstico de Biópsias")
plt.grid(False)
plt.show()
```

---

## 🏆 Desafio Técnico de Validação
Gere a Matriz de Confusão para o modelo **KNN** e compare: qual dos dois modelos cometeu menos Falsos Negativos (casos malignos diagnosticados erroneamente como benignos)?

# Laboratório Prático — Aula 09: Árvores de Decisão, Random Forest & SVM

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Wine Recognition Dataset* (178 vinhos italianos de 3 cultivares com 13 propriedades químicas)

---

## 🎯 Objetivo do Laboratório
Treinar e comparar três dos mais poderosos classificadores clássicos de Machine Learning: visualizar graficamente as ramificações de uma **Árvore de Decisão**, treinar um ensemble de 100 árvores com **Random Forest**, aplicar **SVM** com Kernel RBF e extrair a importância das propriedades químicas (`feature_importances_`).

---

### Passo 1 — Carregando o Dataset Químico de Vinhos

```python
from sklearn.datasets import load_wine
import pandas as pd

# 1. Carregamos o dataset de vinhos
dados_vinho = load_wine(as_frame=True)
X = dados_vinho.data
y = dados_vinho.target

print(f"Total de vinhos analisados: {X.shape[0]} amostras")
print(f"Atributos químicos: {list(X.columns[:5])}...")
```

---

### Passo 2 — Divisão Treino/Teste e Padronização

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### Passo 3 — Treinando e Visualizando a Árvore de Decisão

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Treinamos a Árvore com profundidade controlada (max_depth=3)
arvore = DecisionTreeClassifier(max_depth=3, random_state=42)
arvore.fit(X_train, y_train) # Árvores não exigem scaling

# 2. Desenhamos a estrutura gráfica da árvore
plt.figure(figsize=(16, 8))
plot_tree(
    arvore, 
    feature_names=dados_vinho.feature_names, 
    class_names=dados_vinho.target_names, 
    filled=True, 
    rounded=True
)
plt.title("Visualização das Decisões da Árvore (max_depth=3)", fontsize=14)
plt.show()
```

---

### Passo 4 — Treinando Random Forest e SVM (Kernel RBF)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Treinamos a Floresta Aleatória com 100 árvores
floresta = RandomForestClassifier(n_estimators=100, random_state=42)
floresta.fit(X_train, y_train)

# 2. Treinamos o Support Vector Machine
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

# 3. Avaliamos a acurácia no teste
print(f"🌳 Acurácia Árvore Única:   {accuracy_score(y_test, arvore.predict(X_test)) * 100:.2f}%")
print(f"🌲 Acurácia Random Forest:  {accuracy_score(y_test, floresta.predict(X_test)) * 100:.2f}%")
print(f"🛡️ Acurácia SVM (RBF):      {accuracy_score(y_test, svm.predict(X_test_scaled)) * 100:.2f}%")
```

---

### Passo 5 — Gráfico de Importância de Atributos (`feature_importances_`)

```python
import seaborn as sns

# 1. Extraímos o peso de cada propriedade química usada pela Random Forest
importancias = pd.Series(floresta.feature_importances_, index=dados_vinho.feature_names).sort_values(ascending=False)

# 2. Plotamos o gráfico de barras
plt.figure(figsize=(10, 5))
sns.barplot(x=importancias.values, y=importancias.index, palette='viridis')
plt.title("Importância dos Atributos Químicos na Identificação do Vinho")
plt.xlabel("Grau de Importância Relativa")
plt.show()
```

---

## 🏆 Desafio Técnico de Validação
Identifique qual é o atributo químico mais decisivo do vinho segundo o gráfico de importância e teste treinar uma Random Forest utilizando apenas os 3 atributos do topo.

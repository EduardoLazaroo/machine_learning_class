# Laboratório Prático — Aula 13: DBSCAN & Agrupamento Hierárquico

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real/Sintético:** *Two Moons Dataset* com ruído & *Iris Dataset* (Dendrogramas)

---

## 🎯 Objetivo do Laboratório
Demonstrar na prática a falha do K-Means em dados de geometria não-convexa e o sucesso do **DBSCAN** baseado em densidade (`eps` e `min_samples`), identificar ruídos/outliers marcados com `-1` e gerar o **Dendrograma Hierárquico Aglomerativo**.

---

### Passo 1 — Gerando Dados em Meia-Lua com Ruído

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Geramos 500 pontos em formato entrelaçado de meias-luas
X_moons, _ = make_moons(n_samples=500, noise=0.08, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

plt.figure(figsize=(6, 4))
plt.scatter(X_moons[:, 0], X_moons[:, 1], s=20, color='gray')
plt.title("Dados em Meia-Lua (Não-Lineares)")
plt.show()
```

---

### Passo 2 — K-Means (Falha) vs DBSCAN (Sucesso)

```python
from sklearn.cluster import KMeans, DBSCAN

# 1. K-Means (Assume clusters circulares)
labels_km = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_moons)

# 2. DBSCAN (Agrupa por continuidade de densidade)
labels_db = DBSCAN(eps=0.25, min_samples=5).fit_predict(X_moons)

# 3. Plotamos a comparação lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_km, cmap='viridis', s=20)
ax1.set_title("K-Means (Corta os grupos ao meio)")

ax2.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db, cmap='coolwarm', s=20)
ax2.set_title("DBSCAN (Identifica o formato real)")
plt.show()

print(f"Total de ruídos / outliers identificados pelo DBSCAN: {sum(labels_db == -1)}")
```

---

### Passo 3 — Gerando o Dendrograma no Dataset Iris

```python
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Carregamos uma amostra de 30 flores do Iris
X_iris, _ = load_iris(return_X_y=True)
X_amostra = X_iris[:30]

# 2. Calculamos a matriz de conexões (Linkage com método Ward)
matriz_conexoes = linkage(X_amostra, method='ward')

# 3. Plotamos o Dendrograma
plt.figure(figsize=(10, 5))
dendrogram(matriz_conexoes, truncate_mode='lastp', p=15, show_leaf_counts=True)
plt.title("Dendrograma — Agrupamento Hierárquico Aglomerativo")
plt.xlabel("Índice das Amostras")
plt.ylabel("Distância Euclidiana de Fusão")
plt.show()
```

---

## 🏆 Desafio Técnico de Validação
Altere o parâmetro `eps` do DBSCAN para `0.10` e observe como o número de pontos considerados ruído (`label == -1`) aumenta drasticamente por causa do raio muito restrito.

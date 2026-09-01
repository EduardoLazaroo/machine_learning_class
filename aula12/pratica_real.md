# Laboratório Prático — Aula 12: Clusterização K-Means e Seleção do K Ideal

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Wine Recognition Dataset* (Dados químicos sem rótulos supervisionados)

---

## 🎯 Objetivo do Laboratório
Realizar a segmentação e descoberta de grupos naturais em dados multidimensionais: aplicar padronização obrigatória com `StandardScaler`, calcular a Inércia (WCSS) para $K=1..8$, plotar o **Gráfico do Cotovelo (*Elbow Plot*)**, calcular o **Coeficiente de Silhueta** e visualizar a distribuição dos clusters e centroides.

---

### Passo 1 — Preparando Dados Reais sem Gabarito ($y$)

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Carregamos o dataset e ignoramos totalmente os rótulos de classe
dados_vinho = load_wine(as_frame=True)
X_bruto = dados_vinho.data

# 2. A padronização é obrigatória para que variáveis de escalas grandes não dominem a distância
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bruto)

print(f"Matriz de dados para agrupamento não supervisionado: {X_scaled.shape}")
```

---

### Passo 2 — Cálculo do Método do Cotovelo (*Elbow Method*)

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inercia = []
valores_k = range(1, 9)

# 1. Calculamos o WCSS (Soma dos Quadrados Intra-Cluster) de 1 a 8 grupos
for k in valores_k:
    kmeans_teste = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_teste.fit(X_scaled)
    inercia.append(kmeans_teste.inertia_)

# 2. Plotamos o Gráfico do Cotovelo
plt.figure(figsize=(8, 4))
plt.plot(valores_k, inercia, 'bo-', linewidth=2, markersize=8)
plt.title("Método do Cotovelo — Escolha do Número Ideal de Clusters (K)")
plt.xlabel("Quantidade de Clusters (K)")
plt.ylabel("Inércia / WCSS")
plt.grid(True)
plt.show()
```

---

### Passo 3 — Validação com Coeficiente de Silhueta

```python
from sklearn.metrics import silhouette_score

# 1. Calculamos a silhueta para K de 2 a 6
for k in range(2, 7):
    modelo_km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo_km.fit_predict(X_scaled)
    score_silhueta = silhouette_score(X_scaled, labels)
    print(f"K = {k} | Coeficiente de Silhueta Médio: {score_silhueta:.4f}")
```

---

### Passo 4 — Treinamento Final e Inspeção dos Clusters

```python
# 1. Executamos o K-Means com K=3
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
grupos_atribuidos = kmeans_final.fit_predict(X_scaled)

# 2. Contamos quantos vinhos foram alocados em cada cluster
df_resultado = pd.DataFrame({'Cluster': grupos_atribuidos})
print("--- DISTRIBUIÇÃO DOS VINHOS POR CLUSTER ---")
print(df_resultado['Cluster'].value_counts())
```

---

## 🏆 Desafio Técnico de Validação
Verifique a matriz de centroides (`kmeans_final.cluster_centers_`) e descubra qual cluster possui o maior valor médio padronizado para o primeiro atributo químico (`alcohol`).

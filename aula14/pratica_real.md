# Laboratório Prático — Aula 14: Redução de Dimensionalidade com PCA

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Optical Recognition of Handwritten Digits* (1.797 imagens de dígitos 8x8 = 64 dimensões)

---

## 🎯 Objetivo do Laboratório
Comprimir um espaço de 64 dimensões (pixels) em apenas 2 eixos principais utilizando **PCA (Principal Component Analysis)**: calcular a razão de variância explicada, plotar a dispersão 2D dos 10 dígitos e construir o gráfico de variância acumulada (*Scree Plot*).

---

### Passo 1 — Carregando o Dataset de Dígitos (64 Dimensões)

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregamos 1.797 imagens de dígitos manuscritos de 0 a 9
dados_digitos = load_digits()
X = dados_digitos.data  # 1.797 linhas x 64 pixels (dimensões)
y = dados_digitos.target

print(f"Dimensão da base original: {X.shape[0]} imagens x {X.shape[1]} dimensões (pixels)")

# 2. Padronizamos os dados
X_scaled = StandardScaler().fit_transform(X)
```

---

### Passo 2 — Redução para 2 Componentes Principais (`PCA(n_components=2)`)

```python
from sklearn.decomposition import PCA

# 1. Instanciamos e treinamos o PCA para extrair 2 dimensões
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# 2. Extraímos o percentual de informação preservada
variancia = pca_2d.explained_variance_ratio_
print(f"Variância explicada pelo CP1: {variancia[0] * 100:.2f}%")
print(f"Variância explicada pelo CP2: {variancia[1] * 100:.2f}%")
print(f"Total de informação preservada em 2D: {variancia.sum() * 100:.2f}%")
```

---

### Passo 3 — Visualizando o Dataset de 64D Projetado no Plano 2D

```python
# 1. Plotamos a projeção 2D com cores para cada um dos 10 dígitos (0 a 9)
plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    X_pca_2d[:, 0], 
    X_pca_2d[:, 1], 
    c=y, 
    cmap='tab10', 
    alpha=0.7, 
    s=25
)
plt.colorbar(scatter, label='Dígito Verdadeiro (0 a 9)')
plt.title("Projeção PCA 2D do Dataset de Dígitos (64D ──► 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
```

---

### Passo 4 — Gráfico da Variância Explicada Acumulada (*Scree Plot*)

```python
# 1. Treinamos o PCA com 30 componentes para ver a curva de saturação
pca_30 = PCA(n_components=30)
pca_30.fit(X_scaled)
variancia_acumulada = np.cumsum(pca_30.explained_variance_ratio_) * 100

# 2. Plotamos a curva
plt.figure(figsize=(8, 4))
plt.plot(range(1, 31), variancia_acumulada, 'ro-', linewidth=2)
plt.axhline(y=80, color='b', linestyle='--', label='80% de Variância')
plt.title("Variância Explicada Acumulada por Número de Componentes")
plt.xlabel("Número de Componentes Principais")
plt.ylabel("Variância Total Retida (%)")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🏆 Desafio Técnico de Validação
Descubra no gráfico quantas componentes principais são necessárias para reter pelo menos **$80\%$** de toda a informação original dos dígitos.

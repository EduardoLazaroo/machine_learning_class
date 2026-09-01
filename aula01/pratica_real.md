# Laboratório Prático — Aula 01: Primeiro Pipeline de Machine Learning

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Palmer Penguins Dataset* (344 pinguins do arquipélago Palmer, Antártica)

---

## 🎯 Objetivo do Laboratório
Executar um pipeline completo de Machine Learning no Google Colab utilizando dados biológicos reais: carregar a base de pinguins, separar atributos de medição física, dividir os dados e treinar um classificador para identificar a espécie do pinguim com base em suas características.

---

### Passo 1 — Carregando o Dataset Real com Seaborn

```python
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Carregamos o dataset oficial de pinguins do Seaborn
df_pinguins = sns.load_dataset('penguins')

# 2. Inspecionamos as 5 primeiras linhas e a dimensão
print("--- PRIMEIRAS LINHAS DO DATASET REAL ---")
print(df_pinguins.head())
print(f"\n📐 Dimensões da base: {df_pinguins.shape[0]} linhas x {df_pinguins.shape[1]} colunas")
print(f"🐧 Espécies disponíveis: {df_pinguins['species'].unique()}")
```

---

### Passo 2 — Limpeza Rápida de Valores Nulos

```python
# 1. Removemos registros que possuem medições ausentes
df_limpo = df_pinguins.dropna().copy()

print(f"Linhas antes da limpeza: {len(df_pinguins)}")
print(f"Linhas após a limpeza:   {len(df_limpo)}")
```

---

### Passo 3 — Separação de Entradas ($X$) e Alvo ($y$)

```python
# 1. Selecionamos as 4 variáveis numéricas morfológicas como entradas (X)
atributos = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df_limpo[atributos]

# 2. Selecionamos a espécie como alvo a ser previsto (y)
y = df_limpo['species']

print("--- MATRIZ DE ENTRADA X (Pistas) ---")
print(X.head())
print("\n--- VETOR ALVO y (Respostas) ---")
print(y.head())
```

---

### Passo 4 — Divisão dos Dados em Treino e Teste

```python
from sklearn.model_selection import train_test_split

# 1. Separamos 75% para treino do modelo e 25% para teste de validação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Amostras para treino: {len(X_train)}")
print(f"Amostras para teste:  {len(X_test)}")
```

---

### Passo 5 — Treinamento do Modelo KNN e Avaliação

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Instanciamos o algoritmo KNN (K-Vizinhos Mais Próximos com K=5)
modelo_knn = KNeighborsClassifier(n_neighbors=5)

# 2. Treinamos o modelo com os dados de treino
modelo_knn.fit(X_train, y_train)

# 3. Fazemos previsões no conjunto de teste que o modelo nunca viu
previsoes = modelo_knn.predict(X_test)

# 4. Calculamos a taxa de acerto real
acuracia = accuracy_score(y_test, previsoes)
print(f"🎯 Acurácia do modelo no teste: {acuracia * 100:.2f}%")
```

---

### Passo 6 — Classificando um Novo Pinguim Encontrado na Antártica

```python
# 1. Criamos um novo pinguim fictício com suas medições
# [comprimento_bico, profundidade_bico, comprimento_nadadeira, massa_corporal]
novo_pinguim = pd.DataFrame([[50.0, 15.0, 220.0, 5200.0]], columns=atributos)

# 2. O modelo identifica a espécie automaticamente
especie_prevista = modelo_knn.predict(novo_pinguim)[0]
print(f"🔍 Espécie identificada pelo modelo: >>> {especie_prevista.upper()} <<<")
```

---

## 🏆 Desafio Técnico de Validação
Altere o parâmetro `n_neighbors` para `1`, `3`, `7` e `15`. Anote no console qual valor de $K$ produziu a maior acurácia no conjunto de teste.

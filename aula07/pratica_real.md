# Laboratório Prático — Aula 07: Regressão Linear Simples e Múltipla

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *California Housing Dataset* (20.640 distritos censitários da Califórnia)

---

## 🎯 Objetivo do Laboratório
Treinar um modelo de Regressão Linear Múltipla para prever o valor mediano de imóveis ($y$) a partir de variáveis econômicas e estruturais ($X$): realizar divisão treino/teste com `train_test_split`, treinar o regressor com `fit`, interpretar os coeficientes aprendidos e mensurar o erro com $R^2$, MAE e RMSE.

---

### Passo 1 — Carregando os Dados Imobiliários Reais

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# 1. Carregamos o dataset oficial em formato DataFrame
dados_california = fetch_california_housing(as_frame=True)
df_casas = dados_california.frame

print("--- 5 PRIMEIRAS LINHAS DOS IMÓVEIS ---")
print(df_casas.head())
print(f"\nTotal de distritos imobiliários: {df_casas.shape[0]}")
```

---

### Passo 2 — Definição das Matrizes $X$ e Vetor Alvo $y$

```python
# 1. Entradas: Renda Mediana (MedInc), Idade da Casa (HouseAge) e Cômodos (AveRooms)
atributos_selecionados = ['MedInc', 'HouseAge', 'AveRooms']
X = df_casas[atributos_selecionados]

# 2. Alvo: Valor Mediano da Casa em centenas de milhares de dólares (MedHouseVal)
y = df_casas['MedHouseVal']

print(f"Dimensão da Matriz X: {X.shape}")
print(f"Dimensão do Alvo y:   {y.shape}")
```

---

### Passo 3 — Divisão em Treino (80%) e Teste (20%)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Casas para Treino: {len(X_train)}")
print(f"Casas para Teste:  {len(X_test)}")
```

---

### Passo 4 — Treinamento do Modelo e Inspeção de Coeficientes

```python
from sklearn.linear_model import LinearRegression

# 1. Instanciamos e treinamos o modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 2. Exibimos os coeficientes da equação: y = b0 + b1*MedInc + b2*HouseAge + b3*AveRooms
print(f"Intercepto (b0): {regressor.intercept_:.4f}")
for nome_col, peso in zip(atributos_selecionados, regressor.coef_):
    print(f"Coeficiente ({nome_col}): {peso:+.4f}")
```

---

### Passo 5 — Predição e Avaliação Formal de Métricas ($R^2$, MAE, RMSE)

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Fazemos as predições no conjunto de teste
y_pred = regressor.predict(X_test)

# 2. Calculamos as métricas de performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("--- MÉTRICAS NO CONJUNTO DE TESTE ---")
print(f"R² (Coeficiente de Determinação): {r2 * 100:.2f}%")
print(f"MAE (Erro Médio Absoluto):        ${mae * 100000:.2f} dólares")
print(f"RMSE (Raiz do Erro Quadrático):   ${rmse * 100000:.2f} dólares")
```

---

## 🏆 Desafio Técnico de Validação
Treine um segundo modelo utilizando **todas as 8 colunas** do dataset original (`X = df_casas.drop(columns=['MedHouseVal'])`) e verifique se o $R^2$ do modelo sobe para acima de $60\%$.

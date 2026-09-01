# Laboratório Prático — Aula 11: Otimização de Hiperparâmetros & Regularização

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Datasets Reais:** *Wine Dataset* (GridSearch de Classificação) & *Diabetes Dataset* (Regularização L1/L2)

---

## 🎯 Objetivo do Laboratório
Automatizar a sintonia fina de modelos utilizando **GridSearchCV** em 5-Fold, explorar os rankings de busca no `cv_results_`, e comparar o impacto prático da Regularização **Ridge (L2)** versus **Lasso (L1)** na contração e anulação de coeficientes.

---

### Passo 1 — Otimização Automática em Grade (`GridSearchCV`)

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

# 1. Carregamos o dataset de vinhos
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Definimos a grade de combinações a serem testadas
grade_parametros = {
    'n_estimators': [20, 50, 100],
    'max_depth': [2, 3, 5, None],
    'min_samples_split': [2, 5]
}

# 3. Executamos o GridSearchCV com 5 folds
busca_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=grade_parametros,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
busca_grid.fit(X_train, y_train)

print(f"🏆 Melhores Hiperparâmetros: {busca_grid.best_params_}")
print(f"⭐ Melhor Score Médio de CV: {busca_grid.best_score_ * 100:.2f}%")
print(f"🎯 Acurácia no Teste Final:  {busca_grid.score(X_test, y_test) * 100:.2f}%")
```

---

### Passo 2 — Regularização L2 (Ridge) vs L1 (Lasso)

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Carregamos o dataset de progressão do diabetes
X_diab, y_diab = load_diabetes(return_X_y=True)
scaler = StandardScaler()
X_diab_scaled = scaler.fit_transform(X_diab)

# 2. Treinamos Regressão Pura, Ridge (L2) e Lasso (L1)
reg_linear = LinearRegression().fit(X_diab_scaled, y_diab)
reg_ridge = Ridge(alpha=10.0).fit(X_diab_scaled, y_diab)
reg_lasso = Lasso(alpha=2.0).fit(X_diab_scaled, y_diab)

# 3. Comparamos os coeficientes aprendidos
tabela_coeficientes = pd.DataFrame({
    'Linear_Pura': reg_linear.coef_,
    'Ridge_L2': reg_ridge.coef_,
    'Lasso_L1': reg_lasso.coef_
}).round(2)

print("--- IMPACTO DA REGULARIZAÇÃO NOS COEFICIENTES ---")
print(tabela_coeficientes)
print(f"\nTotal de coeficientes ZERADOS pelo Lasso: {np.sum(reg_lasso.coef_ == 0)}")
```

---

## 🏆 Desafio Técnico de Validação
Aumente o parâmetro `alpha` do Lasso para `10.0` e veja quantas variáveis a regularização L1 zera automaticamente, atuando como um selecionador de atributos.

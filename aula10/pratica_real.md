# Laboratório Prático — Aula 10: Validação Cruzada Estratificada e Curvas ROC/AUC

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Breast Cancer Wisconsin Dataset* (Classificação probabilística de tumores)

---

## 🎯 Objetivo do Laboratório
Garantir a avaliação estatisticamente robusta de modelos preditivos: executar **Validação Cruzada 5-Fold Stratified**, extrair probabilidades reais com `.predict_proba()`, calcular as taxas de Falso Positivo e Verdadeiro Positivo com `roc_curve` e plotar o gráfico comparativo da **Curva ROC** com cálculo de **AUC**.

---

### Passo 1 — Validação Cruzada K-Fold com Múltiplos Folds

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# 1. Carregamos os dados médicos
X, y = load_breast_cancer(return_X_y=True)

# 2. Criamos o pipeline com Scaler + Regressão Logística para evitar data leakage
pipeline_lr = make_pipeline(StandardScaler(), LogisticRegression())

# 3. Executamos 5-Fold Cross Validation estratificado
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_lr = cross_val_score(pipeline_lr, X, y, cv=kfold, scoring='accuracy')

print("--- RESULTADOS DA VALIDAÇÃO CRUZADA (5 FOLDS) ---")
print(f"Notas de cada fold: {np.round(scores_lr * 100, 2)}")
print(f"Acurácia Média:     {scores_lr.mean() * 100:.2f}% (± {scores_lr.std() * 100:.2f}%)")
```

---

### Passo 2 — Extraindo Probabilidades Contínuas no Teste

```python
from sklearn.model_selection import train_test_split

# 1. Divisão simples para cálculo da Curva ROC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

modelo_lr = LogisticRegression()
modelo_lr.fit(X_train_s, y_train)

# 2. Extraímos as probabilidades da classe 1 (Benigno)
probabilidades = modelo_lr.predict_proba(X_test_s)[:, 1]
print("5 primeiras probabilidades estimadas pelo modelo:")
print(np.round(probabilidades[:5], 4))
```

---

### Passo 3 — Plotagem da Curva ROC e Cálculo da AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Calculamos Taxa de Falsos Positivos (FPR) e Verdadeiros Positivos (TPR)
fpr, tpr, limiares = roc_curve(y_test, probabilidades)
auc_score = roc_auc_score(y_test, probabilidades)

# 2. Plotamos a Curva ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Regressão Logística (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)')
plt.title('Curva ROC — Avaliação de Capacidade Preditiva')
plt.legend(loc="lower right")
plt.show()
```

---

## 🏆 Desafio Técnico de Validação
Treine um modelo `RandomForestClassifier` no mesmo conjunto e plote a Curva ROC dele na mesma figura para descobrir qual modelo alcançou maior pontuação AUC.

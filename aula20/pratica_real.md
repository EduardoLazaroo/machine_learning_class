# Laboratório Prático — Aula 20: Projeto Integrador End-to-End

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Titanic Dataset* (Pipeline Completo: ML Clássico vs Deep Learning)

---

## 🎯 Objetivo do Laboratório
Executar o ciclo de vida completo de uma aplicação de Inteligência Artificial: Carregamento $\rightarrow$ Limpeza e Imputação $\rightarrow$ Engenharia de Atributos $\rightarrow$ One-Hot Encoding e Scaling $\rightarrow$ Treinamento e Comparação de **Random Forest** vs **Rede Neural Keras** $\rightarrow$ Construção da Função de Inferência em Produção.

---

### Passo 1 — Carga e Limpeza Completa dos Dados

```python
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Carregamos o dataset real do Titanic
df_raw = sns.load_dataset('titanic')

# 2. Descarte de colunas redundantes ou com excesso de nulos
df_tratado = df_raw.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male']).copy()

# 3. Imputação de nulos
df_tratado['age'] = df_tratado['age'].fillna(df_tratado.groupby('pclass')['age'].transform('median'))
df_tratado['embarked'] = df_tratado['embarked'].fillna(df_tratado['embarked'].mode()[0])

# 4. Engenharia de Recursos: Tamanho da Família
df_tratado['tamanho_familia'] = df_tratado['sibsp'] + df_tratado['parch'] + 1
df_tratado['viaja_sozinho'] = (df_tratado['tamanho_familia'] == 1).astype(int)

print(f"Dataset pronto: {df_tratado.shape[0]} passageiros limpos!")
```

---

### Passo 2 — Encoding, Scaling e Divisão Treino/Teste

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. One-Hot Encoding em variáveis categóricas
df_encoded = pd.get_dummies(df_tratado, columns=['sex', 'embarked'], drop_first=True, dtype=int)

# 2. Separação X e y
X = df_encoded.drop(columns=['survived'])
y = df_encoded['survived']

# 3. Divisão Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 4. Padronização
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

---

### Passo 3 — Treinando o Modelo Clássico: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Treinamos a Random Forest com 100 árvores
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train) # Árvores usam X_train direto

y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"🌲 Random Forest — Acurácia: {acc_rf * 100:.2f}% | F1-Score: {f1_rf * 100:.2f}%")
```

---

### Passo 4 — Treinando o Modelo Deep Learning: Rede Neural Keras

```python
import tensorflow as tf

# 1. Montamos a rede neural sequencial
dl_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_s.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Saída binária
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 2. Treinamos por 40 épocas
dl_model.fit(X_train_s, y_train, epochs=40, batch_size=32, verbose=0)

# 3. Avaliação no teste
y_pred_prob = dl_model.predict(X_test_s, verbose=0).flatten()
y_pred_dl = (y_pred_prob >= 0.5).astype(int)
acc_dl = accuracy_score(y_test, y_pred_dl)
f1_dl = f1_score(y_test, y_pred_dl)

print(f"🧠 Rede Neural DL — Acurácia: {acc_dl * 100:.2f}% | F1-Score: {f1_dl * 100:.2f}%")
```

---

### Passo 5 — Simulação de Produção: Função de Inferência com Novo Passageiro

```python
# 1. Criamos a função que recebe dados de um novo passageiro e retorna a previsão
def prever_sobrevivencia(pclass, age, sibsp, parch, fare, alone, family_size, sex_male, embarked_Q, embarked_S):
    dados_novo = pd.DataFrame([[pclass, age, sibsp, parch, fare, alone, family_size, sex_male, embarked_Q, embarked_S]], columns=X.columns)
    dados_novo_s = scaler.transform(dados_novo)
    
    probabilidade = dl_model.predict(dados_novo_s, verbose=0)[0][0]
    sobreviveu = "SIM (Sobrevivente)" if probabilidade >= 0.5 else "NÃO (Vítima)"
    
    print(f"\n🔮 PREVISÃO DE PRODUÇÃO:")
    print(f"Chance estimada de sobrevivência: {probabilidade * 100:.2f}%")
    print(f"Resultado: {sobreviveu}")

# 2. Testamos com um passageiro: 1ª Classe, Homem de 25 anos, viajou com tarifa cara
prever_sobrevivencia(pclass=1, age=25, sibsp=0, parch=0, fare=150.0, alone=1, family_size=1, sex_male=1, embarked_Q=0, embarked_S=1)
```

---

## 🏆 Desafio Técnico Final
Teste a função de inferência inserindo os seus próprios dados fictícios (sua idade, classe escolhida) e descubra qual seria o seu destino simulado no Titanic!

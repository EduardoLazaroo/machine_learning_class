# Laboratório Prático — Aula 04: Higienização de Dados e Tratamento de Outliers

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Titanic Dataset* (891 passageiros com falhas reais de cadastro, valores nulos e tarifas discrepantes)

---

## 🎯 Objetivo do Laboratório
Executar o diagnóstico e a limpeza de uma base de dados real: mapear percentuais de nulos por coluna, aplicar imputação condicional de valores ausentes pela mediana, remover colunas com comprometimento severo e identificar/filtrar outliers de tarifas usando a Regra do Intervalo Interquartil (IQR).

---

### Passo 1 — Diagnóstico Percentual de Dados Faltantes (`NaN`)

```python
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Carregamos o dataset real do Titanic
df_titanic = sns.load_dataset('titanic')

# 2. Calculamos a quantidade e o percentual exato de nulos por coluna
relatorio_nulos = pd.DataFrame({
    'Total_Nulos': df_titanic.isnull().sum(),
    'Percentual_%': (df_titanic.isnull().sum() / len(df_titanic) * 100).round(2)
})

print("--- RELATÓRIO DE DADOS AUSENTES ---")
print(relatorio_nulos[relatorio_nulos['Total_Nulos'] > 0])
```

---

### Passo 2 — Tomada de Decisão: Descarte vs Imputação

```python
# 1. Coluna 'deck' tem mais de 77% de nulos -> Descartamos a coluna inteira
df_limpo = df_titanic.drop(columns=['deck']).copy()

# 2. Coluna 'embarked' tem apenas 2 nulos -> Preenchemos com a Moda (porto mais frequente)
porto_mais_comum = df_limpo['embarked'].mode()[0]
df_limpo['embarked'] = df_limpo['embarked'].fillna(porto_mais_comum)

# 3. Coluna 'age' tem 19.8% de nulos -> Imputamos pela Mediana de idade por Classe
mediana_idade_por_classe = df_limpo.groupby('pclass')['age'].transform('median')
df_limpo['age'] = df_limpo['age'].fillna(mediana_idade_por_classe)

print(f"Nulos restantes em 'age': {df_limpo['age'].isnull().sum()}")
print(f"Nulos restantes em 'embarked': {df_limpo['embarked'].isnull().sum()}")
```

---

### Passo 3 — Detecção de Outliers na Tarifa (`fare`) via Regra do IQR

```python
# 1. Calculamos o Primeiro Quartil (Q1) e Terceiro Quartil (Q3)
Q1 = df_limpo['fare'].quantile(0.25)
Q3 = df_limpo['fare'].quantile(0.75)
IQR = Q3 - Q1

# 2. Definimos os limites inferior e superior para outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Q1 (25%): {Q1:.2f} | Q3 (75%): {Q3:.2f} | IQR: {IQR:.2f}")
print(f"Limite de Corte Superior para Outliers: R$ {limite_superior:.2f}")

# 3. Contamos quantos passageiros pagaram tarifas consideradas aberrantes
outliers_tarifa = df_limpo[df_limpo['fare'] > limite_superior]
print(f"Total de registros identificados como Outliers de Tarifa: {len(outliers_tarifa)} ({len(outliers_tarifa)/len(df_limpo)*100:.2f}%)")
```

---

### Passo 4 — Filtragem e Comparação Estatística

```python
# 1. Geramos o dataset final sem os outliers de tarifa extrema
df_sem_outliers = df_limpo[df_limpo['fare'] <= limite_superior].copy()

print("--- COMPARAÇÃO DA MÉDIA E DESVIO PADRÃO ---")
print(f"Média original da Tarifa:     R$ {df_limpo['fare'].mean():.2f} (Desvio: {df_limpo['fare'].std():.2f})")
print(f"Média sem Outliers da Tarifa: R$ {df_sem_outliers['fare'].mean():.2f} (Desvio: {df_sem_outliers['fare'].std():.2f})")
```

---

## 🏆 Desafio Técnico de Validação
Verifique se a variável `age` (idade) contém outliers após a imputação aplicando a mesma fórmula do IQR (`Q3 + 1.5 * IQR`).

# Laboratório Prático — Aula 05: Pré-processamento, Encoding & Escalonamento

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Diamonds Dataset* (53.940 diamantes com atributos nominais, ordinais e contínuos de alta variância)

---

## 🎯 Objetivo do Laboratório
Aplicar técnicas industriais de pré-processamento de dados em uma base massiva de diamantes: Mapeamento Ordinal explícito (`cut`), One-Hot Encoding para categorias nominais (`color`), e comparação prática entre `StandardScaler` (Z-Score) e `MinMaxScaler` ($0$ a $1$) nas variáveis de preço e quilates.

---

### Passo 1 — Carregando o Dataset Real de Diamantes

```python
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Carregamos o dataset de 53.940 diamantes
df_diamantes = sns.load_dataset('diamonds')

# 2. Amostramos 1.000 linhas para processamento ágil no laboratório
df_lab = df_diamantes.sample(n=1000, random_state=42).copy()

print("--- ESTRUTURA DOS DADOS HETEROGÊNEOS ---")
print(df_lab[['carat', 'cut', 'color', 'clarity', 'price']].head())
```

---

### Passo 2 — Mapeamento Ordinal Explícito (`cut`)

```python
# 1. Como a qualidade da lapidação possui ordem hierárquica clara, usamos mapeamento ordinal
mapa_lapidacao = {
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Premium': 4,
    'Ideal': 5
}

# 2. Aplicamos a conversão para número inteiro
df_lab['lapidacao_ordinal'] = df_lab['cut'].map(mapa_lapidacao)

print("--- COLUNA LAPIDAÇÃO APÓS MAPEAMENTO ORDINAL ---")
print(df_lab[['cut', 'lapidacao_ordinal']].head())
```

---

### Passo 3 — Codificação Nominal via One-Hot Encoding (`pd.get_dummies`)

```python
# 1. Aplicamos One-Hot Encoding na cor do diamante (D, E, F, G, H, I, J)
# drop_first=True elimina 1 coluna para evitar a armadilha da multicolinearidade
df_encoded = pd.get_dummies(df_lab, columns=['color'], drop_first=True, dtype=int)

print("--- COLUNAS GERADAS PELO ONE-HOT ENCODING ---")
colunas_cor = [c for c in df_encoded.columns if c.startswith('color_')]
print(df_encoded[colunas_cor].head())
```

---

### Passo 4 — Escalonamento Numérico: StandardScaler vs MinMaxScaler

```python
# 1. Selecionamos os atributos numéricos brutos com escalas discrepantes
# carat varia de 0.2 a 3.0 | price varia de R$ 300 a R$ 18.000
atributos_numericos = df_lab[['carat', 'price']]

# 2. Instanciamos e aplicamos o StandardScaler (Média = 0, Desvio Padrão = 1)
scaler_standard = StandardScaler()
dados_standard = scaler_standard.fit_transform(atributos_numericos)
df_standard = pd.DataFrame(dados_standard, columns=['carat_std', 'price_std'])

# 3. Instanciamos e aplicamos o MinMaxScaler (Faixa estrita entre 0 e 1)
scaler_minmax = MinMaxScaler()
dados_minmax = scaler_minmax.fit_transform(atributos_numericos)
df_minmax = pd.DataFrame(dados_minmax, columns=['carat_minmax', 'price_minmax'])

print("--- COMPARAÇÃO DE ESCALONADORES ---")
print("StandardScaler (Z-Score):")
print(df_standard.describe().round(3).loc[['mean', 'std', 'min', 'max']])
print("\nMinMaxScaler (0 a 1):")
print(df_minmax.describe().round(3).loc[['mean', 'std', 'min', 'max']])
```

---

## 🏆 Desafio Técnico de Validação
Aplique o mapeamento ordinal na coluna `clarity` respeitando a ordem oficial de pureza: `{'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}` e confira se a média de pureza foi calculada com sucesso.

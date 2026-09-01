# Laboratório Prático — Aula 03: Exploração Analítica de Tabelas com Pandas

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Restaurant Tips Dataset* (244 contas de restaurante com valor total, gorjeta, gênero, dia e horário)

---

## 🎯 Objetivo do Laboratório
Dominar a manipulação de DataFrames com dados de consumo reais: inspeção estrutural e estatística, seleção via `loc`/`iloc`, filtros lógicos compostos, criação de métricas calculadas e agregações multi-coluna com `groupby`.

---

### Passo 1 — Carregando e Inspecionando o Dataset

```python
import seaborn as sns
import pandas as pd

# 1. Carregamos a base de gorjetas de restaurante
df_tips = sns.load_dataset('tips')

# 2. Inspecionamos estrutura e tipos das variáveis
print("--- INFORMAÇÕES ESTRUTURAIS ---")
df_tips.info()

print("\n--- RESUMO ESTATÍSTICO DE COLUNAS NUMÉRICAS ---")
print(df_tips.describe().round(2))
```

---

### Passo 2 — Engenharia de Atributo: Percentual de Gorjeta

```python
# 1. Calculamos o percentual exato da gorjeta em relação ao total da conta
df_tips['pct_gorjeta'] = (df_tips['tip'] / df_tips['total_bill']) * 100

print("--- PRIMEIRAS LINHAS COM A NOVA COLUNA ---")
print(df_tips[['total_bill', 'tip', 'pct_gorjeta', 'day', 'time']].head())
```

---

### Passo 3 — Filtros Lógicos Compostos

```python
# 1. Filtro: Contas de Jantar no Sábado ou Domingo com mais de 3 pessoas na mesa
filtro_fim_de_semana = (
    (df_tips['day'].isin(['Sat', 'Sun'])) & 
    (df_tips['time'] == 'Dinner') & 
    (df_tips['size'] >= 4)
)

mesas_grandes_fds = df_tips[filtro_fim_de_semana]
print(f"Total de mesas grandes no fim de semana à noite: {len(mesas_grandes_fds)}")
print(mesas_grandes_fds[['total_bill', 'tip', 'pct_gorjeta', 'size']].head())
```

---

### Passo 4 — Agregações Analíticas com `groupby`

```python
# 1. Agrupamos por dia e horário calculando média, mediana e contagem
resumo_por_dia = df_tips.groupby(['day', 'time'], observed=True).agg(
    conta_media=('total_bill', 'mean'),
    gorjeta_media=('tip', 'mean'),
    pct_gorjeta_mediana=('pct_gorjeta', 'median'),
    total_mesas=('total_bill', 'count')
).round(2)

print("--- RELATÓRIO EXECUTIVO POR DIA E HORÁRIO ---")
print(resumo_por_dia)
```

---

### Passo 5 — Ordenação e Identificação dos Maiores Pagadores

```python
# 1. Top 5 mesas que deixaram as maiores gorjetas em valor absoluto
top5_gorjetas = df_tips.sort_values(by='tip', ascending=False).head(5)

print("--- TOP 5 MAIORES GORJETAS DO RESTAURANTE ---")
print(top5_gorjetas[['total_bill', 'tip', 'pct_gorjeta', 'day', 'smoker', 'size']])
```

---

## 🏆 Desafio Técnico de Validação
Descubra se fumantes (`smoker == 'Yes'`) deixam, em média, um percentual de gorjeta (`pct_gorjeta`) maior ou menor do que não fumantes (`smoker == 'No'`): `df_tips.groupby('smoker')['pct_gorjeta'].mean()`.

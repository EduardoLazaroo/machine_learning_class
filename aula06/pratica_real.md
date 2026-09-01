# Laboratório Prático — Aula 06: Visualização Exploratória Avançada

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Palmer Penguins Dataset* (Morfologia e ecologia de 3 espécies de pinguins)

---

## 🎯 Objetivo do Laboratório
Construir uma suíte completa de gráficos para Análise Exploratória de Dados (EDA) com **Matplotlib** e **Seaborn**: Pairplot multivariado, Histogramas com Curva de Densidade (KDE), Boxplots segmentados para detecção de outliers e Mapa de Calor de Correlações de Pearson (`heatmap`).

---

### Passo 1 — Configuração Estética e Carregamento

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Definimos o tema visual moderno e a paleta de cores
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams['figure.figsize'] = (10, 6)

# 2. Carregamos o dataset de pinguins limpo
df_pinguins = sns.load_dataset('penguins').dropna()
print(f"Total de registros válidos para visualização: {len(df_pinguins)}")
```

---

### Passo 2 — A Visão Geral Multidimensional (`sns.pairplot`)

```python
# 1. Geramos o pairplot comparando todas as variáveis numéricas separadas pela espécie
grafico_pairplot = sns.pairplot(
    df_pinguins, 
    hue='species', 
    markers=["o", "s", "D"],
    diag_kind='kde'
)
plt.subplots_adjust(top=0.95)
grafico_pairplot.fig.suptitle("Pairplot Multivariado: Relações Morfológicas por Espécie", fontsize=14)
plt.show()
```

---

### Passo 3 — Distribuição de Massa Corporal por Ilha (Histograma + KDE)

```python
# 1. Plotamos a distribuição de peso dos pinguins em cada ilha
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df_pinguins, 
    x='body_mass_g', 
    hue='island', 
    kde=True, 
    element='step'
)
plt.title("Distribuição da Massa Corporal (g) por Ilha", fontsize=12)
plt.xlabel("Massa Corporal (gramas)")
plt.ylabel("Contagem de Pinguins")
plt.show()
```

---

### Passo 4 — Diagnóstico de Outliers por Espécie (Boxplot)

```python
# 1. Inspecionamos a dispersão do comprimento do bico por espécie
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_pinguins, 
    x='species', 
    y='bill_length_mm', 
    palette='Set2'
)
plt.title("Boxplot: Comprimento do Bico (mm) por Espécie", fontsize=12)
plt.xlabel("Espécie")
plt.ylabel("Comprimento do Bico (mm)")
plt.show()
```

---

### Passo 5 — Matriz de Correlação de Pearson (`sns.heatmap`)

```python
# 1. Selecionamos as colunas estritamente numéricas e calculamos a correlação
colunas_num = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
matriz_correlacao = df_pinguins[colunas_num].corr()

# 2. Plotamos o mapa de calor anotado
plt.figure(figsize=(8, 6))
sns.heatmap(
    matriz_correlacao, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1,
    linewidths=0.5
)
plt.title("Mapa de Calor: Correlação Linear de Pearson", fontsize=12)
plt.show()
```

---

## 🏆 Desafio Técnico de Validação
Identifique no mapa de calor qual par de variáveis físicas possui a correlação positiva mais forte da base (acima de $0.85$).

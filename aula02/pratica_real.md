# Laboratório Prático — Aula 02: Álgebra Numérica Vetorizada com NumPy

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  
**Dataset Real:** *Scikit-Learn Diabetes Dataset* (442 pacientes x 10 variáveis fisiológicas em matriz pura)

---

## 🎯 Objetivo do Laboratório
Manipular matrizes numéricas reais em NumPy sem o uso de laços `for`: operações de slicing, estatísticas por eixo (`axis`), normalização Z-score via broadcasting e cálculo de similaridade vetorial por produto escalar.

---

### Passo 1 — Carregando a Matriz NumPy Real

```python
import numpy as np
from sklearn.datasets import load_diabetes

# 1. Carregamos o dataset em formato de matrizes NumPy brutas
dados_diabetes = load_diabetes()
X = dados_diabetes.data       # Matriz bidimensional (442 x 10)
y = dados_diabetes.target     # Vetor unidimensional (442)

print("--- PROPRIEDADES DA MATRIZ NUMPY ---")
print(f"Tipo da estrutura: {type(X)}")
print(f"Shape de X:        {X.shape} (442 pacientes x 10 exames)")
print(f"Shape de y:        {y.shape} (Progressão da doença)")
print(f"Tipo dos dados:    {X.dtype}")
```

---

### Passo 2 — Fatiamento Matricial (*Slicing*)

```python
# 1. Extraímos apenas os 5 primeiros pacientes e as 3 primeiras colunas
submatriz = X[:5, :3]
print("--- 5 PRIMEIROS PACIENTES (3 PRIMEIROS EXAMES) ---")
print(np.round(submatriz, 4))

# 2. Extraímos todo o vetor da 2ª coluna (Índice de Massa Corporal - IMC)
vetor_imc = X[:, 2]
print(f"\nTotal de medições de IMC extraídas: {vetor_imc.shape[0]}")
```

---

### Passo 3 — Estatísticas por Eixo sem Laços `for`

```python
# 1. Média de cada coluna (axis=0 percorre as linhas, calculando por coluna)
medias_por_exame = np.mean(X, axis=0)

# 2. Desvio padrão de cada coluna
desvio_por_exame = np.std(X, axis=0)

# 3. Mínimo e Máximo global da matriz
minimo_global = np.min(X)
maximo_global = np.max(X)

print(f"Médias das 10 colunas:\n{np.round(medias_por_exame, 4)}")
print(f"Desvio padrão das 10 colunas:\n{np.round(desvio_por_exame, 4)}")
print(f"Faixa global de valores: [{minimo_global:.4f}, {maximo_global:.4f}]")
```

---

### Passo 4 — Padronização Z-Score com Broadcasting

```python
# 1. Criamos uma matriz sintética com números em escala de pressão arterial (100 a 180)
pressao_bruta = np.array([
    [120, 80],
    [140, 90],
    [160, 100],
    [110, 75]
], dtype=float)

# 2. Aplicamos Z-score: Z = (X - média) / desvio_padrão (Vetorizado!)
media_colunas = np.mean(pressao_bruta, axis=0)
desvio_colunas = np.std(pressao_bruta, axis=0)

pressao_padronizada = (pressao_bruta - media_colunas) / desvio_colunas

print("--- MATRIZ BRUTA ---")
print(pressao_bruta)
print("\n--- MATRIZ PADRONIZADA (Média 0, Variância 1) ---")
print(np.round(pressao_padronizada, 4))
```

---

### Passo 5 — Similaridade por Produto Escalar (*Dot Product*)

```python
# 1. Selecionamos o perfil do Paciente 0 e do Paciente 1
paciente_0 = X[0]
paciente_1 = X[1]

# 2. Calculamos a similaridade via produto escalar vetorial (np.dot)
similaridade = np.dot(paciente_0, paciente_1)
print(f"Produto escalar entre Paciente 0 e Paciente 1: {similaridade:.6f}")
```

---

## 🏆 Desafio Técnico de Validação
Utilizando apenas operações NumPy (sem `if` ou `for`), calcule quantos pacientes na base possuem valor de IMC (`X[:, 2]`) maior do que zero: `np.sum(X[:, 2] > 0)`.

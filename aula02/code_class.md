# Aula 02 - Python para Ciência de Dados - Parte 1 (NumPy & Estruturas Numéricas)

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  

---

> [!NOTE]
> 🔙 **De onde viemos:** Na Aula 01, compreendemos o que é Machine Learning e vimos que os algoritmos aprendem a partir de dados numéricos. Mas para que um modelo processe milhares de registros sem travar o computador, precisamos de uma estrutura de dados muito mais rápida do que as listas comuns do Python.
> 🎯 **Objetivo Principal da Aula:** Dominar a biblioteca **NumPy** e a estrutura `ndarray`, compreendendo operações vetorizadas, indexação/slicing multidimensional, broadcasting e manipulação matricial básica.
> 🚀 **Para onde vamos:** Na próxima aula ('Python para Ciência de Dados - Parte 2'), adicionaremos 'etiquetas e cabeçalhos' a essas matrizes numéricas, aprendendo a manipular tabelas completas com a biblioteca **Pandas**.

---

## Organização Tática da Aula

| Módulo | Atividade | Foco Pedagógico |
| :--- | :--- | :--- |
| **Módulo 1** | **Retomada de Python & Por que NumPy?** | Variáveis, listas e operações simples como ponte para a computação vetorial. |
| **Módulo 2** | **Estrutura dos Arrays & Indexação / Fatiamento** | Formas (*shape*), dimensões (*ndim*), tipos de dados (*dtype*) e máscaras booleanas. |
| **Módulo 3** | **Prática Guiada no Google Colab** | 8 blocos de código em Python/NumPy minuciosamente comentados linha a linha com caixas de dúvidas. |
| **Módulo 4** | **Prática Orientada & Experimentação Fácil** | Análise estatística vetorial de status de personagens de RPG (D&D/Skyrim). |
| **Módulo 5** | **Checklist de Autonomia & Bibliografia** | Autoavaliação do estudante e referências oficiais. |

---

## Módulo 1: Fundamentação Teórica & Computação Vetorial

### 1.1 Retomada: Variáveis, Números e Listas Python
Antes de usar uma biblioteca nova, vamos retomar três ideias que aparecerão em todos os códigos: uma **variável** guarda um valor; uma **lista** guarda vários valores; e `print()` mostra o resultado na tela.

```python
# Uma variável guarda um único valor numérico
bonus_forca = 5

# Uma lista guarda vários valores entre colchetes
forcas_herois = [18, 10, 8, 14]

# Podemos acessar o primeiro valor da lista pelo índice 0
print("Força do primeiro herói:", forcas_herois[0])

# Uma operação matemática cria um novo valor
print("Força do primeiro herói com bônus:", forcas_herois[0] + bonus_forca)
```

> [!TIP]
> Antes de executar, tente prever: qual número aparecerá em cada `print()`? Fazer essa pequena previsão ajuda a ler código sem precisar decorar comandos.

### 1.2 Por que usar NumPy?
Em Python nativo, listas são coleções flexíveis e excelentes para começar. Quando precisamos calcular com milhões de números, porém, elas exigem que o Python percorra os valores um a um.

Em Machine Learning, lidamos com operações matriciais pesadas. O **NumPy (Numerical Python)** organiza números em arrays e permite a **vetorização**: aplicar uma operação ao conjunto inteiro sem escrever um `for` explícito. Internamente, ele usa estruturas eficientes em C; esse detalhe explica o desempenho, mas não precisa ser decorado agora.

```mermaid
graph LR
    subgraph Lista Nativa Python - Lento
    P1[Ponteiro] --> O1[Objeto Float 10.5]
    P2[Ponteiro] --> O2[Objeto Float 20.1]
    end
    subgraph Array NumPy em C - Ultra Rápido
    B[Bloco Contínuo de Memória: 10.5 | 20.1 | 30.8]
    end
```

> [!TIP]
> ⚔️ **Analogia Geek — Atributos de Personagem de RPG (D&D / Skyrim):**
> Pense num time de 4 heróis em um RPG. Cada herói possui atributos: `[Força, Destreza, Inteligência, Vida]`. Em vez de fazer um laço de repetição demorado para aplicar um feitiço de "Bênção" (+5 de Força para cada herói), o NumPy faz isso em um único disparo de magia vetorial `matriz_atributos + 5` em microssegundos!

> [!NOTE]
> 💡 **Curiosidade da Aula — A Origem do NumPy:**
> O NumPy foi criado em **2005 por Travis Oliphant** ao fundir duas antigas bibliotecas chamadas *Numeric* e *Numarray*. Travis escreveu a ponte inteira de integração com a linguagem C para que o Python — uma linguagem famosa por ser fácil de ler — pudesse rodar cálculos numéricos na mesma velocidade da linguagem C ou Fortran! Hoje, praticamente TODA a IA do planeta roda sobre o NumPy!

> [!IMPORTANT]
> 💡 **Em 1 Frase:** O NumPy armazena números de forma contínua em C, permitindo realizar cálculos em milhões de números simultaneamente em microssegundos sem precisar de laços `for`.

---

## Módulo 2: O Mecanismo por Dentro & Regras Práticas

### 2.1 Principais Atributos e Métodos de um Array NumPy

| Atributo / Função | Descrição e Utilidade | Exemplo de Retorno |
| :--- | :--- | :--- |
| `array.shape` | Retorna a tupla indicando elementos em cada dimensão. | `(100, 4)` -> 100 linhas, 4 colunas |
| `array.ndim` | Número total de dimensões (1=vetor, 2=matriz, 3=tensor). | `2` |
| `array.dtype` | Tipo dos dados armazenados (ex: `float64`, `int32`). | `float64` |
| `array.reshape()` | Modifica a estrutura sem alterar os dados subjacentes. | `.reshape(-1, 1)` |
| `np.mean()`, `np.std()` | Calcula a média estatística e o desvio padrão. | `15.42` |

> [!IMPORTANT]
> 💡 **Em 1 Frase:** Atributos como `.shape` e `.ndim` revelam o tamanho e a dimensão da matriz, enquanto métodos como `.reshape()` permitem reconfigurar sua forma sem alterar os dados.

---

## Módulo 3: Prática Guiada no Google Colab

Abra o [Google Colab](https://colab.research.google.com), crie um novo Notebook e execute os blocos abaixo.

---

### Bloco 3.1 — Da Lista Python ao Array NumPy

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que devo entender primeiro neste bloco?**  
> Primeiro, observe que uma lista e um array podem guardar os mesmos números. Em seguida, compare como cada estrutura multiplica todos os valores. O tempo de execução é apenas uma demonstração complementar.

```python
# 1. Importamos a biblioteca NumPy com o alias padrão 'np'
import numpy as np

# 2. Criamos uma lista Python e um array NumPy com os mesmos números
lista_python = [10, 20, 30, 40]
array_numpy = np.array([10, 20, 30, 40])

# 3. Em uma lista, multiplicamos cada elemento explicitamente
lista_com_bonus = [numero * 2 for numero in lista_python]

# 4. No NumPy, uma única operação alcança todos os elementos do array
array_com_bonus = array_numpy * 2

# 5. Exibimos os resultados para comparar as duas formas
print("Lista Python com bônus:", lista_com_bonus)
print("Array NumPy com bônus: ", array_com_bonus)

# 6. Prévia: no NumPy, esta operação em conjunto é chamada de vetorização
print("O array inteiro foi multiplicado sem escrever um for explícito.")
```

---

### Bloco 3.2 — Formas de Inicialização de Arrays (`np.array`, `zeros`, `ones`, `arange`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Quando usamos `zeros` ou `ones` na prática?**  
> Usamos para pré-alocar espaço em memória antes de preencher a matriz com dados calculados ou quando criamos matrizes de pesos iniciais para modelos e Redes Neurais.

```python
# 1. Criamos um vetor 1D simples a partir de uma lista de números decimais
vetor_unidimensional = np.array([10.5, 20.0, 30.2, 40.8, 50.1])

# 2. Criamos uma matriz 3x4 (3 linhas x 4 colunas) preenchida com zeros
matriz_zeros = np.zeros(shape=(3, 4))

# 3. Criamos uma matriz 2x3 (2 linhas x 3 colunas) preenchida com uns
matriz_uns = np.ones(shape=(2, 3))

# 4. Criamos uma sequência numérica de 0 a 50 pulando de 10 em 10 (passo 10)
vetor_com_passo = np.arange(start=0, stop=50, step=10)

# 5. Exibimos os arrays gerados no console
print("--- FORMAS DE CRIAR ARRAYS NUMPY ---")
print("1. Vetor 1D Comum:\n", vetor_unidimensional)
print("\n2. Matriz Zeros (3 linhas x 4 colunas):\n", matriz_zeros)
print("\n3. Vetor de Sequência (0 a 50 com passo 10):\n", vetor_com_passo)
```

---

### Bloco 3.3 — Criando a Matriz de Status de RPG (2D Array)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que indicam os atributos `shape` e `ndim`?**  
> `shape` indica `(linhas, colunas)`. Exemplo: `(4, 3)` significa 4 linhas (heróis) por 3 colunas (atributos). `ndim` é a quantidade de dimensões (ex: 2D para matrizes).

```python
# 1. Criamos a matriz 2D com status de 4 Heróis de RPG: [Força, Destreza, Inteligência]
matriz_status_herois = np.array([
    [18, 12, 10], # Guerreiro (Força alta)
    [10, 19, 14], # Arqueiro (Destreza alta)
    [8,  14, 20], # Mago (Inteligência alta)
    [14, 16, 12]  # Paladino (Equilibrado)
])

# 2. Exibimos a matriz no console
print("--- MATRIZ DE STATUS DOS HERÓIS (4 Heróis x 3 Atributos) ---")
print(matriz_status_herois)

# 3. Inspecionamos os atributos estruturais do array NumPy
print("\n--- ATRIBUTOS ESTRUTURAIS DA MATRIZ ---")
print(f"📐 Forma (Linhas, Colunas): {matriz_status_herois.shape}")
print(f"🌐 Dimensões (ndim):         {matriz_status_herois.ndim}D")
print(f"💾 Tipo dos dados (dtype):   {matriz_status_herois.dtype}")
```

---

### Bloco 3.4 — Indexação e Fatiamento (*Slicing*) de Matrizes

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Como funciona a sintaxe `matriz[linha, coluna]`?**  
> O primeiro valor antes da vírgula indica a linha desejada; o segundo valor indica a coluna. O símbolo de dois-pontos `:` significa "selecionar todos os elementos daquela dimensão". Lembre-se: em Python, os índices começam no zero (`0`)!

```python
# 1. Acessamos a inteligência do Mago (Linha 2, Coluna 2)
inteligencia_mago = matriz_status_herois[2, 2]

# 2. Extraímos a primeira coluna inteira (Força de todos os heróis) usando ':' nas linhas
forca_todos_herois = matriz_status_herois[:, 0]

# 3. Extraímos a linha inteira do Guerreiro (Linha 0) usando ':' nas colunas
guerreiro_atributos = matriz_status_herois[0, :]

# 4. Exibimos os resultados fatiados
print("--- EXEMPLOS DE FATIAMENTO (SLICING) ---")
print(f"🧙‍♂️ Inteligência do Mago (Linha 2, Coluna 2): {inteligencia_mago}")
print(f"⚔️ Força de todos os 4 heróis (Coluna 0):    {forca_todos_herois}")
print(f"🛡️ Status completos do Guerreiro (Linha 0):  {guerreiro_atributos}")
```

---

### Bloco 3.5 — Operações Vetoriais e Broadcasting (Poções e Buffs)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que é a operação vetorial? E o que é broadcasting?**  
> Operação vetorial é calcular com todos os valores do array de uma vez. Broadcasting é a regra que permite, por exemplo, que o único valor `5` seja usado junto com todos os valores da matriz. Nesta aula, basta observar esse caso simples.

```python
# 1. Aplicamos um feitiço de bônus (+5) em todos os atributos de todos os heróis em 1 linha
matriz_herois_com_bônus = matriz_status_herois + 5

# 2. Exibimos as matrizes original e modificada
print("--- OPERAÇÃO VETORIAL (BROADCASTING) ---")
print("Matriz Original:")
print(matriz_status_herois)
print("\n✨ Matriz após o Feitiço de Bênção (+5 em tudo sem usar laço FOR):")
print(matriz_herois_com_bônus)
```

---

### Bloco 3.6 — Máscaras Booleanas (*Boolean Masking*)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que é uma Máscara Booleana?**  
> É um vetor contendo `True` ou `False` para cada elemento da matriz. Quando aplicamos essa máscara de volta no array, o NumPy filtra e retorna apenas os elementos onde a condição foi verdadeira (`True`).

```python
# 1. Isolamos o vetor de Força de todos os heróis (Coluna 0)
vetor_forca = matriz_status_herois[:, 0]

# 2. Criamos uma máscara booleana testando a condição: Força >= 14
mascara_guerreiros_fortes = vetor_forca >= 14

# 3. Exibimos os resultados da filtragem
print("--- FILTRAGEM COM MÁSCARA BOOLEANA ---")
print("Vetor de Força de Todos:        ", vetor_forca)
print("Máscara Booleana (Força >= 14): ", mascara_guerreiros_fortes)
print("Apenas as Forças dos FORTES:    ", vetor_forca[mascara_guerreiros_fortes])
```

---

### Bloco 3.7 — Redimensionamento de Arrays (`reshape`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que usamos `.reshape(-1, 1)` com tanta frequência no Scikit-Learn?**  
> Porque o Scikit-Learn exige que a matriz de entrada $X$ seja sempre bidimensional (linhas e colunas). O valor `-1` diz ao NumPy: *"Calcule automaticamente o número de linhas necessário para manter a matriz com 1 coluna"*.

```python
# 1. Criamos um vetor unidimensional simples com 12 números (de 0 a 11)
vetor_original_12_elementos = np.arange(12)

# 2. Redimensionamos para uma matriz de 3 linhas por 4 colunas
matriz_3x4 = vetor_original_12_elementos.reshape(3, 4)

# 3. Redimensionamos para uma matriz coluna (12 linhas por 1 coluna) usando -1
matriz_coluna_12x1 = vetor_original_12_elementos.reshape(-1, 1)

# 4. Exibimos as matrizes reconfiguradas
print("--- REDIMENSIONAMENTO DE MATRIZES (RESHAPE) ---")
print("Vetor Original 1D (12 elementos):\n", vetor_original_12_elementos)
print("\nMatriz Redimensionada (3x4):\n", matriz_3x4)
print("\nMatriz Coluna (12x1):\n", matriz_coluna_12x1)
```

---

### Bloco 3.8 — Primeiro Contato com Operações por Eixo (`axis=0` e `axis=1`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Qual a diferença entre `axis=0` e `axis=1`?**  
> - `axis=0`: opera no sentido vertical (percorre as colunas -> calcula média por atributo).  
> - `axis=1`: opera no sentido horizontal (percorre as linhas -> calcula média por herói).

> [!TIP]
> Não tente decorar `axis` agora. Leia sempre a pergunta: queremos um resultado para cada **atributo** (colunas, `axis=0`) ou para cada **herói** (linhas, `axis=1`)?

```python
# 1. Calculamos a média de cada atributo para o time inteiro (axis=0 percorre colunas)
media_atributos_time = np.mean(matriz_status_herois, axis=0)

# 2. Calculamos a média geral de atributos de cada herói (axis=1 percorre linhas)
media_atributos_por_heroi = np.mean(matriz_status_herois, axis=1)

# 3. Exibimos os relatórios estatísticos da equipe
print("--- MÉDIAS ESTATÍSTICAS DA EQUIPE ---")
print(f"📊 Média do time em [Força, Destreza, Inteligência]: {media_atributos_time.round(1)}")
print(f"📊 Média geral por Herói:                           {media_atributos_por_heroi.round(1)}")
print(f"🌟 Média geral de toda a matriz de status:          {np.mean(matriz_status_herois):.2f}")
```

---

## Módulo 4: Prática Orientada & Experimentação Fácil

### Roteiro de Personalização para o Estudante:
O código abaixo simula os status de um time de heróis de RPG. Você não precisa programar do zero, apenas **alterar a poção de bônus no comentário 1** e rodar para ver o resultado do seu time!

1. **Aplique uma poção de bônus:** Na linha `valor_pocao_bonus = 5`, experimente mudar para `10` ou `15`.
2. **Re-execute e observe:** Veja como o NumPy aplica o bônus em todo o time vetorialmente em microssegundos!

```python
# =============================================================================
# CÓDIGO BASE PRONTO PARA SUA EXPERIMENTAÇÃO
# =============================================================================
import numpy as np

# 1. Definimos a matriz de status do seu time: [Força, Destreza, Inteligência]
meu_time_rpg = np.array([
    [15, 10, 8],  # Herói 1 (Guerreiro)
    [12, 18, 14], # Herói 2 (Arqueiro)
    [9,  12, 19]  # Herói 3 (Mago)
])

# -----------------------------------------------------------------------------
# STEP 1: PASSO DE EXPERIMENTAÇÃO DO ALUNO — ALTERE O BÔNUS DA POÇÃO ABAIXO:
# -----------------------------------------------------------------------------
valor_pocao_bonus = 5 # Tente mudar para 10, 15 ou 20

# 2. O NumPy aplica o bônus para TODO o time em 1 única linha (Broadcasting)
time_herois_bufado = meu_time_rpg + valor_pocao_bonus

# 3. Exibimos o time atualizado e a média de inteligência recalculada
print("--- SEU TIME DE RPG APÓS A POÇÃO MÁGICA ---")
print(time_herois_bufado)
print(f"\n📊 Média de Inteligência do seu time bufado: {np.mean(time_herois_bufado[:, 2]):.1f}")
```

---

### 🔮 Aquecimento & Spoiler da Próxima Aula (Para Ir Além)

> [!TIP]
> 🧠 **Conceito-Semente — Matrizes são rápidas, mas cadê o nome das colunas?**  
> Uma matriz NumPy é incrível para cálculo matemático puro, mas no mundo real os dados vêm com nomes: *Nome*, *Idade*, *Salário*, *Cidade*. Na **Aula 03**, conheceremos o **Pandas** e sua estrutura principal, o **DataFrame** — que funciona como uma super planilha inteligente do Excel turbinada por código Python, permitindo filtros complexos, agrupamentos e consultas em milissegundos!
> 
> 🚀 **Desafio Proativo de Autoestudo (Opcional):**  
> Pesquise no Google ou execute no Colab: `import pandas as pd; df = pd.DataFrame({'Nome': ['Ana', 'Bruno'], 'Idade': [22, 28]}); print(df.head())`. Veja a diferença visual entre uma matriz NumPy e uma tabela Pandas!

---

## Módulo 5: Checklist de Autonomia do Estudante

- [ ] Compreendi a importância do NumPy para a eficiência computacional em Machine Learning.
- [ ] Sei criar vetores e matrizes usando `np.array()` e `np.arange()`.
- [ ] Entendi como acessar elementos específicos e fatiar colunas via `matriz[linha, coluna]`.
- [ ] Consigo identificar se uma média será calculada por atributo (`axis=0`) ou por herói (`axis=1`).
- [ ] Consigo alterar o bônus da poção mágica no script e observar a atualização do meu time.

---

## Referências Bibliográficas & Documentações Oficiais

- 📖 **Livro Texto:** McKinney, Wes. — *Python para Análise de Dados: Tratamento de Dados com Pandas, NumPy e Jupyter*. Editora Novatec.
- 🔗 **Documentação Oficial NumPy:** [numpy.org/doc/stable](https://numpy.org/doc/stable/)
- 🎥 **Vídeo Recomendado (Curso em Vídeo - Guanabara):** [Curso de Python para Iniciantes (YouTube)](https://www.youtube.com/watch?v=S9uPNppGsGo)

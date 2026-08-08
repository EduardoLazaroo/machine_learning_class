# Aula 01 - Introdução à Inteligência Artificial e Machine Learning

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  

---

> [!NOTE]
> 🔙 **De onde viemos:** Esta é a aula inaugural! Iniciamos nossa jornada conectando o conhecimento prévio dos alunos sobre computação tradicional com a revolução impulsionada por dados.
> 🎯 **Objetivo Principal da Aula:** Compreender a evolução da IA, diferenciar Inteligência Artificial, Machine Learning e Deep Learning, dominar os 3 tipos de aprendizado e executar o primeiro script Python no Google Colab.
> 🚀 **Para onde vamos:** Na próxima aula ('Python para Ciência de Dados - Parte 1'), iniciaremos a computação vetorial com NumPy para manipular matrizes e vetores numéricos.

---

## Organização Tática da Aula

| Módulo | Atividade | Foco Pedagógico |
| :--- | :--- | :--- |
| **Módulo 1** | **Fundamentação Teórica & IA no Cotidiano** | Recomendações (Netflix/Spotify), evolução histórica e a relação IA vs ML vs Deep Learning. |
| **Módulo 2** | **Paradigmas de Aprendizado & Ciclo de Projeto** | Aprendizado Supervisionado, Não Supervisionado e por Reforço + O Pipeline de ML. |
| **Módulo 3** | **Primeiro Contato Hands-On no Google Colab** | 9 blocos de código em Python minuciosamente comentados no Scikit-Learn com caixas de dúvidas comuns. |
| **Módulo 4** | **Prática Orientada & Experimentação Fácil** | Personalização rápida do modelo de classificação de flores com alteração de semente e sua própria flor. |
| **Módulo 5** | **Checklist de Autonomia & Bibliografia** | Autoavaliação do estudante e referências oficiais. |

---

## Módulo 1: Fundamentação Teórica & Contexto do Mundo Real

### 1.1 A IA Invisível do Nosso Dia a Dia
A Inteligência Artificial não é uma promessa do futuro; ela opera continuamente nos bastidores da sociedade moderna:
- 🎬 **Recomendações Customizadas:** Netflix, Spotify e YouTube prevendo o que você deseja assistir ou ouvir com base no seu histórico.
- 📱 **Visão Computacional:** Desbloqueio facial em smartphones e filtros interativos do Instagram/TikTok.
- 🚗 **Mobilidade e Rotas:** Waze e Google Maps calculando o menor tempo de percurso ajustando-se ao trânsito em tempo real.
- 💬 **Processamento de Linguagem Natural (PLN):** Corretores ortográficos, tradutores instantâneos e assistentes virtuais (ChatGPT, Alexa, Siri).

> [!TIP]
> 🎮 **Analogia Geek — O Código Tradicional vs. A Matrix:**
> Na programação tradicional, nós somos como os arquitetos do programa: escrevemos linha por linha de regras `if/else` explícitas. Em **Machine Learning**, nós tomamos a "Pílula Vermelha" (*Red Pill*): fornecemos os dados históricos e as respostas certas, e a própria máquina descobre as regras sozinha, como se estivesse decodificando o código da Matrix!

> [!NOTE]
> 💡 **Curiosidade da Aula — A Origem do Termo "Machine Learning" & O AlphaGo:**
> - O termo *Machine Learning* foi cunhado em **1959 por Arthur Samuel**, um pioneiro da IBM que criou um programa de computador para jogar Damas. O programa jogava contra si mesmo milhares de vezes e acabou aprendendo estratégias melhores do que o próprio criador!
> - Em 2016, no **Aprendizado por Reforço**, a IA **AlphaGo** (da Google DeepMind) chocou o mundo ao derrotar Lee Sedol, o campeão mundial de Go (um jogo de tabuleiro milenar asiático com mais combinações possíveis do que o número de átomos no universo visível!).

> [!IMPORTANT]
> 💡 **Em 1 Frase:** Na programação tradicional nós escrevemos as regras na mão; em Machine Learning a máquina analisa os dados históricos e descobre as regras sozinha.

---

### 1.2 A Hierarquia Conceitual: IA vs. Machine Learning vs. Deep Learning

```mermaid
graph TD
    A["🧠 INTELIGÊNCIA ARTIFICIAL (IA)<br/>Sistemas que simulam capacidade humana"] --> B["⚙️ MACHINE LEARNING (ML)<br/>Algoritmos que aprendem com dados"]
    B --> C["🕸️ DEEP LEARNING (DL)<br/>Redes Neurais Profundas para dados complexos"]
```

> [!IMPORTANT]
> 💡 **Em 1 Frase:** Inteligência Artificial é o conceito amplo, Machine Learning é o método de aprender com dados, e Deep Learning é a técnica avançada usando redes neurais profundas.

---

## Módulo 2: Paradigmas de Aprendizado & Ciclo de Projeto

### 2.1 Os 3 Paradigmas do Aprendizado de Máquina

| Paradigma | Como Funciona | Entrada de Dados | Exemplo Prático |
| :--- | :--- | :--- | :--- |
| **Supervisionado** | O algoritmo aprende com exemplos rotulados (Entrada + Resposta Correta). | Dados ($X$) com Rótulos ($y$) | Detecção de Spam, Previsão de Vendas, Diagnósticos Médicos. |
| **Não Supervisionado** | O algoritmo busca padrões ocultos ou agrupamentos sem respostas prévias. | Dados ($X$) sem Rótulos | Segmentação de Clientes, Detecção de Anomalias. |
| **Por Reforço** | Um agente aprende executando ações em um ambiente e recebendo recompensas ou punições. | Interação contínua com ambiente | Carros Autônomos, Jogos (AlphaGo, Dota 2, Robótica). |

> [!NOTE]
> 📌 **Por que usar o Iris Dataset no nosso primeiro código?**
> O dataset *Iris* foi criado pelo estatístico Ronald Fisher em 1936 e é considerado o "Hello World" oficial da Ciência de Dados. Escolhemos ele porque possui apenas 150 amostras, 4 colunas simples, zero nulos e permite que você foque 100% no aprendizado do código sem se preocupar com problemas de dados!  
> 🔗 **Onde encontrar mais datasets públicos para praticar?**  
> - [Kaggle Datasets](https://www.kaggle.com/datasets) (O maior portal do mundo para praticar Ciência de Dados).  
> - [UCI Machine Learning Repository](https://archive.ics.uci.edu/) (Repositório acadêmico clássico).

> [!IMPORTANT]
> 💡 **Em 1 Frase:** No Aprendizado Supervisionado temos a "prova gabaritada", no Não Supervisionado buscamos "grupos parecidos", e no Por Reforço aprendemos por "tentativa, erro e recompensa".

---

## Módulo 3: Prática Guiada no Google Colab

Abra o [Google Colab](https://colab.research.google.com) e acompanhe a execução dos blocos de código abaixo.

---

### Bloco 3.1 — Verificando o Ambiente Python e a Versão do Interpretador

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que é o interpretador Python e por que verificamos sua versão?**  
> O interpretador é o "cérebro" que executa nosso código no servidor em nuvem do Google Colab. Checar a versão garante que nosso código rodará em uma versão moderna (Python 3.x) sem incompatibilidades.

```python
# 1. Importamos o módulo nativo 'sys' para acessar informações do sistema operacional
import sys

# 2. Exibimos a versão exata do interpretador Python em execução no servidor do Colab
print("--- VERIFICAÇÃO DO AMBIENTE PYTHON ---")
print(f"🐍 Versão do Python em execução no servidor: {sys.version}")

# 3. Imprimimos uma mensagem amigável de confirmação
print("🔍 Tudo pronto para começarmos os primeiros passos em Ciência de Dados!")
```

---

### Bloco 3.2 — Importando as Bibliotecas Essenciais da Ciência de Dados

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Para que serve cada uma dessas bibliotecas?**  
> - `NumPy`: realiza cálculos matemáticos com vetores e matrizes ultra velozes.  
> - `Pandas`: cria e manipula tabelas estruturadas (DataFrames), como se fosse um Excel avançado.  
> - `Matplotlib` e `Seaborn`: desenham gráficos estatísticos bonitos e explicativos.  
> - `Scikit-Learn`: a biblioteca principal que contém todos os modelos de Inteligência Artificial prontos para uso.

```python
# 1. Importamos o NumPy para cálculos numéricos e matriciais
import numpy as np

# 2. Importamos o Pandas para criação e manipulação de tabelas (DataFrames)
import pandas as pd

# 3. Importamos o Matplotlib para construção de gráficos e visualizações
import matplotlib.pyplot as plt

# 4. Importamos o Seaborn para criar gráficos estatísticos visualmente atraentes
import seaborn as sns

# 5. Importamos o Scikit-Learn (sklearn), nossa caixa de ferramentas de Machine Learning
import sklearn

# 6. Exibimos as versões instaladas para confirmar o carregamento de cada ferramenta
print("--- CHECAGEM DE FERRAMENTAS INSTALADAS ---")
print(f"✅ NumPy (Matemática Vetorial) versão:        {np.__version__}")
print(f"✅ Pandas (Manipulação de Tabelas) versão:   {pd.__version__}")
print(f"✅ Scikit-Learn (Machine Learning) versão:   {sklearn.__version__}")
print("🚀 Todas as bibliotecas fundamentais foram carregadas com sucesso!")
```

---

### Bloco 3.3 — Carregando o Iris Dataset do Scikit-Learn

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que contêm as chaves `data` e `target` no objeto do dataset?**  
> - `data` (Matriz $X$): contém as medidas físicas das flores (comprimento/largura de sépalas e pétalas).  
> - `target` (Vetor $y$): contém o número identificador de cada espécie de flor (0 = Setosa, 1 = Versicolor, 2 = Virginica).

```python
# 1. Importamos a função específica para carregar os dados das flores Iris
from sklearn.datasets import load_iris

# 2. Carregamos o objeto completo contendo dados, rótulos e descrição do dataset
dados_brutos_iris = load_iris()

# 3. Exibimos as chaves principais contidas no dicionário de dados
print("--- ESTRUTURA DO DATASET IRIS ---")
print("📋 Chaves de dados disponíveis no dicionário:", dados_brutos_iris.keys())

# 4. Exibimos os nomes dos 4 atributos de entrada (features de medição das flores)
print("\n📏 Nomes dos Atributos de Entrada (Features):")
for indice, nome_atributo in enumerate(dados_brutos_iris.feature_names):
    print(f"   {indice + 1}. {nome_atributo}")

# 5. Exibimos as 3 classes botânicas de flores possíveis (target names)
print("\n🌸 Classes de Flores Possíveis (Target Names):")
for indice, nome_especie in enumerate(dados_brutos_iris.target_names):
    print(f"   ID {indice}: {nome_especie.upper()}")
```

---

### Bloco 3.4 — Estruturando os Dados em um DataFrame Pandas

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que convertemos matrizes numéricas em um DataFrame Pandas?**  
> Para que possamos visualizar os dados em formato de tabela amigável, com cabeçalhos legíveis (`sepal length`, `petal width`), facilitando a análise e exploração.

```python
# 1. Criamos a tabela Pandas inserindo as 4 colunas numéricas de características
tabela_iris = pd.DataFrame(
    data=dados_brutos_iris.data, 
    columns=dados_brutos_iris.feature_names
)

# 2. Criamos uma nova coluna contendo os códigos numéricos das espécies (0, 1 ou 2)
tabela_iris['codigo_especie'] = dados_brutos_iris.target

# 3. Mapeamos os códigos numéricos 0, 1 e 2 para os nomes botânicos reais
tabela_iris['nome_especie'] = tabela_iris['codigo_especie'].map({
    0: 'setosa', 
    1: 'versicolor', 
    2: 'virginica'
})

# 4. Exibimos o tamanho da tabela (linhas x colunas) e as 5 primeiras amostras
print("--- VISUALIZAÇÃO DA TABELA PANDAS ESTRUTURADA ---")
print(f"📐 Dimensão da Tabela: {tabela_iris.shape[0]} linhas por {tabela_iris.shape[1]} colunas")
print("\nPrimeiros 5 registros da tabela:")
print(tabela_iris.head())
```

---

### Bloco 3.5 — Análise Exploratória Visual (Scatter Plot com Seaborn)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que é um Scatter Plot (Gráfico de Dispersão)?**  
> É um gráfico onde cada flor vira um ponto no plano cartesiano $X \times Y$. Usamos cores distintas para cada espécie para ver se elas formam "grupos separados" visivelmente.

```python
# 1. Definimos as dimensões da figura gráfica (8 polegadas de largura por 5 de altura)
plt.figure(figsize=(8, 5))

# 2. Desenhamos o gráfico de dispersão relacionando comprimento da sépala x largura da pétala
sns.scatterplot(
    data=tabela_iris,
    x='sepal length (cm)',
    y='petal width (cm)',
    hue='nome_especie', # Colore os pontos de acordo com o nome da espécie
    palette='Set1',     # Paleta de cores contrastantes
    s=90                # Tamanho dos pontos no gráfico
)

# 3. Adicionamos títulos e rótulos legíveis aos eixos do gráfico
plt.title("Separação Visual das Espécies de Flores Iris", fontsize=12, fontweight='bold')
plt.xlabel("Comprimento da Sépala (cm)")
plt.ylabel("Largura da Pétala (cm)")
plt.grid(True, linestyle='--', alpha=0.5)

# 4. Exibimos o gráfico na tela
print("🎨 Exibindo o gráfico de dispersão visual no Colab...")
plt.show()
```

---

### Bloco 3.6 — Divisão dos Dados em Treino e Teste (`train_test_split`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **1. Por que NUNCA testamos o modelo com os mesmos dados de treino?**  
> Porque se testarmos com os mesmos dados usados no treino, a máquina pode apenas "decorar a prova" (*Overfitting*) e tirar nota 100, mas falhar miseravelmente na vida real com dados novos.  
> **2. O que faz o parâmetro `random_state=42`?**  
> Trava o sorteio das linhas. Assim, todos os 45 alunos da sala de aula dividem os dados exatamente nas mesmas linhas, obtendo o mesmo resultado ao comparar os códigos.

```python
# 1. Isolamos a matriz de entrada X (4 colunas de medidas) e o vetor de respostas y (espécies)
matriz_entradas = dados_brutos_iris.data
vetor_respostas = dados_brutos_iris.target

# 2. Realizamos a divisão: reservamos 80% dos dados para treino e 20% para teste
matriz_entradas_treino, matriz_entradas_teste, vetor_respostas_treino, vetor_respostas_teste = train_test_split(
    matriz_entradas, 
    vetor_respostas, 
    test_size=0.20, 
    random_state=42
)

# 3. Exibimos a contagem das amostras separadas para cada etapa
print("--- DIVISÃO DE DADOS (TREINO E TESTE) ---")
print(f"📦 Amostras reservadas para TREINO (80%): {matriz_entradas_treino.shape[0]} flores")
print(f"🧪 Amostras reservadas para TESTE  (20%): {matriz_entradas_teste.shape[0]} flores")
print("✅ Os dados foram separados com sucesso sem contaminação do conjunto de teste!")
```

---

### Bloco 3.7 — Treinamento do Modelo de Árvore de Decisão (`fit`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que faz o método `.fit(X, y)`?**  
> É a linha onde a **mágica do aprendizado acontece**! O algoritmo lê as medidas de treino (`X`) e as respostas corretas (`y`) para calcular as regras de separação.

```python
# 1. Importamos a classe DecisionTreeClassifier do Scikit-Learn
from sklearn.tree import DecisionTreeClassifier

# 2. Instanciamos o modelo de Árvore de Decisão com semente fixa
modelo_arvore_decisao = DecisionTreeClassifier(random_state=42)

# 3. O MOMENTO DO APRENDIZADO: A máquina analisa os dados de treino para aprender as regras
modelo_arvore_decisao.fit(matriz_entradas_treino, vetor_respostas_treino)

# 4. Imprimimos mensagens de confirmação do aprendizado
print("🎉 MODELO DE ÁRVORE DE DECISÃO TREINADO COM SUCESSO!")
print("🧠 A IA já aprendeu a diferenciar os 3 tipos de flores com base em sépalas e pétalas!")
```

---

### Bloco 3.8 — Predição e Avaliação da Acurácia (`predict` e `accuracy_score`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que é Acurácia?**  
> É a porcentagem de acertos do modelo. Exemplo: se o modelo previu 30 flores de teste e acertou 29, a acurácia é de 96,67%.

```python
# 1. Importamos a função de cálculo da métrica de acurácia
from sklearn.metrics import accuracy_score

# 2. A IA gera previsões para as 30 flores reservadas para o teste que ela nunca viu
previsoes_teste = modelo_arvore_decisao.predict(matriz_entradas_teste)

# 3. Comparamos as respostas reais com as previsões geradas para calcular a taxa de acerto
acuracia_final = accuracy_score(vetor_respostas_teste, previsoes_teste)

# 4. Exibimos os relatórios numéricos no console
print("--- RESULTADO DA AVALIAÇÃO DO MODELO ---")
print(f"📊 Respostas Reais do Teste:    {vetor_respostas_teste}")
print(f"🔮 Respostas Previstas pela IA: {previsoes_teste}")
print(f"\n🌟 Taxa de Acurácia Obtida: {acuracia_final * 100:.2f}% de acertos nos dados de teste!")
```

---

### Bloco 3.9 — Fazendo uma Previsão para uma Planta Totalmente Nova

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que passamos as medidas dentro de colchetes duplos `[[5.1, 3.5, 1.4, 0.2]]`?**  
> Porque o Scikit-Learn exige que a entrada de dados $X$ seja sempre uma **matriz bidimensional (tabela)**, mesmo quando temos apenas 1 única flor para prever!

```python
# 1. Definimos as 4 medidas de uma flor desconhecida: [sépala_comp, sépala_larg, pétala_comp, pétala_larg]
medidas_flor_nova = [[5.1, 3.5, 1.4, 0.2]]

# 2. O modelo treinado analisa essas 4 medidas e retorna o ID numérico da flor prevista
codigo_classe_previsto = modelo_arvore_decisao.predict(medidas_flor_nova)[0]

# 3. Traduzimos o ID numérico (0, 1 ou 2) para o nome científico real da flor
nome_flor_prevista = dados_brutos_iris.target_names[codigo_classe_previsto]

# 4. Exibimos a previsão final no console
print("--- PREVISÃO EM TEMPO REAL PARA UMA NOVA FLOR ---")
print(f"📏 Medidas Informadas: {medidas_flor_nova[0]}")
print(f"🌷 Resultado da IA: A flor foi classificada como ---> {nome_flor_prevista.upper()} <---")
```

---

## Módulo 4: Prática Orientada & Experimentação Fácil

### Roteiro de Personalização para o Estudante:
Agora é a sua vez de testar o seu próprio modelo! O código abaixo já está 100% pronto. Você só precisa **alterar os números indicados nos comentários 1 e 2**, rodar e ver o seu resultado personalizado!

1. **Altere a sua data/semente:** Na linha `semente_aniversario = 15`, mude o número `15` para o dia do seu aniversário (ex: `7`, `22`, `30`).
2. **Crie a sua própria flor:** Altere as 4 medidas em `minhas_medidas_flor = [[6.2, 2.8, 4.8, 1.8]]` colocando valores de sua preferência e veja qual espécie a IA identifica para a sua flor!

```python
# =============================================================================
# CÓDIGO BASE PRONTO PARA SUA EXPERIMENTAÇÃO
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Carregamos o dataset Iris
dados_iris = load_iris()
matriz_entradas = dados_iris.data
vetor_respostas = dados_iris.target

# -----------------------------------------------------------------------------
# STEP 1: PASSO DE EXPERIMENTAÇÃO DO ALUNO — ALTERE O DIA DO SEU ANIVERSÁRIO:
# -----------------------------------------------------------------------------
semente_aniversario = 15 # Altere para o dia do seu aniversario (ex: 7, 22, 30)

# 2. Dividimos os dados usando o seu aniversário como semente aleatória
matriz_treino, matriz_teste, respostas_treino, respostas_teste = train_test_split(
    matriz_entradas, 
    vetor_respostas, 
    test_size=0.25, 
    random_state=semente_aniversario
)

# 3. Instanciamos e treinamos o algoritmo KNN (K-Vizinhos Mais Próximos)
modelo_knn_estudante = KNeighborsClassifier(n_neighbors=5)
modelo_knn_estudante.fit(matriz_treino, respostas_treino)

# 4. Avaliamos a acurácia obtida pela sua semente de aniversário
acuracia_estudante = accuracy_score(respostas_teste, modelo_knn_estudante.predict(matriz_teste))
print(f"📊 Acurácia do seu modelo personalizado (Semente {semente_aniversario}): {acuracia_estudante * 100:.2f}%")

# -----------------------------------------------------------------------------
# STEP 2: PASSO DE EXPERIMENTAÇÃO DO ALUNO — DIGITE 4 MEDIDAS PARA A SUA FLOR:
# -----------------------------------------------------------------------------
minhas_medidas_flor = [[6.2, 2.8, 4.8, 1.8]] # [sépala_comp, sépala_larg, pétala_comp, pétala_larg]

# 5. O algoritmo identifica a espécie da flor que você criou
codigo_previsto = modelo_knn_estudante.predict(minhas_medidas_flor)[0]
especie_descoberta = dados_iris.target_names[codigo_previsto]
print(f"🌻 A sua flor personalizada foi classificada como: ---> {especie_descoberta.upper()} <---")
```

---

## Módulo 5: Checklist de Autonomia do Estudante

- [ ] Consigo explicar a diferença entre IA, Machine Learning e Deep Learning.
- [ ] Sei diferenciar os 3 paradigmas: Supervisionado, Não Supervisionado e por Reforço.
- [ ] Sei acessar o Google Colab e executar blocos de código em Python.
- [ ] Entendi a utilidade das bibliotecas NumPy, Pandas e Scikit-Learn.
- [ ] Consegui alterar os valores da minha flor e ver a IA classificá-la no Colab.

---

## Referências Bibliográficas & Documentações Oficiais

- 📖 **Livro Texto:** Géron, A. — *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow* (Capítulo 1: O Panorama do Aprendizado de Máquina).
- 🔗 **Documentação Scikit-Learn:** [scikit-learn.org - DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- 🎥 **Vídeo Recomendado (Didática Tech):** [O que é Machine Learning? (YouTube)](https://www.youtube.com/watch?v=0Prg8D0qf9U)

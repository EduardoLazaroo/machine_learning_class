# Aula 18 - Redes Neurais com TensorFlow/Keras - Parte 2 (Regressão & Classificação Multiclasse)

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  

---

> [!NOTE]
> 🔙 **De onde viemos:** Na Aula 17, construímos nosso primeiro modelo Sequencial no Keras e aprendemos a compilar, treinar e avaliar redes profundas no dataset Fashion-MNIST. Agora, precisamos dominar a flexibilidade do Keras para alternar entre tarefas de regressão contínua e classificação multiclasse.
> 🎯 **Objetivo Principal da Aula:** Dominar o design de arquiteturas neurais no Keras especializadas para **Regressão Contínua** (`loss='mse'`, métrica MAE) e **Classificação Multiclasse**, comparando a codificação de rótulos inteiros (`sparse_categorical_crossentropy`) versus One-Hot Encoding (`to_categorical` + `categorical_crossentropy`).
> 🚀 **Para onde vamos:** Na próxima aula ('Redes Neurais com TensorFlow/Keras - Parte 3'), aprenderemos a transformar modelos amadores em modelos de produção: como evitar que a rede decore o treino usando **Dropout** e como usar **Callbacks (EarlyStopping & ModelCheckpoint)** para salvar automaticamente o melhor modelo no disco.

---

## Organização Tática da Aula

| Módulo | Atividade | Foco Pedagógico |
| :--- | :--- | :--- |
| **Módulo 1** | **Fundamentação Teórica & Redes Neurais para Regressão vs Classificação** | A camada de saída para Regressão ($1$ neurônio sem ativação) vs Classificação ($N$ neurônios + Softmax). |
| **Módulo 2** | **Sparse vs Categorical Crossentropy & One-Hot Encoding no Keras** | Como preparar o target $y$ (`to_categorical`) e quando usar cada função de perda. |
| **Módulo 3** | **Prática Guiada no Google Colab** | 7 blocos de código em Python/TensorFlow minuciosamente comentados linha por linha com caixas de dúvidas. |
| **Módulo 4** | **Prática Orientada & Experimentação Fácil** | Regressão de valor imobiliário com Redes Neurais Keras alterando camadas e neurônios. |
| **Módulo 5** | **Checklist de Autonomia & Bibliografia** | Autoavaliação do estudante e referências oficiais. |

---

## Módulo 1: Fundamentação Teórica & Configuração da Camada de Saída

### 1.1 Diferenças entre Redes de Regressão e Classificação

| Tipo de Problema | Neurônios na Saída | Ativação da Saída | Função de Perda (`loss`) | Métrica (`metrics`) |
| :--- | :--- | :--- | :--- | :--- |
| **Regressão Numérica** | **1 Neurônio** | `None` (Linear / Sem Ativação) | `'mse'` (Mean Squared Error) ou `'mae'` | `['mae', 'mse']` |
| **Classificação Binária** | **1 Neurônio** | `'sigmoid'` | `'binary_crossentropy'` | `['accuracy']` |
| **Classificação Multiclasse** | **$N$ Neurônios** | `'softmax'` | `'sparse_categorical_crossentropy'` ou `'categorical_crossentropy'` | `['accuracy']` |

> [!NOTE]
> 💡 **Curiosidade da Aula — Por que a Softmax Soma Exatamente 100%?**
> A função **Softmax** na camada de saída pega o vetor de pontuações brutas (chamadas *logits*) e calcula o exponencial de cada um dividido pela soma de todos os exponenciais ($\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$). Isso força a saída a se comportar como uma **Distribuição de Probabilidade Perfeita**, onde a soma de todas as classes dá exatamente $1.0$ ($100\%$!)  
> 🔗 **Documentação Keras Losses:** [keras.io/api/losses](https://keras.io/api/losses/)

> [!IMPORTANT]
> 💡 **Em 1 Frase:** Redes de Regressão usam 1 neurônio sem ativação na saída com perda MSE, enquanto redes de Classificação usam N neurônios com ativação Softmax para estimar probabilidades de cada classe.

---

## Módulo 2: O Mecanismo por Dentro & Formatação dos Rótulos

### 2.1 Sparse Categorical vs Categorical Crossentropy
1. **`sparse_categorical_crossentropy`:** Usa rótulos numéricos inteiros diretos ($y = [0, 1, 2, 0, \dots]$).
2. **`categorical_crossentropy`:** Exige matrizes **One-Hot Encoded** geradas via `tf.keras.utils.to_categorical(y)`.

> [!IMPORTANT]
> 💡 **Em 1 Frase:** Use `sparse_categorical_crossentropy` quando seus rótulos forem números inteiros simples (0, 1, 2) e `categorical_crossentropy` quando transformar os rótulos em matrizes One-Hot binárias.

---

## Módulo 3: Prática Guiada no Google Colab

Abra o [Google Colab](https://colab.research.google.com), crie um novo Notebook e acompanhe os blocos de código abaixo.

> [!NOTE]
> **Roteiro de estudo:** compare os dois tipos de problema antes de olhar o código: regressão prevê um número, enquanto classificação escolhe uma classe. No Keras, a estrutura geral é parecida; mudam principalmente a saída, a função de perda e a forma dos rótulos. One-Hot é apenas uma forma de representar uma categoria como números.

---

### Bloco 3.1 — Importação de Pacotes e Gerando Dados Sintéticos de Veículos

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que o script simula?**  
> Ele gera dados numéricos contínuos de 300 veículos contendo Potência (CV) e Peso (kg) como entradas $X$, e estima o Preço em reais como alvo numérico contínuo $y$.

```python
# 1. Importamos as bibliotecas de deep learning, dados e gráficos
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. Travamos a semente aleatória para dados reproduzíveis
np.random.seed(42)
n = 300
potencia_cv = np.random.uniform(70, 350, size=n)
peso_kg = np.random.uniform(900, 2500, size=n)
preco_reais = 180 * potencia_cv + 45 * peso_kg + np.random.normal(0, 3000, size=n)

# 3. Montamos a matriz de entradas X e o vetor de respostas y
matriz_entradas_veiculos = np.column_stack((potencia_cv, peso_kg))
vetor_respostas_precos = preco_reais

# 4. Exibimos a dimensão no console
print("--- DATASET SINTÉTICO DE VEÍCULOS GERADO ---")
print(f"📐 Dimensão dos Dados de Entrada: {matriz_entradas_veiculos.shape} (300 carros, 2 atributos)")
```

---

### Bloco 3.2 — Divisão Treino e Teste e Padronização (`StandardScaler`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que padronizamos a Potência e o Peso?**  
> Porque a Potência varia de 70 a 350 e o Peso varia de 900 a 2500. Sem o `StandardScaler`, a diferença de magnitude faria a rede dar peso excessivo ao Peso do veículo.

```python
# 1. Dividimos a base: 80% treino e 20% teste
matriz_entradas_treino, matriz_entradas_teste, vetor_respostas_treino, vetor_respostas_teste = train_test_split(
    matriz_entradas_veiculos, 
    vetor_respostas_precos, 
    test_size=0.20, 
    random_state=42
)

# 2. Instanciamos e aplicamos a padronização de escala
escalonador_padronizado = StandardScaler()
matriz_treino_padronizada = escalonador_padronizado.fit_transform(matriz_entradas_treino)
matriz_teste_padronizada = escalonador_padronizado.transform(matriz_entradas_teste)

# 3. Exibimos a contagem das amostras
print("--- DIVISÃO E PADRONIZAÇÃO DE DADOS ---")
print(f"📦 Treino: {len(matriz_treino_padronizada)} amostras | 🧪 Teste: {len(matriz_teste_padronizada)} amostras")
```

---

### Bloco 3.3 — Construindo a Rede Neural Keras de REGRESSÃO (`Sequential`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Como configuramos a camada de saída para estimar um número contínuo?**  
> Usamos **1 único neurônio** (`Dense(1)`) com ativação `None` (linear), permitindo que a rede produza qualquer valor numérico positivo ou negativo.

```python
# 1. Montamos o modelo sequencial do Keras para Regressão Numérica
modelo_regressao_keras = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)), # 1ª camada oculta com 64 neurônios
    tf.keras.layers.Dense(32, activation='relu'),                    # 2ª camada oculta com 32 neurônios
    tf.keras.layers.Dense(1, activation=None)                        # Camada de saída: 1 neurônio sem ativação
])

# 2. Exibimos o resumo da arquitetura da rede no console
print("--- ESTRUTURA DA REDE NEURAL DE REGRESSÃO KERAS ---")
modelo_regressao_keras.summary()
```

---

### Bloco 3.4 — Compilação com Perda MSE (`loss='mse'`, `metrics=['mae']`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que significa a perda `loss='mse'`?**  
> Significa *Mean Squared Error* (Erro Quadrático Médio). É a função matemática que penaliza os grandes erros quadráticos no treinamento de regressão.

```python
# 1. Compilamos a rede de regressão informando loss='mse' e métrica de erro MAE
modelo_regressao_keras.compile(
    optimizer='adam', 
    loss='mse', 
    metrics=['mae']
)

# 2. Exibimos a mensagem de confirmação
print("--- COMPILAÇÃO DA REDE NEURAL DE REGRESSÃO ---")
print("✅ Rede compilada com Perda MSE (Mean Squared Error) e Métrica MAE (Erro Médio Absoluto)!")
```

---

### Bloco 3.5 — Treinando a Rede de Regressão por 50 Épocas

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que faz o parâmetro `verbose=0`?**  
> O `verbose=0` silencia a impressão das barras de progresso linha por linha durante as 50 épocas, tornando o console mais limpo.

```python
# 1. Executamos o treinamento por 50 épocas nos dados padronizados
historico_regressao = modelo_regressao_keras.fit(
    matriz_treino_padronizada, 
    vetor_respostas_treino, 
    epochs=50, 
    batch_size=16, 
    verbose=0
)

# 2. Exibimos mensagem de confirmação de treino concluído
print("🎉 TREINAMENTO DA REDE NEURAL DE REGRESSÃO CONCLUÍDO COM SUCESSO!")
```

---

### Bloco 3.6 — Avaliação do Erro Médio Absoluto (MAE) no Teste

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Como interpretar o Erro Médio Absoluto (MAE)?**  
> O MAE informa o valor exato em reais que a rede neural está errando em média, para mais ou para menos, ao prever o preço dos veículos no conjunto de teste.

```python
# 1. Avaliamos a perda MSE e a métrica MAE nos dados de teste padronizados
perda_mse, erro_medio_absoluto_teste = modelo_regressao_keras.evaluate(matriz_teste_padronizada, vetor_respostas_teste, verbose=0)

# 2. Exibimos os relatórios numéricos no console
print("--- AVALIAÇÃO DE REGRESSÃO NO TESTE ---")
print(f"📊 PERDA MSE NO TESTE:                {perda_mse:,.2f}")
print(f"📊 ERRO MÉDIO ABSOLUTO (MAE) NO TESTE: R$ {erro_medio_absoluto_teste:,.2f}")
```

---

### Bloco 3.7 — Convertendo Rótulos Inteiros para One-Hot no Keras (`to_categorical`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Como a função `to_categorical` converte os rótulos?**  
> Ela transforma um número inteiro (ex: `2`) em um vetor binário de zeros com o número 1 na posição 2 (`[0, 0, 1]`), necessário para a perda `categorical_crossentropy`.

```python
# 1. Criamos um vetor de demonstração com rótulos de 3 classes (0, 1 ou 2)
vetor_rotulos_inteiros = np.array([0, 1, 2, 1, 0])

# 2. Convertemos o vetor em uma matriz One-Hot Encoded usando a função to_categorical do Keras
matriz_onehot_convertida = tf.keras.utils.to_categorical(vetor_rotulos_inteiros, num_classes=3)

# 3. Exibimos o resultado antes e depois da conversão
print("--- DEMONSTRAÇÃO DA CONVERSÃO TO_CATEGORICAL DO KERAS ---")
print("Vetor Inteiro Original:   ", vetor_rotulos_inteiros)
print("\nMatriz One-Hot Encoded:\n", matriz_onehot_convertida)
```

---

## Módulo 4: Prática Orientada & Experimentação Fácil

### Roteiro de Personalização para o Estudante:
Altere o código de previsão de preços de imóveis abaixo para testar diferentes épocas e camadas:

1. **Altere a quantidade de neurônios:** Na linha `quantidade_neuronios_escolhida = 64`, mude para `128` ou `32`.
2. **Altere as épocas:** Na linha `quantidade_epocas_escolhida = 60`, mude para `80` ou `100`.
3. **Re-execute e observe:** Veja como o erro MAE diminui com o novo treinamento!

```python
# =============================================================================
# CÓDIGO BASE PRONTO PARA SUA EXPERIMENTAÇÃO
# =============================================================================
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Geramos dados imobiliários sintéticos reproduzíveis
np.random.seed(88)
n = 200
tamanho_m2 = np.random.uniform(40, 220, size=n)
quartos = np.random.randint(1, 5, size=n)
preco_k = 3.9 * tamanho_m2 + 25 * quartos + 40 + np.random.normal(0, 25, size=n)

matriz_imoveis_x = np.column_stack((tamanho_m2, quartos))
vetor_imoveis_y = preco_k

# 2. Dividimos em treino e teste e padronizamos
matriz_treino, matriz_teste, respostas_treino, respostas_teste = train_test_split(matriz_imoveis_x, vetor_imoveis_y, test_size=0.25, random_state=88)

escalonador = StandardScaler()
matriz_treino_padronizada = escalonador.fit_transform(matriz_treino)
matriz_teste_padronizada = escalonador.transform(matriz_teste)

# -----------------------------------------------------------------------------
# STEP 1: PASSO DE EXPERIMENTAÇÃO DO ALUNO — PERSONALIZE NEURÔNIOS E ÉPOCAS:
# -----------------------------------------------------------------------------
quantidade_neuronios_escolhida = 64  # Tente mudar para 128 ou 32
quantidade_epocas_escolhida = 60     # Tente mudar para 80 ou 100 épocas

# 3. Criamos a rede neural de regressão do estudante
modelo_imoveis_estudante = tf.keras.Sequential([
    tf.keras.layers.Dense(quantidade_neuronios_escolhida, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation=None) # 1 Neurônio para estimar preço
])

# 4. Compilamos e treinamos
modelo_imoveis_estudante.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelo_imoveis_estudante.fit(matriz_treino_padronizada, respostas_treino, epochs=quantidade_epocas_escolhida, batch_size=16, verbose=0)

# 5. Avaliamos nos dados de teste
perda_mse, erro_mae_final = modelo_imoveis_estudante.evaluate(matriz_teste_padronizada, respostas_teste, verbose=0)

# 6. Exibimos os relatórios
print(f"--- RELATÓRIO DO SEU TESTE ({quantidade_neuronios_escolhida} Neurônios, {quantidade_epocas_escolhida} Épocas) ---")
print(f"📊 Erro Médio Absoluto (MAE) no Teste: R$ {erro_mae_final * 1000:,.2f}")
```

---

### 🔮 Aquecimento & Spoiler da Próxima Aula (Para Ir Além)

> [!TIP]
> 🧠 **Conceito-Semente — Ensinando a Rede a 'Desistir' no Momento Certo (EarlyStopping)**  
> Se você treinar uma rede por 500 épocas, ela vai memorizar os dados de treino perfeitamente, mas seu erro no teste começará a explodir (*Overfitting*). Em vez de chutar o número de épocas, na **Aula 19** usaremos **Callbacks**: o Keras monitora o teste a cada época e, assim que o erro começar a subir, ele puxa o freio de mão sozinho e salva a melhor versão no disco!
> 
> 🚀 **Desafio Proativo de Autoestudo (Opcional):**  
> Pesquise a técnica de **Dropout** em Redes Neurais. Por que 'desligar aleatoriamente 20% dos neurônios a cada rodada de treino' faz a rede ficar mais inteligente e resiliente?

---

## Módulo 5: Checklist de Autonomia do Estudante

- [ ] Sei configurar a camada de saída para Regressão (1 neurônio, ativação `None`, loss `'mse'`).
- [ ] Sei configurar a camada de saída para Classificação Multiclasse ($N$ neurônios, ativação `'softmax'`).
- [ ] Entendi a diferença entre `sparse_categorical_crossentropy` e `categorical_crossentropy`.
- [ ] Sei converter alvos numéricos em matrizes One-Hot com `tf.keras.utils.to_categorical`.
- [ ] Consigo alterar o número de épocas e neurônios no script e observar a variação do erro.

---

## Referências Bibliográficas & Documentações Oficiais

- 📖 **Livro Texto:** Géron, A. — *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow* (Capítulo 10).
- 🔗 **Documentação Keras Losses:** [keras.io/api/losses](https://keras.io/api/losses/)

# Aula 11 - Avaliação de Modelos - Parte 2 (Otimização de Hiperparâmetros & Regularização)

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  

---

> [!NOTE]
> 🔙 **De onde viemos:** Na Aula 10, aprendemos a mensurar a performance real dos modelos sem viés usando Validação Cruzada e Curvas ROC/AUC. Agora precisamos automatizar o processo de sintonia fina desses modelos para extrair o máximo de desempenho com o menor risco de sobreajuste.
> 🎯 **Objetivo Principal da Aula:** Automatizar a otimização de hiperparâmetros com **GridSearchCV** e **RandomizedSearchCV**, e aplicar técnicas de **Regularização (Lasso L1 e Ridge L2)** para penalizar a complexidade desnecessária e simplificar modelos preditivos.
> 🚀 **Para onde vamos:** Na próxima aula ('Aprendizado Não Supervisionado - Parte 1'), mudaremos radicalmente de paradigma: entraremos no mundo onde **NÃO EXISTEM RESPOSTAS CERTAS ($y$)** e aprenderemos como a máquina descobre grupos e personas de clientes sozinha com o algoritmo **K-Means**.

---

## Organização Tática da Aula

| Módulo | Atividade | Foco Pedagógico |
| :--- | :--- | :--- |
| **Módulo 1** | **Fundamentação Teórica & Parâmetros vs Hiperparâmetros** | O que a máquina aprende (`fit`) vs O que o humano configura, busca exaustiva vs aleatória. |
| **Módulo 2** | **Regularização L1 (Lasso) vs L2 (Ridge)** | Penalização da complexidade para conter coeficientes gigantescos e evitar overfitting. |
| **Módulo 3** | **Prática Guiada no Google Colab** | 7 blocos de código em Python minuciosamente comentados linha por linha com caixas de dúvidas. |
| **Módulo 4** | **Prática Orientada & Experimentação Fácil** | Otimização do grid de hiperparâmetros de uma Árvore de Decisão com GridSearchCV. |
| **Módulo 5** | **Checklist de Autonomia & Bibliografia** | Autoavaliação do estudante e referências oficiais. |

---

## Módulo 1: Fundamentação Teórica & Otimização de Hiperparâmetros

### 1.1 Parâmetros Internos vs. Hiperparâmetros
- **Parâmetros do Modelo:** Ajustados **automaticamente pela máquina** durante o comando `.fit()` (ex: os coeficientes $a$ e $b$ da reta de regressão, os pesos de uma Rede Neural).
- **Hiperparâmetros:** Configurações definidas **pelo Cientista de Dados antes do treinamento** (ex: o número $K$ no KNN, a profundidade `max_depth` na Árvore, o número de árvores `n_estimators` no Random Forest).

### 1.2 Estratégias de Busca Automática

1. **`GridSearchCV` (Busca em Grade Exaustiva):** Testa TODAS as combinações possíveis do dicionário.
2. **`RandomizedSearchCV` (Busca Aleatória Amostrada):** Sorteia $N$ combinações aleatórias dentro do espaço de busca.

> [!TIP]
> 🧙‍♂️ **Analogia Geek — O Tuning de Feitiços em um RPG:**
> Encontrar os melhores hiperparâmetros para um algoritmo é como tunar a varinha ou feitiço de um Mago. Em vez de testar cada combinação na mão (o que levaria dias), o **GridSearchCV** testa automaticamente todas as combinações de elementos mágicos e te entrega o cajado com o maior dano por segundo (maior Acurácia)!

> [!NOTE]
> 💡 **Curiosidade da Aula — Por que a Regularização Lasso (L1) Consegue Zerar Atributos?**
> A penalidade da Regularização L1 ($|w|$) cria um espaço de restrição em formato de **diamante (losango)** com cantos pontiagudos nos eixos. Quando a elipse da função de erro encosta nas pontas desse diamante, os coeficientes de atributos irrelevantes se tornam **exatamente ZERO**! Ou seja, o Lasso faz uma seleção natural de atributos embutida na equação matemática!  
> 🔗 **Documentação Scikit-Learn:** [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

> [!IMPORTANT]
> 💡 **Em 1 Frase:** O GridSearchCV testa todas as combinações possíveis de hiperparâmetros configurados pelo cientista de dados, encontrando a combinação exata que maximiza o desempenho do modelo.

---

## Módulo 2: O Mecanismo por Dentro & Regularização L1/L2

### 2.1 Penalização da Complexidade (Regularização)

$$\text{Custo} = \text{Erro Tradicional (MSE)} + \text{Penalidade}(\alpha)$$

1. **Regularização L2 (Ridge Regression):** Penaliza o quadrado dos coeficientes ($\alpha \sum w_i^2$). Encolhe os coeficientes para valores próximos de zero, mas nunca zera nenhum atributo.
2. **Regularização L1 (Lasso Regression):** Penaliza o valor absoluto dos coeficientes ($\alpha \sum |w_i|$). Consegue **zerar completamente** atributos irrelevantes, agindo como uma seleção de atributos embutida!

> [!IMPORTANT]
> 💡 **Em 1 Frase:** A regularização Ridge (L2) encolhe pesos exagerados para estabilizar o modelo, enquanto a regularização Lasso (L1) zera coeficientes de variáveis inúteis para simplificar a solução.

---

## Módulo 3: Prática Guiada no Google Colab

Abra o [Google Colab](https://colab.research.google.com), crie um novo Notebook e acompanhe os blocos de código abaixo.

> [!NOTE]
> **Roteiro de estudo:** pense em hiperparâmetros como opções de configuração escolhidas antes do treino. `GridSearchCV` e `RandomizedSearchCV` testam combinações dessas opções automaticamente. Nesta primeira exploração, acompanhe a pergunta “qual configuração teve melhor resultado?” sem tentar decorar cada parâmetro.

---

### Bloco 3.1 — Importação de Módulos e Carregando o Wine Dataset

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Quais ferramentas do Scikit-Learn são usadas para otimização de hiperparâmetros e regularização?**  
> `GridSearchCV` para busca exaustiva, `RandomizedSearchCV` para busca por amostragem rápida, `Ridge` para regularização L2 e `Lasso` para regularização L1.

```python
# 1. Importamos as bibliotecas de processamento e dados
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import accuracy_score, r2_score

# 2. Carregamos o dataset de vinhos
dados_brutos_vinho = load_wine()
matriz_entradas_vinho = dados_brutos_vinho.data
vetor_respostas_vinho = dados_brutos_vinho.target

# 3. Dividimos em conjunto de treino (75%) e teste (25%) com estratificação
matriz_entradas_treino, matriz_entradas_teste, vetor_respostas_treino, vetor_respostas_teste = train_test_split(
    matriz_entradas_vinho, 
    vetor_respostas_vinho, 
    test_size=0.25, 
    random_state=42, 
    stratify=vetor_respostas_vinho
)

# 4. Exibimos a contagem de amostras no console
print("--- CHECAGEM DE DADOS PARA BUSCA DE HIPERPARÂMETROS ---")
print(f"📦 Treino: {len(matriz_entradas_treino)} amostras | 🧪 Teste: {len(matriz_entradas_teste)} amostras")
```

---

### Bloco 3.2 — Criando o Dicionário de Hiperparâmetros para Busca

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Como calcular quantas combinações serão testadas na grade?**  
> Basta multiplicar o tamanho de cada lista no dicionário: 2 (opções de `n_estimators`) $\times$ 3 (opções de `max_depth`) $\times$ 2 (opções de `criterion`) = **12 combinações únicas**. Com `cv=5`, o `GridSearchCV` rodará $12 \times 5 = 60$ treinos automaticamente!

```python
# 1. Definimos o dicionário contendo as listas de opções de hiperparâmetros a serem testados
grade_hiperparametros_busca = {
    'n_estimators': [50, 100],         # Quantidade de árvores na floresta
    'max_depth': [3, 5, None],          # Profundidade máxima por árvore
    'criterion': ['gini', 'entropy']    # Métrica de divisão por impureza
}

# 2. Exibimos a grade configurada
print("--- GRADE DE HIPERPARÂMETROS CONFIGURADA ---")
print(grade_hiperparametros_busca)
```

---

### Bloco 3.3 — Instanciando e Executando o `GridSearchCV`

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que faz o parâmetro `n_jobs=-1` no GridSearchCV?**  
> Ele instrui o Scikit-Learn a utilizar **todos os núcleos do processador** da sua máquina ou do servidor em nuvem em paralelo, acelerando dramaticamente a busca!

```python
# 1. Instanciamos o objeto GridSearchCV integrando a Floresta Aleatória e a validação cruzada 5-Fold
busca_em_grade = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=grade_hiperparametros_busca,
    cv=5,
    scoring='accuracy',
    n_jobs=-1 # Usa processamento paralelo em todos os núcleos da CPU
)

# 2. O MOMENTO DA BUSCA: O algoritmo testa todas as combinações da grade
busca_em_grade.fit(matriz_entradas_treino, vetor_respostas_treino)

# 3. Exibimos o campeão das combinações e sua acurácia média de validação
print("--- RESULTADO DO GRIDSEARCHCV (BUSCA EXAUSTIVA) ---")
print(f"🏆 Melhor Acurácia na Validação Cruzada: {busca_em_grade.best_score_ * 100:.2f}%")
print(f"🔧 Melhores Hiperparâmetros Vencedores:   {busca_em_grade.best_params_}")
```

---

### Bloco 3.4 — Avaliação do Melhor Modelo no Conjunto de Teste

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que o atributo `.best_estimator_` contém?**  
> Ele recupera automaticamente o modelo de Inteligência Artificial **já retreinado com a combinação vencedora de hiperparâmetros**, pronto para gerar previsões em dados de teste.

```python
# 1. Extraímos o modelo campeão já treinado
melhor_modelo_floresta = busca_em_grade.best_estimator_

# 2. Geramos previsões nos dados de teste usando o modelo vencedor
previsoes_melhor_modelo = melhor_modelo_floresta.predict(matriz_entradas_teste)

# 3. Calculamos a acurácia final nos dados de teste não vistos
acuracia_melhor_modelo = accuracy_score(vetor_respostas_teste, previsoes_melhor_modelo)

# 4. Exibimos o resultado no console
print("--- AVALIAÇÃO DO MODELO VENCEDOR NO TESTE ---")
print(f"📊 Acurácia Final do Modelo Vencedor no Teste: {acuracia_melhor_modelo * 100:.2f}%")
```

---

### Bloco 3.5 — Executando a Busca Aleatória (`RandomizedSearchCV`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Quando usar `RandomizedSearchCV` em vez do `GridSearchCV`?**  
> Quando sua grade de parâmetros for gigantesca (milhares de combinações) e você não tiver horas para esperar. O `RandomizedSearchCV` sorteia apenas $N$ combinações (`n_iter=5`) obtendo um resultado muito próximo em uma fração do tempo.

```python
# 1. Instanciamos a busca aleatória sortendo apenas 5 combinações (n_iter=5)
busca_aleatoria = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=grade_hiperparametros_busca,
    n_iter=5,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# 2. O MOMENTO DA BUSCA ALEATÓRIA
busca_aleatoria.fit(matriz_entradas_treino, vetor_respostas_treino)

# 3. Exibimos a acurácia alcançada pelo modelo sorteado campeão
print("--- RESULTADO DO RANDOMIZEDSEARCHCV (BUSCA ALEATÓRIA) ---")
print(f"🎲 Melhor Acurácia no RandomizedSearch: {busca_aleatoria.best_score_ * 100:.2f}%")
```

---

### Bloco 3.6 — Regularização L2 (Ridge Regression)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que o hiperparâmetro `alpha=1.0` faz no Ridge?**  
> O `alpha` (ou $\lambda$) controla a intensidade da penalidade. Quanto maior o `alpha`, mais o Ridge encolhe os coeficientes das variáveis para evitar que o modelo decore ruídos.

```python
# 1. Criamos dados sintéticos com 10 atributos
np.random.seed(42)
matriz_entradas_regressao = np.random.randn(100, 10)
vetor_respostas_regressao = 3 * matriz_entradas_regressao[:, 0] - 2 * matriz_entradas_regressao[:, 1] + np.random.randn(100) * 0.5

# 2. Instanciamos e treinamos o modelo Ridge (Regularização L2)
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(matriz_entradas_regressao, vetor_respostas_regressao)

# 3. Exibimos os coeficientes ajustados (note que nenhum coeficiente é zerado)
print("--- COEFICIENTES DA REGRESSÃO RIDGE (L2) ---")
print(modelo_ridge.coef_.round(3))
```

---

### Bloco 3.7 — Regularização L1 (Lasso Regression)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que o Lasso é considerado um selecionador natural de variáveis?**  
> Porque devido à matemática da norma L1 ($|w|$), ele força os coeficientes de atributos irrelevantes a virarem **exatamente $0.0$**, eliminando colunas inúteis do cálculo.

```python
# 1. Instanciamos o modelo Lasso com penalidade alpha=0.3
modelo_lasso = Lasso(alpha=0.3)

# 2. Treinamos o Lasso nos mesmos dados
modelo_lasso.fit(matriz_entradas_regressao, vetor_respostas_regressao)

# 3. Extraímos o vetor de coeficientes aprendidos
coeficientes_regressao_lasso = modelo_lasso.coef_

# 4. Exibimos os coeficientes e a contagem de colunas totalmente zeradas
print("--- COEFICIENTES DA REGRESSÃO LASSO (L1 - SELEÇÃO NATURAL DE ATRIBUTOS) ---")
print(coeficientes_regressao_lasso.round(3))
print(f"✂️ Total de Atributos Zerados pelo Lasso: {np.sum(coeficientes_regressao_lasso == 0)} de 10 colunas!")
```

---

## Módulo 4: Prática Orientada & Experimentação Fácil

### Roteiro de Personalização para o Estudante:
Altere a grade de hiperparâmetros da Árvore de Decisão abaixo para personalizar a sua busca:

1. **Altere os valores da grade:** Adicione ou remova valores na lista `'max_depth'` no comentário 1 (ex: inclua `[2, 4, 6, 8]`).
2. **Re-execute e observe:** Veja qual foi a combinação vencedora escolhida pelo `GridSearchCV` para o seu teste!

```python
# =============================================================================
# CÓDIGO BASE PRONTO PARA SUA EXPERIMENTAÇÃO
# =============================================================================
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Geramos dados sintéticos de concessão de crédito
np.random.seed(77)
n = 120
idade = np.random.randint(18, 70, size=n)
score = np.random.randint(300, 850, size=n)
aprovado = ((idade > 25) & (score > 600)).astype(int)

X_matriz = np.column_stack((idade, score))
y_vetor = aprovado

# 2. Dividimos em treino e teste
matriz_treino, matriz_teste, respostas_treino, respostas_teste = train_test_split(X_matriz, y_vetor, test_size=0.25, random_state=77)

modelo_arvore_base = DecisionTreeClassifier(random_state=77)

# -----------------------------------------------------------------------------
# STEP 1: PASSO DE EXPERIMENTAÇÃO DO ALUNO — PERSONALIZE OS VALORES DA GRADE:
# -----------------------------------------------------------------------------
grade_hiperparametros_estudante = {
    'max_depth': [2, 4, 6, 8],            # Tente mudar a lista de profundidades!
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10]
}

# 3. Executamos o GridSearchCV do estudante com a grade personalizada
busca_grid_estudante = GridSearchCV(estimator=modelo_arvore_base, param_grid=grade_hiperparametros_estudante, cv=5, scoring='accuracy')
busca_grid_estudante.fit(matriz_treino, respostas_treino)

# 4. Exibimos os relatórios da otimização
print("--- RELATÓRIO DA SUA OTIMIZAÇÃO PERSONALIZADA ---")
print(f"🏆 Melhor Acurácia no Grid: {busca_grid_estudante.best_score_ * 100:.2f}%")
print(f"🔧 Melhores Parâmetros Vencedores: {busca_grid_estudante.best_params_}")

acuracia_teste_final = accuracy_score(respostas_teste, busca_grid_estudante.best_estimator_.predict(matriz_teste))
print(f"📊 Acurácia Final no Teste: {acuracia_teste_final * 100:.2f}%")
```

---

### 🔮 Aquecimento & Spoiler da Próxima Aula (Para Ir Além)

> [!TIP]
> 🧠 **Conceito-Semente — Como aprender quando não há gabarito ($y$)?**  
> Até hoje, sempre demos o gabarito para a máquina: *'aqui estão os dados e esta casa custa R$ 500k'* ou *'este cliente é inadimplente'*. Mas imagine que uma loja tem 50.000 clientes e quer dividi-los em 3 perfis de consumo sem saber de antemão quem é quem. Na **Aula 12**, iniciaremos o **Aprendizado Não Supervisionado** com o **K-Means**, onde a máquina agrupa os pontos por proximidade espacial!
> 
> 🚀 **Desafio Proativo de Autoestudo (Opcional):**  
> Como você decidiria matematicamente se uma base de clientes deve ser dividida em 3, 4 ou 5 grupos? Pense em como medir se os grupos formados ficaram 'juntinhos' ou 'espalhados'.

---

## Módulo 5: Checklist de Autonomia do Estudante

- [ ] Compreendi a diferença entre Parâmetros do modelo e Hiperparâmetros.
- [ ] Sei implementar o `GridSearchCV` para encontrar as melhores combinações de parâmetros.
- [ ] Entendi a diferença prática entre Regularização L2 (Ridge) e L1 (Lasso).
- [ ] Consigo extrair o melhor modelo treinado através da propriedade `.best_estimator_`.
- [ ] Sei alterar os valores do dicionário de busca no script.

---

## Referências Bibliográficas & Documentações Oficiais

- 📖 **Livro Texto:** Géron, A. — *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow* (Capítulo 2: Ajuste Fino do Modelo).
- 🔗 **Documentação Scikit-Learn:** [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

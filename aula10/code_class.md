# Aula 10 - Avaliação de Modelos - Parte 1 (Validação Cruzada, Curvas ROC & AUC)

**Disciplina:** Aprendizado de Máquina e Redes Neurais  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  

---

> [!NOTE]
> 🔙 **De onde viemos:** Na Aula 09, aprendemos algoritmos avançados como Árvores de Decisão, Random Forest e SVM.
> 🎯 **Objetivo Principal da Aula:** Dominar a **Validação Cruzada (K-Fold e Stratified K-Fold)**, plotar a **Curva ROC**, calcular o **AUC (Area Under Curve)** e ajustar o **Limiar de Decisão (Threshold)**.
> 🚀 **Para onde vamos:** Na próxima aula ('Avaliação de Modelos - Parte 2'), utilizaremos estas métricas avançadas para otimizar hiperparâmetros automaticamente via GridSearchCV e RandomizedSearchCV.

---

## Organização Tática da Aula

| Módulo | Atividade | Foco Pedagógico |
| :--- | :--- | :--- |
| **Módulo 1** | **Fundamentação Teórica & Os Riscos do Train/Test Split** | Por que uma única divisão treino/teste pode gerar sorte amostral e o conceito de Validação Cruzada ($K$-Fold). |
| **Módulo 2** | **Curvas ROC / AUC & Ajuste do Limiar (Threshold)** | Taxa de Verdadeiros Positivos (TPR) vs Falsos Positivos (FPR), o significado do AUC e variação do threshold. |
| **Módulo 3** | **Prática Guiada no Google Colab** | 7 blocos de código em Python minuciosamente comentados linha por linha com caixas de dúvidas. |
| **Módulo 4** | **Prática Orientada & Experimentação Fácil** | Validação cruzada com variação de dobras $K$ em dataset de risco de crédito. |
| **Módulo 5** | **Checklist de Autonomia & Bibliografia** | Autoavaliação do estudante e referências oficiais. |

---

## Módulo 1: Fundamentação Teórica & Validação Cruzada

### 1.1 O Perigo de Confiar em uma Única Divisão Treino/Teste
Até aqui, usamos o `train_test_split` dividindo os dados uma única vez. No entanto, se o conjunto de teste der a "sorte" de conter apenas exemplos fáceis, o modelo parecerá incrível; se contiver amostras difíceis, parecerá péssimo.

A **Validação Cruzada K-Fold ($K$-Fold Cross-Validation)** elimina essa incerteza amostral dividindo os dados em $K$ partes (dobras/folds) iguais:

```mermaid
graph TD
    subgraph Validação Cruzada 5-Fold
    A1["Dobra 1: TESTE | Dobras 2,3,4,5: TREINO -> Acurácia 1 = 92%"]
    A2["Dobra 2: TESTE | Dobras 1,3,4,5: TREINO -> Acurácia 2 = 89%"]
    A3["Dobra 3: TESTE | Dobras 1,2,4,5: TREINO -> Acurácia 3 = 94%"]
    A4["Dobra 4: TESTE | Dobras 1,2,3,5: TREINO -> Acurácia 4 = 91%"]
    A5["Dobra 5: TESTE | Dobras 1,2,3,4: TREINO -> Acurácia 5 = 93%"]
    end
    A1 & A2 & A3 & A4 & A5 --> B["📊 MÉDIA FINAL ROBUSTA: 91.8% ± 1.7%"]
```

> [!TIP]
> 🖖 **Analogia Geek — O Teste Kobayashi Maru de Star Trek:**
> Submeter um modelo a uma única divisão treino/teste é como fazer uma simulação simples. A **Validação Cruzada $K$-Fold** é o teste *Kobayashi Maru*: ela testa a resistência do algoritmo sob TODAS as variações possíveis de cenários para garantir que ele não entre em colapso no espaço real!

> [!NOTE]
> 💡 **Curiosidade da Aula — A Origem Militar do Nome "Curva ROC":**
> A Curva ROC (Receiver Operating Characteristic) foi criada durante a **Segunda Guerra Mundial pelos operadores de RADAR britânicos e americanos**. Eles precisavam medir a capacidade dos radaristas de diferenciar sinal real (aviões inimigos se aproximando = Verdadeiros Positivos) de ruídos e interferências (bandos de pássaros = Falsos Positivos). Mais tarde, a medicina e a inteligência artificial adotaram o mesmo gráfico!  
> 🎥 **Vídeo Recomendado:** [StatQuest: ROC and AUC (YouTube)](https://www.youtube.com/watch?v=4jRBRDbJemM)

> [!IMPORTANT]
> 💡 **Em 1 Frase:** A Validação Cruzada K-Fold divide os dados em K pedaços e treina K vezes, garantindo que todo o dataset seja usado tanto no treino quanto no teste para evitar a "sorte amostral".

---

## Módulo 2: O Mecanismo por Dentro & Curva ROC e AUC

### 2.1 A Curva ROC e a Métrica AUC
A **Curva ROC** avalia o desempenho de um classificador probabilístico em **todos os limiares de decisão possíveis** (de $0.0$ a $1.0$).

- **Eixo Y:** Taxa de Verdadeiros Positivos / Recall / Sensibilidade ($\text{TPR} = \frac{VP}{VP + FN}$).
- **Eixo X:** Taxa de Falsos Positivos ($\text{FPR} = \frac{FP}{FP + VN}$).
- **AUC (Área Sob a Curva):** Quantifica o gráfico em um único número de $0.0$ a $1.0$:
  - $\text{AUC} = 0.50$: Modelo chuta aleatoriamente (igual a jogar moeda).
  - $\text{AUC} = 0.80 - 0.90$: Bom classificador.
  - $\text{AUC} = 1.00$: Classificador Perfeito!

> [!IMPORTANT]
> 💡 **Em 1 Frase:** A Curva ROC plota o equilíbrio entre os acertos (TPR) e alarmes falsos (FPR), e a pontuação AUC mede a capacidade global do modelo em separar as duas classes (sendo 1.0 a perfeição).

---

## Módulo 3: Prática Guiada no Google Colab

Abra o [Google Colab](https://colab.research.google.com), crie um novo Notebook e acompanhe os blocos de código abaixo.

> [!NOTE]
> **Roteiro de estudo:** até aqui avaliamos um modelo em uma divisão de treino e teste. Agora vamos observar a avaliação por vários recortes dos dados. Antes de focar nas siglas, guarde a ideia central: validação cruzada verifica se o resultado se mantém estável; ROC e AUC mostram como o modelo separa as classes em diferentes limiares.

---

### Bloco 3.1 — Importando Bibliotecas e Carregando o Dataset Médico

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que contêm os dados de `load_breast_cancer`?**  
> Contêm 569 amostras de biópsias de tumores com 30 características clínicas (ex: raio, textura, área) e rótulos indicando se o tumor é Maligno (1) ou Benigno (0).

```python
# 1. Importamos as bibliotecas de processamento numérico e gráfico
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Importamos o dataset médico de diagnóstico de câncer do Scikit-Learn
from sklearn.datasets import load_breast_cancer

# 3. Importamos os utilitários de validação cruzada, divisão e padronização
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

# 4. Carregamos o objeto bruto de dados
dados_brutos_cancer = load_breast_cancer()
matriz_entradas_medicas = dados_brutos_cancer.data
vetor_respostas_medicas = dados_brutos_cancer.target

# 5. Exibimos a confirmação de carregamento
print("--- DATASET MÉDICO CARREGADO ---")
print(f"📐 Dimensão do Dataset: {matriz_entradas_medicas.shape} (569 pacientes, 30 variáveis médicas)")
```

---

### Bloco 3.2 — Executando Validação Cruzada 5-Fold (`cross_val_score`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que o parâmetro `cv=5` faz exatamente?**  
> Ele divide o dataset em 5 partes iguais, treina o modelo 5 vezes (usando 4 partes para treino e 1 para teste em cada rodada) e retorna as 5 pontuações individuais de acurácia.

```python
# 1. Instanciamos a Regressão Logística com limite alto de iterações para garantir convergência
modelo_logistico = LogisticRegression(max_iter=5000, random_state=42)

# 2. Executamos a validação cruzada 5-Fold nos dados completos
pontuacoes_validacao_cruzada = cross_val_score(
    modelo_logistico, 
    matriz_entradas_medicas, 
    vetor_respostas_medicas, 
    cv=5, 
    scoring='accuracy'
)

# 3. Exibimos o relatório com as 5 acurácias, a média e o desvio padrão no console
print("--- RESULTADOS DA VALIDAÇÃO CRUZADA (5-FOLD) ---")
print(f"Pontuações das 5 dobras: {pontuacoes_validacao_cruzada.round(4)}")
print(f"📊 Acurácia Média Final: {pontuacoes_validacao_cruzada.mean() * 100:.2f}% ± {pontuacoes_validacao_cruzada.std() * 100:.2f}%")
```

---

### Bloco 3.3 — Executando Stratified K-Fold Recomendado para Manter Proporções

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Qual a diferença entre `KFold` e `StratifiedKFold`?**  
> O `StratifiedKFold` embaralha e garante que a proporção exata de tumores malignos/benignos seja idêntica em cada uma das 5 dobras, sendo o método ideal para problemas de classificação.

```python
# 1. Criamos a estrutura do validador estratificado com 5 dobras e embaralhamento ativo
validador_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Executamos a validação cruzada alinhada ao validador estratificado
pontuacoes_estratificadas = cross_val_score(
    modelo_logistico, 
    matriz_entradas_medicas, 
    vetor_respostas_medicas, 
    cv=validador_estratificado, 
    scoring='accuracy'
)

# 3. Exibimos a acurácia média e a estabilidade obtida
print("--- RESULTADOS DA VALIDAÇÃO CRUZADA ESTRATIFICADA (STRATIFIED 5-FOLD) ---")
print(f"Pontuações das dobras: {pontuacoes_estratificadas.round(4)}")
print(f"📊 Acurácia Média Estratificada: {pontuacoes_estratificadas.mean() * 100:.2f}% ± {pontuacoes_estratificadas.std() * 100:.2f}%")
```

---

### Bloco 3.4 — Divisão Treino/Teste e Padronização para Curva ROC

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que padronizamos os dados antes de treinar a Regressão Logística para a Curva ROC?**  
> Para colocar as 30 medições médicas na mesma escala (`StandardScaler`), permitindo que a Regressão Logística calcule probabilidades contínuas sem viés numérico.

```python
# 1. Separamos 30% dos dados para o teste final da curva ROC com estratificação
matriz_entradas_treino, matriz_entradas_teste, vetor_respostas_treino, vetor_respostas_teste = train_test_split(
    matriz_entradas_medicas, 
    vetor_respostas_medicas, 
    test_size=0.30, 
    random_state=42, 
    stratify=vetor_respostas_medicas
)

# 2. Instanciamos e aplicamos o escalonador padronizado
escalonador_padronizado = StandardScaler()
matriz_treino_padronizada = escalonador_padronizado.fit_transform(matriz_entradas_treino)
matriz_teste_padronizada = escalonador_padronizado.transform(matriz_entradas_teste)

# 3. Treinamos o modelo logístico nos dados de treino padronizados
modelo_logistico.fit(matriz_treino_padronizada, vetor_respostas_treino)
```

---

### Bloco 3.5 — Extraindo Probabilidades com `.predict_proba()`

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que selecionamos a segunda coluna `[:, 1]` no `.predict_proba()`?**  
> O `.predict_proba()` retorna uma matriz com 2 colunas: `[Prob_Benigno, Prob_Maligno]`. A segunda coluna `[:, 1]` contém a probabilidade do caso ser positivo (Maligno), necessária para desenhar a curva ROC.

```python
# 1. Extraímos as probabilidades contínuas de pertencer à classe positiva (coluna 1)
probabilidades_teste_positivas = modelo_logistico.predict_proba(matriz_teste_padronizada)[:, 1]

# 2. Exibimos as 5 primeiras probabilidades calculadas no console
print("--- PROBABILIDADES DE DIAGNÓSTICO CALCULADAS ---")
print("Primeiras 5 probabilidades contínuas calculadas:")
print(probabilidades_teste_positivas[:5].round(4))
```

---

### Bloco 3.6 — Calculando as Taxas da Curva ROC (`roc_curve` e `roc_auc_score`)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que a função `roc_curve` calcula?**  
> Ela testa dezenas de limiares de decisão e calcula a Taxa de Falsos Positivos (`FPR`) e a Taxa de Verdadeiros Positivos (`TPR`) para cada limiar.

```python
# 1. Calculamos os vetores de taxas FPR, TPR e os limiares correspondentes
taxa_falsos_positivos, taxa_verdadeiros_positivos, limiares_decisao = roc_curve(
    vetor_respostas_teste, 
    probabilidades_teste_positivas
)

# 2. Calculamos o valor numérico único da área sob a curva (AUC)
pontuacao_auc = roc_auc_score(vetor_respostas_teste, probabilidades_teste_positivas)

# 3. Exibimos o resultado do AUC no console
print("--- MÉTRICA AUC CALCULADA ---")
print(f"🌟 Pontuação AUC (Area Under Curve): {pontuacao_auc:.4f} (Desempenho excelente!)")
```

---

### Bloco 3.7 — Plotando o Gráfico da Curva ROC

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Como interpretar a linha vermelha pontilhada no gráfico ROC?**  
> Ela representa a linha do "chute aleatório" ($\text{AUC} = 0.50$). Quanto mais a linha azul do seu modelo curvar em direção ao canto superior esquerdo (afastando-se da linha vermelha), melhor é a capacidade de classificação da IA!

```python
# 1. Definimos as dimensões da figura do gráfico
plt.figure(figsize=(7, 5))

# 2. Desenhamos a Curva ROC azul do modelo e a linha vermelha pontilhada do chute aleatório
plt.plot(taxa_falsos_positivos, taxa_verdadeiros_positivos, color='blue', linewidth=2.5, label=f'Regressão Logística (AUC = {pontuacao_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', label='Chute Aleatório (AUC = 0.50)')

# 3. Adicionamos rótulos e legendas explicativas ao gráfico
plt.title("Curva ROC — Avaliação de Diagnóstico Médico", fontweight='bold')
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 4. Renderizamos o gráfico no notebook
print("🎨 Exibindo o gráfico da Curva ROC no Colab...")
plt.show()
```

---

## Módulo 4: Prática Orientada & Experimentação Fácil

### Roteiro de Personalização para o Estudante:
Altere o número de dobras da Validação Cruzada abaixo e veja como a média se comporta:

1. **Altere as dobras:** Na linha `quantidade_dobras_k = 5`, mude para `3`, `8` ou `10`.
2. **Re-execute e observe:** Veja como o desvio padrão varia de acordo com a quantidade de dobras escolhida por você!

```python
# =============================================================================
# CÓDIGO BASE PRONTO PARA SUA EXPERIMENTAÇÃO
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. Geramos dados bancários simulados reproduzíveis
np.random.seed(88)
n = 150
renda = np.random.uniform(2000, 15000, size=n)
divida = np.random.uniform(500, 50000, size=n)
inadimplente = ((divida / renda) > 3.0).astype(int)

df_banco = pd.DataFrame({'renda': renda, 'divida': divida, 'inadimplente': inadimplente})
matriz_banco_x = df_banco[['renda', 'divida']].values
vetor_banco_y = df_banco['inadimplente'].values

# -----------------------------------------------------------------------------
# STEP 1: PASSO DE EXPERIMENTAÇÃO DO ALUNO — ALTERE A QUANTIDADE DE DOBRAS K:
# -----------------------------------------------------------------------------
quantidade_dobras_k = 5 # Tente mudar para 3, 8 ou 10 dobras!

# 2. Instanciamos a Floresta Aleatória e executamos a Validação Cruzada calculando AUC
floresta_banco = RandomForestClassifier(n_estimators=50, random_state=88)
pontuacoes_auc_dobras = cross_val_score(
    floresta_banco, 
    matriz_banco_x, 
    vetor_banco_y, 
    cv=quantidade_dobras_k, 
    scoring='roc_auc'
)

# 3. Exibimos os relatórios da sua experimentação
print(f"--- RESULTADO DA SUA VALIDAÇÃO CRUZADA ({quantidade_dobras_k}-FOLD) ---")
print(f"Pontuações AUC nas {quantidade_dobras_k} dobras: {pontuacoes_auc_dobras.round(4)}")
print(f"📊 Pontuação AUC Média Final: {pontuacoes_auc_dobras.mean():.4f} ± {pontuacoes_auc_dobras.std():.4f}")
```

---

## Módulo 5: Checklist de Autonomia do Estudante

- [ ] Compreendi as limitações de confiar em uma única divisão treino/teste.
- [ ] Sei implementar a Validação Cruzada K-Fold com `cross_val_score`.
- [ ] Entendi como interpretar a Média e o Desvio Padrão das dobras da Validação Cruzada.
- [ ] Sei plotar e interpretar a Curva ROC e a métrica AUC (Area Under Curve).
- [ ] Consigo alterar a quantidade de dobras no script e observar a variação.

---

## Referências Bibliográficas & Documentações Oficiais

- 📖 **Livro Texto:** Géron, A. — *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow* (Capítulo 3: Validação Cruzada e Curvas ROC).
- 🔗 **Documentação Scikit-Learn:** [scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

# Aprendizado de Máquina e Redes Neurais

> **Instituição:** UNIVEM — Centro Universitário de Marília  
> **Professor:** Eduardo Lázaro Roesler de Oliveira  
> **Semestre:** Material Oficial de Apoio às Aulas Ministradas  

---

## 📌 Sobre a Disciplina

Bem-vindo(a) ao repositório oficial da disciplina **Aprendizado de Máquina e Redes Neurais**!

Este repositório contém todo o conteúdo prático e conceitual do semestre. Para cada aula ministrada, você encontrará dois materiais fundamentais:
1. **`code_class.md` (Guia Didático e Conceitual):** Explicação da matéria, analogias do cotidiano, blocos comentados passo a passo e seções de spoiler/aquecimento para a próxima semana.
2. **`pratica_real.md` (Laboratório Técnico com Datasets Reais):** Código direto, sequencial e sem dados simulados — utilizando bases consagradas da Ciência de Dados (Titanic, California Housing, Breast Cancer, Fashion-MNIST, etc.).

---

## 🗺️ Mapa Curricular das 20 Aulas

| Aula | Tema da Aula | Guia Didático | Laboratório Prático | Dataset do Laboratório |
| :---: | :--- | :---: | :---: | :--- |
| **01** | Introdução à IA & Primeiro Pipeline de ML | [code_class.md](./aula01/code_class.md) | [pratica_real.md](./aula01/pratica_real.md) | `sns.load_dataset('penguins')` |
| **02** | NumPy & Computação Numérica Vetorizada | [code_class.md](./aula02/code_class.md) | [pratica_real.md](./aula02/pratica_real.md) | `load_diabetes()` |
| **03** | Pandas DataFrames & Exploração de Tabelas | [code_class.md](./aula03/code_class.md) | [pratica_real.md](./aula03/pratica_real.md) | `sns.load_dataset('tips')` |
| **04** | Limpeza de Dados: Nulos (`NaN`) & Outliers (IQR) | [code_class.md](./aula04/code_class.md) | [pratica_real.md](./aula04/pratica_real.md) | `sns.load_dataset('titanic')` |
| **05** | Feature Engineering, Scaling & Encoding | [code_class.md](./aula05/code_class.md) | [pratica_real.md](./aula05/pratica_real.md) | `sns.load_dataset('diamonds')` |
| **06** | Visualização de Dados (Matplotlib & Seaborn) | [code_class.md](./aula06/code_class.md) | [pratica_real.md](./aula06/pratica_real.md) | `sns.load_dataset('penguins')` |
| **07** | Regressão Linear Simples e Múltipla | [code_class.md](./aula07/code_class.md) | [pratica_real.md](./aula07/pratica_real.md) | `fetch_california_housing()` |
| **08** | Classificação: Regressão Logística & KNN | [code_class.md](./aula08/code_class.md) | [pratica_real.md](./aula08/pratica_real.md) | `load_breast_cancer()` |
| **09** | Árvores de Decisão, Random Forest & SVM | [code_class.md](./aula09/code_class.md) | [pratica_real.md](./aula09/pratica_real.md) | `load_wine()` |
| **10** | Validação Cruzada Estratificada, ROC & AUC | [code_class.md](./aula10/code_class.md) | [pratica_real.md](./aula10/pratica_real.md) | `load_breast_cancer()` |
| **11** | Otimização em Grade (`GridSearchCV`) & Regularização | [code_class.md](./aula11/code_class.md) | [pratica_real.md](./aula11/pratica_real.md) | `load_wine()` & `load_diabetes()` |
| **12** | Agrupamento Não Supervisionado: K-Means & Cotovelo | [code_class.md](./aula12/code_class.md) | [pratica_real.md](./aula12/pratica_real.md) | `load_wine()` |
| **13** | Agrupamento por Densidade (DBSCAN) & Dendrogramas | [code_class.md](./aula13/code_class.md) | [pratica_real.md](./aula13/pratica_real.md) | `make_moons()` & `load_iris()` |
| **14** | Redução de Dimensionalidade com PCA (64D ──► 2D) | [code_class.md](./aula14/code_class.md) | [pratica_real.md](./aula14/pratica_real.md) | `load_digits()` |
| **15** | O Neurônio Artificial, Soma Ponderada & Ativações | [code_class.md](./aula15/code_class.md) | [pratica_real.md](./aula15/pratica_real.md) | `load_iris()` |
| **16** | Multi-Layer Perceptron (MLP) & Backpropagation | [code_class.md](./aula16/code_class.md) | [pratica_real.md](./aula16/pratica_real.md) | `load_breast_cancer()` |
| **17** | TensorFlow/Keras: Modelo Sequencial no Fashion-MNIST | [code_class.md](./aula17/code_class.md) | [pratica_real.md](./aula17/pratica_real.md) | `tf.keras.datasets.fashion_mnist` |
| **18** | Redes Neurais para Regressão & Classificação Multiclasse | [code_class.md](./aula18/code_class.md) | [pratica_real.md](./aula18/pratica_real.md) | `fetch_california_housing()` |
| **19** | Regularização com Dropout, Callbacks & Persistência | [code_class.md](./aula19/code_class.md) | [pratica_real.md](./aula19/pratica_real.md) | `tf.keras.datasets.fashion_mnist` |
| **20** | Projeto Integrador End-to-End (ML vs Deep Learning) | [code_class.md](./aula20/code_class.md) | [pratica_real.md](./aula20/pratica_real.md) | `sns.load_dataset('titanic')` |

---

### 💻 Como Executar as Práticas:
- **Google Colab (Recomendado):** Acesse [colab.research.google.com](https://colab.research.google.com), crie um novo notebook e copie os blocos de código dos arquivos `.md`. Todos os datasets são carregados diretamente pelo Python sem necessidade de fazer upload manual de arquivos!
- **Ambiente Local (VS Code / Jupyter Notebooks):** Certifique-se de ter as bibliotecas instaladas no seu ambiente virtual:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow scipy
  ```

# 🧠 Aprendizado de Máquina e Redes Neurais

> **Instituição:** UNIVEM — Centro Universitário de Marília  
> **Professor:** Eduardo Lázaro Roesler de Oliveira  
> **Semestre:** Material de Apoio às Aulas Ministradas  

---

## 📌 Sobre a Disciplina

Bem-vindo(a) ao repositório oficial da disciplina **Aprendizado de Máquina e Redes Neurais**!

Este repositório foi construído para acompanhar nossas aulas semanais. Aqui você encontrará os materiais teóricos, códigos explicativos (`code_class.md`), listas de atividades e apresentações que utilizaremos ao longo de todo o semestre.

> [!NOTE]
> 📅 **Liberação Semanal do Conteúdo:**  
> O conteúdo deste repositório é atualizado semanalmente. A cada nova aula ministrada, a respectiva pasta será liberada e disponibilizada para a turma.

---

## 🗺️ Cronograma das Aulas

| Aula | Tema / Conteúdo | Status |
| :---: | :--- | :---: |
| **[Aula 01](./aula01)** | Introdução à Inteligência Artificial e Machine Learning | ✅ **Liberada** |
| **Aula 02** | Python para Ciência de Dados - Parte 1 (NumPy & Estruturas Numéricas) | 🔒 Em breve |
| **Aula 03** | Python para Ciência de Dados - Parte 2 (Pandas DataFrames & Exploração) | 🔒 Em breve |
| **Aula 04** | Preparação e Tratamento de Dados - Parte 1 (Limpeza, Nulos e Outliers) | 🔒 Em breve |
| **Aula 05** | Preparação e Tratamento de Dados - Parte 2 (Feature Engineering, Scaling & Encoding) | 🔒 Em breve |
| **Aula 06** | Visualização de Dados (Matplotlib & Seaborn) | 🔒 Em breve |
| **Aula 07** | Aprendizado Supervisionado - Parte 1 (Regressão Linear Simples & Scikit-Learn Pipeline) | 🔒 Em breve |
| **Aula 08** | Aprendizado Supervisionado - Parte 2 (Classificação: Regressão Logística & KNN) | 🔒 Em breve |
| **Aula 09** | Aprendizado Supervisionado - Parte 3 (Árvores de Decisão, Random Forest & SVM) | 🔒 Em breve |
| **Aula 10** | Avaliação de Modelos - Parte 1 (Validação Cruzada, Curvas ROC & AUC) | 🔒 Em breve |
| **Aula 11** | Avaliação de Modelos - Parte 2 (Otimização de Hiperparâmetros & Regularização) | 🔒 Em breve |
| **Aula 12** | Aprendizado Não Supervisionado - Parte 1 (Agrupamento K-Means & Escolha do K) | 🔒 Em breve |
| **Aula 13** | Aprendizado Não Supervisionado - Parte 2 (Agrupamento Hierárquico & DBSCAN) | 🔒 Em breve |
| **Aula 14** | Aprendizado Não Supervisionado - Parte 3 (Redução de Dimensionalidade & PCA) | 🔒 Em breve |
| **Aula 15** | Fundamentos de Redes Neurais - Parte 1 (O Neurônio Artificial & Funções de Ativação) | 🔒 Em breve |
| **Aula 16** | Fundamentos de Redes Neurais - Parte 2 (Multi-Layer Perceptron - MLP & Backpropagation) | 🔒 Em breve |
| **Aula 17** | Redes Neurais com TensorFlow/Keras - Parte 1 (Arquitetura Sequencial & Keras API) | 🔒 Em breve |
| **Aula 18** | Redes Neurais com TensorFlow/Keras - Parte 2 (Regressão & Classificação Multiclasse) | 🔒 Em breve |
| **Aula 19** | Redes Neurais com TensorFlow/Keras - Parte 3 (Regularização, Callbacks & Persistência) | 🔒 Em breve |
| **Aula 20** | Projeto Integrador: Pipeline Completo de Machine Learning & Deep Learning | 🔒 Em breve |

---

## 📁 Estrutura de Cada Aula

Cada pasta de aula (`aula01/`, `aula02/`, ...) possui a seguinte estrutura organizada:

```text
aulaXX/
├── code_class.md               # Roteiro teórico, explicações detalhadas e código guia
└── aulaXX.pptx                 # Slides apresentados em sala de aula
```

---

## 💻 Como Utilizar os Materiais

### 1️⃣ Para os Alunos (Como Acompanhar o Curso)

#### Clonando o Repositório pela Primeira Vez:
```bash
git clone <URL_DO_REPOSITORIO>
cd machine_learning
```

#### Atualizando o Repositório Toda Semana:
Para baixar o conteúdo das novas aulas liberadas pelo professor:
```bash
git pull origin main
```

#### Executando o Código Prático:
- **Google Colab (Recomendado):** Acesse [colab.research.google.com](https://colab.research.google.com), crie um novo notebook e copie os trechos de código do arquivo `code_class.md` de cada aula.
- **VS Code / Jupyter Notebooks:** Você também pode utilizar o ambiente Python local instalando as bibliotecas indicadas nas aulas (ex: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`).

---

## 🛠️ Guia para o Professor (Liberação Semanal via Git)

Para liberar uma nova aula na semana correspondente:

1. Abra o arquivo `.gitignore`.
2. Comente ou remova a linha da pasta correspondente (exemplo para a Aula 02):
   ```gitignore
   # Altere de:
   /aula02/
   
   # Para:
   # /aula02/
   ```
3. Atualize o status da aula na tabela deste `README.md` para **`✅ Liberada`**.
4. No terminal, envie a nova aula para o repositório remoto:
   ```bash
   git add .
   git commit -m "feat: libera conteudo da Aula 02"
   git push origin main
   ```

---

> [!TIP]
> ✉️ **Dúvidas ou Sugestões?** Utilize os canais oficiais de comunicação da UNIVEM para entrar em contato com o professor durante o semestre. Bons estudos!

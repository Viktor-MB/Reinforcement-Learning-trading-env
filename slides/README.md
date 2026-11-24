# Slides da disciplina (Prof. Rodrigo Luna)

Material de aula de Aprendizado por Reforço (IBMEC). Os arquivos estão em ordem de aula, e a última coluna diz onde o assunto aparece no projeto.

## Aulas (`aulas/`)

| Arquivo | Assunto | Onde entra no projeto |
|---|---|---|
| `Aula_00_Apresentacao_do_Curso.pdf` | Ementa, avaliações, ferramentas e bibliografia | — |
| `Aula_01_Introducao_ao_Aprendizado_por_Reforco.pdf` | Tipos de aprendizado, componentes do RL, MDP em 60 segundos, Bellman | Visão geral (slides 1–2) |
| `Aula_02_Revisao_de_Probabilidade_e_Estatistica.pdf` | Espaço amostral, probabilidade condicional, Bayes, esperança, variância, Lei dos Grandes Números | Base para $\widehat P$, $\widehat R$ e Monte Carlo |
| `Aula_03_Probabilistic_Machine_Learning.pdf` | Normal, máxima verossimilhança (MLE), normalização, regressão linear | Estimação de $\widehat P$; *benchmark* econométrico (slide 18) |
| `Aula_04a_Cadeias_de_Markov.pdf` | Cadeias de Markov: definição, matriz de transição, irredutibilidade, periodicidade, distribuição estacionária | Propriedade de Markov e discretização (slides 3–5) |
| `Aula_04b_Cadeias_de_Markov_MRP_e_MDP.pdf` | Revisão de cadeias, MRP, retorno e desconto, função de valor, equação de Bellman, MDP, *Value* e *Policy Iteration* | Bellman, $\gamma$ e *Value Iteration* (slides 2, 6 e 10) |
| `Aula_05_Monte_Carlo_Model_Free_e_Gymnasium.pdf` | Lei dos Grandes Números, Monte Carlo, *model-free*, *first-visit* × *every-visit*, Gymnasium (FrozenLake, Taxi, Blackjack) | Monte Carlo (slides 8 e 13) |
| `Aula_06_Monte_Carlo_Prediction_e_Controle.pdf` | Monte Carlo *prediction* e controle, cálculo de $V(s)$ e $Q(s,a)$, $\varepsilon$-greedy | Monte Carlo e $\varepsilon$-greedy (slides 8 e 17) |
| `Aula_07_TD_Learning.pdf` | TD(0), TD × Monte Carlo, SARSA, Q-learning | Q-learning (slides 9 e 12) |

As aulas 05 e 06 têm o mesmo título, mas são versões diferentes, cada uma com exemplos próprios. A aula 04 veio em duas partes: a 04a (só cadeias de Markov) e a 04b (revisão, MRP e MDP).

## Listas (`listas/`)

| Arquivo | Assunto |
|---|---|
| `Lista_0_Revisao_de_Probabilidade.pdf` | Exercícios de revisão de probabilidade |

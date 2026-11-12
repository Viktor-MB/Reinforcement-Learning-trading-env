# Aprendizado por Reforço aplicado a Trading: MDP + Value Iteration

**Autores:** Tiago Macedo · Viktor Mayer · Enzo Guedes, IBMEC

Este projeto modela a decisão de *trading* de um ativo (Bitcoin, `BTC-USD`, com o ouro `GLD` como reserva caso o download falhe) como um **Processo de Decisão de Markov (MDP)** finito, **model-based**. A política ótima é obtida por **Value Iteration** e avaliada no simulador `gym-trading-env`, fora da amostra, contra *benchmarks* e modelos supervisionados de *machine learning*.

---

## Ideia central

Em vez de prever o preço, o agente aprende **qual posição assumir** a cada dia, levando em conta o custo de operar. O simulador de mercado não fornece a matriz de transições; por isso a dinâmica ($\widehat{P}$) e a recompensa ($\widehat{R}$) são **estimadas empiricamente** na base de treino, o que permite aplicar o Value Iteration de forma exata (de *model-free* para *model-based*).

---

## Modelagem do MDP

### 1. Estado e ações

O agente observa o regime de tendência do ativo e a sua posição corrente:

$$s_t = (x_t,\ x_{t-1},\ h_{t-1})$$

- **Sinal de tendência:** $x_t = \ln\left(\dfrac{c_t}{\text{MA}_{50}(c_t)}\right)$, o desvio do preço de fechamento em relação à média móvel de 50 dias ($x_t > 0$: regime altista; $x_t < 0$: baixista).
- **Posição mantida (*holding*):** $h_{t-1} \in \{-1, 0, +1\}$, atualizada por $h_t = a_t$.
- **Ações:** $a_t \in \{-1, 0, +1\}$, em que $-1$ é vendido (*short*), $0$ é neutro (fora do mercado) e $+1$ é comprado (*long*).

### 2. Discretização

O sinal é discretizado em 3 faixas pelos **tercis do conjunto de treino** (limiares $\tau_1, \tau_2$ calculados só com dados de treino, sem vazamento de informação):

$$\phi(x) = \begin{cases} 0, & x \le \tau_1 \ \text{(tendência de baixa)} \\ 1, & \tau_1 < x \le \tau_2 \ \text{(regime neutro)} \\ 2, & x > \tau_2 \ \text{(tendência de alta)} \end{cases}$$

O estado discreto é $\tilde{s}_t = (\phi(x_t), \phi(x_{t-1}), h_{t-1})$, com $|\mathcal{S}| = 3^2 \times 3 = $ **27 estados**.

> Usar $x_t$ (médio prazo) em vez de ruído diário reduz rotações excessivas da carteira e preserva o capital frente aos custos de transação.

### 3. Recompensa e transição

A recompensa imediata é o log-retorno líquido do patrimônio $W_t$:

$$R_{t+1} = \ln\left(\frac{W_{t+1}}{W_t}\right) = \ln(g_{t+1}), \qquad g_{t+1} = 1 + a_t\, r_{t+1} - \kappa\,\lvert a_t - h_{t-1}\rvert$$

em que $r_{t+1} = c_{t+1}/c_t - 1$ é o retorno linear do ativo e $\kappa = 0{,}1\%$ é a corretagem cobrada a cada mudança de posição. A transição pretendida é $\tilde{s}_{t+1} = (\phi(x_{t+1}), \phi(x_t), a_t)$: a parte da posição é determinística e a parte de mercado é estimada dos dados históricos.

> **Observação:** no simulador, a exposição vendida mantida não é rebalanceada diariamente para $-100\%$. Por isso a recompensa do ambiente pode diferir levemente da fórmula simplificada nos períodos de *short* mantido (documentado nos testes do notebook).

### 4. Equações de Bellman

$$V^\pi(s) = \sum_{a} \pi(a \mid s)\, Q^\pi(s,a), \qquad Q^\pi(s,a) = \sum_{s'} \widehat{P}(s' \mid s,a)\left[\widehat{R}(s,a,s') + \gamma V^\pi(s')\right]$$

### 5. Inferência empírica

$$\widehat{P}(s' \mid s,a) = \frac{N(s,a,s')}{N(s,a)}, \qquad \widehat{R}(s,a,s') = \frac{1}{N(s,a,s')}\sum_{k=1}^{N(s,a,s')} \ln\left(1 + a\, r^{(k)}_{t+1} - \kappa\,\lvert a - h_{t-1}\rvert\right)$$

A recompensa usa o retorno linear bruto $r_{t+1}$; $x_t$ apenas indexa o estado.

### 6. Value Iteration e política ótima

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \sum_{s'} \widehat{P}(s' \mid s,a)\left[\widehat{R}(s,a,s') + \gamma V_k(s')\right], \qquad \pi^*(s) = \arg\max_a Q^*(s,a)$$

O algoritmo avalia todas as ações para os 27 estados até a convergência (tolerância $< 10^{-12}$), e a política resultante tem a seguinte leitura:

- **Baixa estável:** sai do mercado.
- **Entrando na baixa:** vende (*short*).
- **Neutro ou alta:** fica comprado.
- **Saindo da baixa para neutro:** espera, sem comprar.

---

## Resultados (fora da amostra)

Treino até 2021-12-31; teste a partir de 2022-01-01. Parâmetros de referência: $\kappa = 0{,}1\%$, $\gamma$ conforme o notebook.

### Execução no `gym-trading-env` (teste)

| Estratégia | Retorno total | Retorno anual | Sharpe | Máx. drawdown | Nº de trocas |
|---|---|---|---|---|---|
| **MDP (Value Iteration)** | **137,44%** | **19,52%** | **0,51** | **-47,93%** | 135 |
| MDP ε-greedy (ε = 0,1) | 110,28% | 16,56% | 0,43 | -49,48% | 327 |
| Buy & Hold | 58,81% | 10,01% | 0,19 | -66,93% | 1 |
| Fora do mercado | 0,00% | 0,00% | 0,00 | 0,00% | 0 |
| Aleatória (ε = 1) | -93,70% | -43,46% | -1,35 | -94,54% | 1.155 |

*ε-greedy: com probabilidade ε o agente escolhe uma ação aleatória, ou seja, $\pi(a^* \mid s) = 1 - \epsilon + \epsilon/3$.*

### *Benchmark* completo contra modelos supervisionados (ordenado por Sharpe)

| Estratégia / modelo | Retorno total | Retorno anual | Sharpe | Máx. drawdown | Trocas |
|---|---|---|---|---|---|
| **MDP (Value Iteration)** | +136,28% | +19,64% | **0,51** | -70,22% | 135 |
| Support Vector Machine (SVM) | +209,50% | +26,57% | 0,42 | -76,76% | 93 |
| Classificador Ridge (L2) | +198,89% | +25,65% | 0,41 | -75,83% | 69 |
| Buy & Hold (referência) | +108,30% | +16,54% | 0,26 | -67,35% | 1 |
| Regressão Lasso (L1) | +108,30% | +16,54% | 0,26 | -67,35% | 1 |
| Regressão Linear (OLS) / MA50 | +57,99% | +10,01% | 0,23 | -62,13% | 105 |
| Random Forest | +49,78% | +8,79% | 0,14 | -51,53% | 600 |
| Árvore de Decisão (CART) | -56,93% | -16,11% | -0,39 | -70,34% | 611 |
| Gradient Boosting | -89,20% | -37,14% | -0,95 | -92,38% | 590 |
| K-Nearest Neighbors (KNN) | -90,77% | -39,16% | -1,02 | -93,43% | 718 |

### Principais conclusões

- **Melhor eficiência ajustada ao risco:** o MDP atingiu o maior Sharpe (0,51). SVM e Ridge tiveram maior retorno nominal, mas com volatilidade e *drawdowns* excessivos (cerca de -76%).
- **Impacto do atrito operacional:** modelos não lineares complexos (KNN, Boosting) colapsaram por *overtrading* (mais de 590 trocas). O MDP conteve o atrito ao integrar a posição anterior $h_{t-1}$ na função de valor de Bellman.
- **Interpretabilidade:** a política inteira cabe em uma tabela de 27 estados.

> Os números podem variar levemente entre execuções, pois dependem da data de download dos dados (`yfinance`) e dos parâmetros usados ($\gamma$, $\kappa$).

---

## Estrutura do notebook (`RL_TRADE.ipynb`)

1. Instalação das dependências
2. Imports e parâmetros
3. Dados e discretização (preço vs. MA50)
4. Estimação de $\widehat{P}$ e $\widehat{R}$
5. Value Iteration
6. Curva de aprendizado do Value Iteration
7. Testes de consistência do modelo
8. Execução das políticas no `gym-trading-env`
9. Testes de consistência do ambiente
10. Gráficos do período de teste
11. *Benchmark*: MDP vs. modelos supervisionados
12. Risco × retorno e Sharpe
13. Curva de capital no teste

---

## Como executar

```bash
pip install gym-trading-env yfinance matplotlib scikit-learn
jupyter notebook RL_TRADE.ipynb
```

Execute as células **em ordem**. Se houver incompatibilidade de versões:

```bash
pip install "gymnasium==0.29.1" "numpy<2" gym-trading-env yfinance
```

Depois reinicie a sessão.

**Dependências:** `numpy`, `pandas`, `matplotlib`, `yfinance`, `scikit-learn`, `gym-trading-env`.

### Parâmetros principais

```python
TICKER = "BTC-USD"      # ativo (reserva: "GLD")
INICIO = "2010-01-01"
CORTE  = "2022-01-01"   # treino antes, teste depois
KAPPA  = 0.001          # corretagem
GAMMA  = 0.8            # fator de desconto (ajuste conforme sua versão final)
```

---

## Arquivos gerados

- `curva_value_iteration.png`: convergência do *Value Iteration*
- `benchmark_modelos.png`: risco × retorno e Sharpe dos modelos
- `curva_patrimonio.png`: evolução do patrimônio no teste

---

## Limitações

- **Estado simples:** apenas 3 faixas do sinal preço/MA50 e as duas últimas observações; não captura volatilidade, volume ou regimes macro.
- **Um único ativo e uma única divisão treino/teste**, sem validação *walk-forward*.
- **Custos simplificados:** apenas corretagem fixa; sem *slippage*, impacto de mercado ou juros de aluguel nas posições vendidas.
- **Divergência de *short* mantido** entre o MDP e o simulador, descrita acima.
- **Não é recomendação de investimento.** Resultados passados não garantem desempenho futuro.

---

## Referências

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- [`gym-trading-env`](https://github.com/ClementPerroud/Gym-Trading-Env)
- [`yfinance`](https://github.com/ranaroussi/yfinance)

---

## Licença

Projeto acadêmico desenvolvido no IBMEC. Defina aqui a licença desejada (por exemplo, MIT).

---

**Atenção a uma inconsistência entre as fontes:** o notebook mostra $\gamma = 0{,}8$, convergência em 98 iterações e Sharpe de 0,46 / drawdown de -29,35% no teste, enquanto a apresentação traz $\gamma = 0{,}95$, 419 iterações e drawdown de -47,93% (ou -70,22% no *benchmark*). Isso indica execuções diferentes. Usei no README os números da apresentação, mas confira qual versão é a final e alinhe `GAMMA` e as tabelas antes de publicar. Se quiser, adapto o texto para a versão do notebook.
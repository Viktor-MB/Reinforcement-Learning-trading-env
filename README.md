# Aprendizado por Reforço aplicado a Trading: Value Iteration, Monte Carlo e Q-learning

**Autores:** Tiago Macedo · Viktor Mayer · Enzo Guedes, IBMEC
**Professor:** Rodrigo Luna

O projeto modela a decisão diária de *trading* em Bitcoin (`BTC-USD`) como um **Processo de Decisão de Markov (MDP)** finito, com 27 estados. O ouro (`GLD`) entra como reserva caso o download falhe. O problema é resolvido de três formas:

| | Value Iteration | Monte Carlo | Q-learning |
|---|---|---|---|
| Tipo | *model-based* | *model-free* | *model-free* (diferença temporal) |
| Como aprende | estima $\widehat P$ e $\widehat R$ no treino e calcula a política ótima | vive episódios no histórico e tira a média dos retornos | vive os mesmos episódios, mas atualiza a cada dia com a estimativa do dia seguinte |
| Precisa do modelo? | sim | não | não |

As políticas são avaliadas fora da amostra no simulador `gym-trading-env`. A comparação inclui *benchmarks* simples, modelos econométricos e uma validação *walk-forward*.

**Regra do trabalho:** tudo é aprendido só com o **treino** (até 31/12/2021). O **teste** (a partir de 01/01/2022) é usado uma única vez, no fim.

---

## Modelagem do MDP

### 1. Estado e ações

$$s_t = (x_t,\ x_{t-1},\ h_{t-1}), \qquad x_t = \ln\left(\frac{c_t}{\text{MA}_{50}(c_t)}\right)$$

- $x_t$: desvio do preço de fechamento em relação à média móvel de 50 dias.
- $h_{t-1} \in \{-1, 0, +1\}$: posição anterior.
- Ações $a_t \in \{-1, 0, +1\}$: vendido (*short*), fora do mercado, comprado (*long*).

**Frequência diária:** uma decisão por dia. O bitcoin negocia 24h, 7 dias por semana. O fechamento $c_t$ é o corte diário do Yahoo Finance (0h UTC, 21h de Brasília), por isso 1 ano = 365 dias nas métricas.

### 2. Discretização pelos tercis do treino

$$\phi(x) = \begin{cases} 0 \ \text{(baixa)}, & x \le \tau_1 \\ 1 \ \text{(neutro)}, & \tau_1 < x \le \tau_2 \\ 2 \ \text{(alta)}, & x > \tau_2 \end{cases}$$

- Calcula-se $x_t$ em cada dia do treino (2.614 dias), ordena-se e corta-se em 3 partes iguais (percentis 33% e 67%).
- Resultado: $\tau_1 = -0{,}0234$ (preço 2,3% abaixo da média) e $\tau_2 = 0{,}0953$ (10% acima).
- Os limiares não são simétricos porque no treino o BTC passou mais tempo acima da média.
- O teste usa os mesmos limiares, sem recalcular.

Estado discreto: $\tilde{s}_t = (\phi(x_t), \phi(x_{t-1}), h_{t-1})$, com $|\mathcal{S}| = 3 \times 3 \times 3 =$ **27 estados**.

### 3. Recompensa e transição

$$R_{t+1} = \ln\left(\frac{W_{t+1}}{W_t}\right) = \ln\left(1 + a_t\, r_{t+1} - \kappa\,\lvert a_t - h_{t-1}\rvert\right)$$

- $r_{t+1} = c_{t+1}/c_t - 1$ é o retorno do ativo.
- $\kappa = 0{,}1\%$ é a corretagem, cobrada quando a posição muda.
- Transição: $\tilde{s}_{t+1} = (\phi(x_{t+1}), \phi(x_t), a_t)$. A parte da posição é determinística; a parte de mercado vem dos dados.

> No simulador, o *short* mantido não é rebalanceado para −100% todo dia. Por isso a recompensa do ambiente difere levemente da fórmula nesses passos. Isso está documentado nos testes do notebook (seção 14).

### 4. Inferência empírica e Value Iteration

$$\widehat{P}(s' \mid s,a) = \frac{N(s,a,s')}{N(s,a)}, \qquad \widehat{R}(s,a,s') = \frac{1}{N(s,a,s')}\sum_{k} \ln\left(1 + a\, r^{(k)}_{t+1} - \kappa\,\lvert a - h_{t-1}\rvert\right)$$

$$V_{k+1}(s) = \max_{a} \sum_{s'} \widehat{P}(s' \mid s,a)\left[\widehat{R}(s,a,s') + \gamma V_k(s')\right], \qquad \pi^*(s) = \arg\max_a Q^*(s,a)$$

Com $\gamma = 0{,}8$, converge em **98 iterações** (tolerância $< 10^{-12}$). Leitura da política ótima:

- **Baixa estável:** não abre posição.
- **Entrando na baixa:** vende.
- **Neutro:** só mantém o *long* já aberto.
- **Alta:** compra.

### 5. Monte Carlo (*model-free*)

$$G_t = R_{t+1} + \gamma\, G_{t+1}, \qquad Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\big[G_t - Q(s,a)\big]$$

- 30.000 episódios de 84 dias, sorteados no treino, com *exploring starts*.
- Exploração $\varepsilon$-greedy, com $\varepsilon$ caindo de 1 para 0,05.
- *First-visit* e *every-visit* foram comparados. A versão principal (**every-visit**) foi escolhida pelo Sharpe no treino.

### 6. Q-learning

$$Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\Big[R_{t+1} + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]$$

- Usa os mesmos episódios e a mesma exploração do Monte Carlo.
- Atualiza a cada passo (*bootstrap*), é *off-policy* e supõe que o estado é Markov.

| | Value Iteration | Monte Carlo | Q-learning |
|---|---|---|---|
| Atualizações / iterações | 98 iterações | 1.260.000 | 2.520.000 |
| Concordância com o VI | 27/27 | 22/27 | 21/27 |
| Erro médio $\lvert Q - Q^*\rvert$ | 0 | 0,0073 | 0,0028 |
| Sharpe no treino | 1,09 | 1,20 | 1,37 |

---

## Métricas

| Métrica | Fórmula | O que mede |
|---|---|---|
| Sharpe | $\sqrt{365}\ \bar e / \sigma_e$ | retorno acima da taxa livre de risco por unidade de volatilidade |
| Max drawdown | $\min_t (W_t / P_t - 1)$ | maior queda do patrimônio a partir do pico $P_t$ |
| Calmar | $R_{\text{anual}} / \lvert\text{MDD}\rvert$ | retorno anual por unidade de queda máxima |

- $e_t$ é o retorno diário menos a taxa livre de risco: o T-bill americano de 3 meses (`^IRX`), convertido para taxa diária.
- O T-bill foi usado porque o ativo é cotado em dólar.
- Taxa média: 0,78% a.a. no treino e 3,94% a.a. no teste.
- No simulador o caixa não rende juros. Por isso o Sharpe de estratégias que ficam muito tempo fora é conservador.

---

## Resultados

Treino: 2014-11 a 2021-12. Teste: 2022-01 a 2026-11. $\kappa = 0{,}1\%$, $\gamma = 0{,}8$.

### Execução no `gym-trading-env` (teste)

| Estratégia | Retorno total | Retorno anual | Sharpe | Max drawdown | Calmar | Trocas |
|---|---|---|---|---|---|---|
| **MDP (Value Iteration)** | **127,80%** | **18,41%** | **0,46** | **−29,35%** | 0,63 | 113 |
| Monte Carlo (every-visit) | 119,47% | 17,51% | 0,44 | −26,95% | 0,65 | 108 |
| Q-learning | 86,39% | 13,64% | 0,26 | −51,25% | 0,27 | 190 |
| Buy & Hold | 37,05% | 6,68% | 0,05 | −66,93% | 0,10 | 1 |
| Fora do mercado | 0,00% | 0,00% | 0,00 | 0,00% | – | 0 |
| Aleatória | −93,83% | −43,56% | −1,45 | −94,54% | −0,46 | 1.162 |

O teste começa perto de um topo (BTC a ~US$ 47 mil em jan/2022, ~US$ 16 mil no fim de 2022). Por isso o Buy & Hold rende pouco, e o foco está no retorno ajustado ao risco.

### Política gulosa × ε-greedy (ε = 0,1, média de 100 sementes)

| Política | Sharpe | Max drawdown | Trocas | ε vence a gulosa em |
|---|---|---|---|---|
| MDP: gulosa / ε-greedy | 0,46 / 0,20 | −29,35% / −49,77% | 113 / 286 | 24% das sementes |
| Monte Carlo: gulosa / ε-greedy | 0,44 / 0,14 | −26,95% / −50,57% | 108 / 289 | 21% das sementes |
| Q-learning: gulosa / ε-greedy | 0,26 / 0,04 | −51,25% / −59,21% | 190 / 373 | 23% das sementes |

Explorar serve para aprender, não para operar.

### Monte Carlo e Q-learning com 5 sementes (42 a 46)

| Método | Sharpe treino (média ± dp) | Sharpe teste (média ± dp) | Max drawdown teste (média) |
|---|---|---|---|
| MC first-visit | 1,42 ± 0,17 | 0,30 ± 0,26 | −39,07% |
| MC every-visit | 1,17 ± 0,04 | 0,34 ± 0,09 | −33,24% |
| Q-learning | 1,31 ± 0,10 | 0,24 ± 0,12 | −49,58% |

- O every-visit é o mais estável.
- Na média do treino o first-visit vence. Com uma semente só, a escolha da versão dependeu da sorte.

### *Benchmark*: RL × modelos econométricos (teste)

| Modelo | Retorno total | Sharpe | Max drawdown | Calmar | Trocas |
|---|---|---|---|---|---|
| **MDP (Value Iteration)** | +126,86% | **0,46** | −29,35% | 0,62 | 113 |
| Monte Carlo | +119,46% | 0,44 | −26,95% | 0,65 | 108 |
| Q-learning | +86,08% | 0,26 | −51,28% | 0,27 | 190 |
| AR(1) | +118,97% | 0,24 | −73,07% | 0,24 | 49 |
| ARMA(1,1) | +107,92% | 0,22 | −72,85% | 0,22 | 43 |
| Probit | +60,49% | 0,12 | −72,95% | 0,14 | 113 |
| Logit | +56,35% | 0,11 | −72,95% | 0,13 | 115 |
| Buy & Hold (passeio aleatório) | +37,05% | 0,05 | −66,93% | 0,10 | 1 |
| Markov-switching (Hamilton) | +37,05% | 0,05 | −66,93% | 0,10 | 1 |
| Regressão preditiva (OLS) | −50,08% | −0,36 | −77,59% | −0,17 | 224 |

O *benchmark* usa um simulador próprio, com as mesmas fórmulas. Por isso há diferenças pequenas em relação à tabela do `gym-trading-env`.

### *Walk-forward*: reestimação periódica

- **Configurações:** retreinar nunca, ou a cada 3, 6 ou 12 meses; janela expansiva (todo o histórico) ou móvel (últimos 4 anos). São 7 configurações por modelo.
- **Escolha:** cada modelo fica com a configuração de maior Sharpe na validação (2019–2021, dentro do treino), sem olhar o teste.
- A configuração "sem reestimação" reproduz exatamente o *benchmark*, o que valida o código.

| Modelo | Nunca retreina | 12 meses | 6 meses | 3 meses | Escolha na validação |
|---|---|---|---|---|---|
| **MDP (VI)** | **0,46** | 0,28 | 0,35 | 0,07 | nunca |
| Monte Carlo | 0,44 | 0,14 | 0,18 | 0,26 | 3 meses |
| Q-learning | 0,26 | −0,03 | 0,25 | −0,15 | 3 meses |
| AR(1) | 0,24 | 0,19 | 0,07 | 0,08 | 12 meses |

*Sharpe no teste, janela expansiva.* Retreinar não melhorou nenhum modelo. Cada retreino recalcula os tercis e as probabilidades, e a política troca de ação.

### Principais conclusões

- O **MDP** tem o maior Sharpe (0,46) e menos da metade da queda máxima dos econométricos.
- O **Monte Carlo** chega perto do MDP sem precisar do modelo.
- O **Q-learning** chega mais perto de $Q^*$, mas no teste troca mais de posição e cai mais.
- Os três métodos de RL lideram em Sharpe e Calmar.
- A política inteira cabe em uma tabela de 27 estados.

> Os números podem variar levemente entre execuções, porque os dados do `yfinance` são atualizados todo dia (o último candle muda).

---

## Estrutura do notebook (`RL_TRADE.ipynb`)

- **Parte I. Preparação:** 1. Instalação · 2. Imports e parâmetros · 3. Dados e estados (com o histograma dos tercis)
- **Parte II. Value Iteration:** 4. Estimando o modelo · 5. Value Iteration · 6. Curva de aprendizado · 7. Testes do modelo
- **Parte III. Monte Carlo:** 8. Treino · 9. Escolha da versão principal · 10. Curvas de aprendizado
- **Parte IV. Q-learning:** 11. Treino · 12. Q-learning × Monte Carlo × Value Iteration
- **Parte V. Avaliação:** 13. Simulador (gulosa × ε-greedy, 5 sementes) · 14. Testes do ambiente · 15. Gráficos do teste · 16. *Benchmark* econométrico · 17. Risco × retorno · 18. Curva de capital · 19. *Walk-forward* · 20. Resumo e conclusão

---

## Como executar

```bash
pip install gym-trading-env yfinance matplotlib statsmodels
jupyter notebook RL_TRADE.ipynb
```

- Execute as células **em ordem**. O notebook inteiro leva cerca de 10 minutos.
- O *walk-forward* é a parte mais lenta (~8 min) e roda em paralelo, com `multiprocessing`.

Se houver incompatibilidade de versões:

```bash
pip install "gymnasium==0.29.1" "numpy<2" gym-trading-env yfinance
```

**Dependências:** `numpy`, `pandas`, `matplotlib`, `yfinance`, `statsmodels`, `gym-trading-env`.

### Parâmetros principais

```python
TICKER = "BTC-USD"      # ativo (reserva: "GLD")
INICIO = "2010-01-01"
CORTE  = "2022-01-01"   # treino antes, teste depois
KAPPA  = 0.001          # corretagem
GAMMA  = 0.8            # fator de desconto
SEED   = 42             # sementes 42 a 46 na análise de variância
```

---

## Arquivos

- `RL_TRADE.ipynb`: notebook principal (todo o código e os resultados)
- `apresentacao.tex`: slides em Beamer
- `imagens/`: figuras geradas pelo notebook
  - `limiares_tercis.png`: histograma de $x_t$ no treino com os tercis
  - `curva_value_iteration.png`: convergência do Value Iteration
  - `curva_monte_carlo.png`: aprendizado do Monte Carlo
  - `curva_q_learning.png`: aprendizado do Q-learning
  - `curva_sementes.png`: MC e Q-learning, média ± 1 desvio padrão em 5 sementes
  - `grafico_teste.png`: preço, patrimônio e posição no teste
  - `benchmark_modelos.png`: risco × retorno e Sharpe dos modelos
  - `curva_patrimonio.png`: evolução do patrimônio no teste

---

## Limitações

- **Estado simples:** 3 faixas do sinal preço/MA50 e as duas últimas observações. Não captura volatilidade, volume ou regimes macro.
- **Um único ativo** e um único corte treino/teste. O *walk-forward* mostrou que retreinar recalculando os tercis deixa a política instável.
- **Custos simplificados:** só corretagem fixa. Não há *slippage*, impacto de mercado ou custo de aluguel no *short*.
- **Caixa sem juros** no simulador, o que torna o Sharpe conservador.
- **Divergência no *short* mantido** entre o MDP e o simulador, descrita acima.
- **Não é recomendação de investimento.** Resultados passados não garantem desempenho futuro.

---

## Referências

- Sutton & Barto, *Reinforcement Learning: An Introduction* (Monte Carlo: cap. 5; Q-learning e viés de maximização: cap. 6)
- [`gym-trading-env`](https://github.com/ClementPerroud/Gym-Trading-Env)
- [`yfinance`](https://github.com/ranaroussi/yfinance)
- [`statsmodels`](https://www.statsmodels.org/)

---

## Licença

Projeto acadêmico desenvolvido no IBMEC, para fins educacionais.

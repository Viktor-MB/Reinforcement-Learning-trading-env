# Aprendizado por Reforço aplicado a Trading: MDP + Value Iteration

**Autores:** Tiago Macedo · Viktor Mayer · Enzo Guedes, IBMEC

Este projeto modela a decisão de *trading* de um ativo (Bitcoin, `BTC-USD`, com o ouro `GLD` como reserva caso o download falhe) como um **Processo de Decisão de Markov (MDP)** finito. A política ótima é obtida por **Value Iteration** e comparada, fora da amostra, com *benchmarks* e com modelos supervisionados de *machine learning*.

---

## Ideia central

Em vez de prever o preço, o agente aprende **qual posição assumir** a cada dia, levando em conta o custo de operar.

### Modelagem do MDP

| Elemento | Definição |
|---|---|
| **Sinal** | $x_t = \ln\left(P_t / \text{MA}_{50}(P)_t\right)$, o preço relativo à média móvel de 50 dias |
| **Discretização** | $\phi(x_t) \in \{0, 1, 2\}$ (baixa / neutro / alta), com limiares nos tercis **calculados só com dados de treino** (sem vazamento de informação) |
| **Estado** | $s_t = (\phi(x_t), \phi(x_{t-1}), h_{t-1})$, ou seja, **27 estados** (3 × 3 × 3) |
| **Ações** | $a_t \in \{-1, 0, +1\}$: vendido (*short*), fora do mercado, comprado (*long*) |
| **Recompensa** | $R = \ln\left(1 + a \cdot r_{t+1} - \kappa \cdot \lvert a - h\rvert\right)$, o log-retorno líquido de corretagem ($\kappa = 0{,}1\%$) |
| **Fator de desconto** | $\gamma = 0{,}8$ |
| **Transição** | $\widehat{P}$ estimada por contagem de frequências no treino; a parte da posição ($h_t = a_t$) é determinística |

### Algoritmo

1. Baixa os dados (`yfinance`) e a taxa livre de risco (`^IRX`, T-bill de 3 meses).
2. Separa **treino** (até 2021-12-31) e **teste** (a partir de 2022-01-01).
3. Estima $\widehat{P}$ e $\widehat{R}$ no conjunto de treino.
4. Resolve o MDP com **Value Iteration** (convergência em ~98 iterações, 99% do valor ótimo em ~15).
5. Executa a política $\pi^*$ no `gym-trading-env` e mede o desempenho no teste.

---

## Estrutura do notebook (`RL_TRADE.ipynb`)

1. Instalação das dependências
2. Imports e parâmetros
3. Dados e discretização (preço vs. MA50)
4. Estimação de $\widehat{P}$ e $\widehat{R}$
5. Value Iteration
6. Curva de aprendizado do Value Iteration
7. Testes de consistência do modelo (9 asserts)
8. Execução das políticas no `gym-trading-env`
9. Testes de consistência do ambiente
10. Gráficos do período de teste
11. *Benchmark*: MDP vs. modelos supervisionados
12. Risco × retorno e Sharpe
13. Curva de capital no teste

---

## Resultados (período de teste, 2022-01 → 2026-11)

### Execução no `gym-trading-env`

| Política | Retorno anual | Sharpe | Máx. drawdown | Nº de trocas |
|---|---|---|---|---|
| **MDP (Value Iteration)** | **18,49%** | **0,46** | **-29,35%** | 113 |
| Buy & Hold | 9,96% | 0,11 | -66,93% | 1 |
| Fora do mercado | 0,00% | 0,00 | 0,00% | 0 |
| Aleatória | -43,46% | -1,45 | -94,54% | 1156 |

### *Benchmark* contra modelos supervisionados (ranking por Sharpe)

| Modelo | Retorno anual | Sharpe | Máx. drawdown |
|---|---|---|---|
| **MDP (Value Iteration)** | +18,64% | **0,46** | **-29,35%** |
| SVM linear | +23,60% | 0,34 | -62,59% |
| Classificador Ridge (L2) | +22,71% | 0,33 | -62,28% |
| Buy & Hold | +13,81% | 0,18 | -66,74% |
| Heurística Preço/MA50 | +12,33% | 0,15 | -62,13% |
| Regressão Logística (filtro) | +10,81% | 0,14 | -62,22% |
| Random Forest | +6,22% | 0,05 | -53,58% |
| Árvore de Decisão | -18,10% | -0,47 | -70,39% |
| Gradient Boosting | -31,15% | -0,81 | -88,69% |
| KNN | -40,53% | -1,10 | -93,43% |

> **Leitura dos resultados:** o MDP não tem o maior retorno absoluto (SVM e Ridge o superam), mas entrega o **melhor Sharpe e o menor drawdown**, isto é, o melhor retorno por unidade de risco, operando com poucas trocas e um modelo totalmente interpretável (uma tabela de 27 estados).

Os valores podem variar levemente entre execuções, pois dependem da data de download dos dados.

---

## Como executar

```bash
pip install gym-trading-env yfinance matplotlib scikit-learn
jupyter notebook RL_TRADE.ipynb
```

Execute as células **em ordem**. Se o ambiente travar por incompatibilidade de versões:

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
GAMMA  = 0.8            # fator de desconto
```

---

## Arquivos gerados

- `curva_value_iteration.png`: convergência do *Value Iteration*
- `benchmark_modelos.png`: risco × retorno e Sharpe dos modelos
- `curva_patrimonio.png`: evolução do patrimônio no teste

---

## Limitações

- **Estado muito simples:** só 3 faixas do sinal preço/MA50 e as duas últimas observações; não captura volatilidade, volume ou regimes macro.
- **Um único ativo e uma única divisão treino/teste**, sem validação cruzada temporal (*walk-forward*).
- **Custos simplificados:** apenas corretagem fixa; sem *slippage*, impacto de mercado ou juros de aluguel nas posições vendidas.
- No ambiente, o *short* mantido não rebalanceia para -100%, o que gera pequenas diferenças em relação à fórmula de recompensa do MDP (documentado no teste da seção 9).
- **Não é recomendação de investimento.** Resultados passados não garantem desempenho futuro.

---

## Referências

- Sutton & Barton, *Reinforcement Learning: An Introduction*
- [`gym-trading-env`](https://github.com/ClementPerroud/Gym-Trading-Env)
- [`yfinance`](https://github.com/ranaroussi/yfinance)

---

## Licença

Projeto acadêmico desenvolvido no IBMEC. Defina aqui a licença desejada (por exemplo, MIT).

---

Se quiser, posso ajustar o tom (mais acadêmico ou mais enxuto), traduzir para inglês, ou gerar o arquivo `README.md` pronto para download.
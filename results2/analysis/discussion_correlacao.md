# Discussão: desempenho × características do dataset

Correlação entre o desempenho do OptIForest (AUC-ROC e AUC-PR médios sobre 15 runs) e três
características de cada dataset, nos 22 conjuntos avaliados. Valores em
`correlations.csv`; dispersões em `plots/correlations.png`.

| Característica | Métrica | Pearson r | Spearman ρ | Leitura |
|---|---|---|---|---|
| Nº de amostras | AUC-ROC | 0.03 | −0.03 | **Sem relação** |
| Nº de amostras | AUC-PR | −0.54 | −0.50 | **Negativa moderada** |
| Nº de features | AUC-ROC | −0.35 | −0.38 | Negativa fraca/moderada |
| Nº de features | AUC-PR | −0.07 | −0.05 | Sem relação |
| Taxa de anomalia | AUC-ROC | −0.24 | −0.35 | Negativa fraca |
| Taxa de anomalia | AUC-PR | 0.50 | 0.40 | **Positiva moderada** |

## Interpretação

**1. Dimensionalidade penaliza o ranking (AUC-ROC ↓ com nº de features, r ≈ −0.36).**
Quanto mais features, menor o AUC-ROC. É coerente com a maldição da dimensionalidade em
isolation forests baseadas em distância/hash: em alta dimensão as distâncias se concentram
e o particionamento por LSH perde poder discriminativo. Casos extremos: `mnist` (100 feat,
ROC 0.60) e `arrhythmia` (274 feat, ROC 0.77) ficam entre os piores em ROC.

**2. Taxa de anomalia governa o AUC-PR (r ≈ +0.50), não o AUC-ROC.**
Datasets com mais anomalias têm AUC-PR maior; com anomalia rara, o AUC-PR despenca
(`cover` 0.96% → AUC-PR 0.015; `smtp` 0.03%). **Atenção metodológica:** parte dessa
correlação é *mecânica* — a linha de base aleatória do AUC-PR é a própria prevalência de
anomalias, então AUC-PR alto em dataset balanceado não significa, sozinho, melhor método.
Por isso o AUC-ROC (insensível à prevalência) é o critério mais justo para comparar
generalização entre datasets.

**3. Tamanho não afeta o ranking (AUC-ROC vs nº de amostras ≈ 0), mas aparece no AUC-PR
(r ≈ −0.54).** O OptIForest mantém a qualidade de ranqueamento independentemente da escala
(de 129 a 619 mil amostras) — bom sinal de generalização. A correlação negativa no AUC-PR
é, em boa medida, reflexo do item 2: datasets maiores na amostra tendem a ter anomalias
mais raras, arrastando o AUC-PR para baixo.

## Conclusão para a generalização

O método **generaliza bem no eixo do tamanho** (escala não degrada o ROC) e é **sensível à
dimensionalidade** (alta dimensão degrada o ROC). O desempenho em AUC-PR é fortemente
condicionado pela raridade da anomalia — uma propriedade do *dado*, não falha do modelo.
Isso explica diretamente os dois datasets novos:

- **fraud** (29 feat, bem separável) → ROC 0.948: baixa dimensão + estrutura clara →
  generaliza bem, como o padrão (1) prevê.
- **annthyroid** (6 feat, anomalia sutil) → ROC 0.568: dimensão baixa **não** garante
  desempenho quando a anomalia é pouco separável; o threshold ε=55 (regra do paper para
  ≤10k amostras) agrava o caso. É o contraexemplo útil — mostra os limites da
  generalização.

## Ressalvas
- `kddcup99` usa o mesmo arquivo de `http` no pipeline público (proxy documentado em
  `research/reproduction.md`), então conta como ~21 datasets efetivamente distintos.
- 22 pontos é uma amostra pequena: as correlações indicam **tendências**, não leis. Reportar
  Pearson e Spearman juntos (resultados consistentes entre os dois) reforça a robustez.

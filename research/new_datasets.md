# Datasets novos (Requisito 2 do projeto)

Reprodução do artigo **OptIForest: Optimal Isolation Forest for Anomaly Detection**
(IJCAI 2023). A reprodução do artigo (Requisito 1) usa os 20 datasets do paper. Este
documento cobre o **Requisito 2**: avaliar o sistema em *outro* conjunto de dados, fora
dos 20 do artigo, com a escolha devidamente justificada.

## Por que estes datasets (e não outros)

Os 20 datasets do paper são: AD, campaign, Arrhythmia, cardio, backdoor, KDDCup99,
Celeba, mnist, Census, Donors, Cover, http, Ionosphere, smtp, Satellite, Shuttle, Spam,
Vowel, Waveform, Wine. Qualquer um destes **não** conta como "novo". Por isso candidatos
como Satellite/Shuttle/KDDCup99 foram descartados.

Critérios usados para escolher:
- **Fora dos 20 do paper** (exigência do enunciado).
- **Disponível no ADBench** → reaproveita o loader `npz` existente, zero código novo de I/O.
- **Runtime viável** (deadline curto): um dataset rápido + um pesado para cobrir os dois extremos.
- **Característica/domínio distintos** dos já reproduzidos (arrhythmia 274d médico,
  mnist 100d imagem, cover 10d enorme/raro) → fortalece a discussão.
- **Bem citados** na literatura de outlier detection → permite comparar resultados.

## Datasets escolhidos

### 1. annthyroid (médico — doença de tireoide)
- **Shape:** 7.200 amostras × 6 features, **7.42%** anomalias.
- **Threshold OptIForest:** 55 (regra do paper: ≤10k amostras e ≤1000 features).
- **Justificativa:** regime de **baixíssima dimensão** (6 features) num domínio médico,
  contrastando com o arrhythmia (274 features) também médico — isola o efeito da
  dimensionalidade. Runtime na casa de minutos.

### 2. fraud (finança — fraude em cartão de crédito)
- **Shape:** 284.807 amostras × 29 features, **0.17%** anomalias.
- **Threshold OptIForest:** 403 (regra do paper: >10k amostras).
- **Justificativa:** **domínio totalmente novo** (finança) e anomalia **ultra-rara**
  (0.17%), abaixo do menor caso do paper (cover, 0.96%) — testa a robustez do método em
  desbalanceamento extremo. Tamanho comparável ao cover (~5 h de execução).

## Configuração (idêntica à reprodução do paper)

- Árvores: **100** | branch: **0** (aproximação-e do paper) | threshold_mode: **paper**
- **15 runs** por dataset, seeds 42–56 (`seed_base=42`).
- Métricas: **AUC-ROC** e **AUC-PR** (mesmas do paper).

Comando usado:

```bash
.venv/bin/python scripts/run_optiforest_study.py \
    --datasets annthyroid,fraud --runs 15 --version v1 --workers 2
```

## Fontes dos dados

| Dataset | Arquivo ADBench | Link |
|---|---|---|
| annthyroid | `2_annthyroid.npz` | https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/2_annthyroid.npz |
| fraud | `13_fraud.npz` | https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/13_fraud.npz |

- **ADBench** (benchmark de origem dos arquivos): https://github.com/Minqi824/ADBench
  — Han et al., "ADBench: Anomaly Detection Benchmark", NeurIPS 2022.
- O loader baixa de `.../Classical/<arquivo>.npz` (ver `ADBENCH_BASE_URL` em
  `scripts/run_optiforest_study.py`) e cacheia em `data/benchmarks/`.

## Resultados esperados (faixas de referência)

Como estes datasets **não** estão no paper do OptIForest, não há número oficial para
comparar. As faixas abaixo vêm de benchmarks públicos de isolation forest / iForest
(ADBench, ODDS, PyOD) e servem como **sanity-check** dos resultados da equipe e como base
para a discussão no relatório. Esperam-se valores na mesma ordem de grandeza — não
idênticos, pois o OptIForest usa learning-to-hash e o threshold de corte ε do paper.

| Dataset | Métrica | Faixa de referência (iForest/literatura) | Comentário |
|---|---|---|---|
| annthyroid | AUC-ROC | ~0.62 – 0.82 | Dataset notoriamente difícil; muito sensível ao sampling/threshold |
| annthyroid | AUC-PR | ~0.20 – 0.35 | Anomalia moderada (7.4%) |
| fraud | AUC-ROC | ~0.94 – 0.95 | Anomalias bem separáveis → ROC alto |
| fraud | AUC-PR | ~0.10 – 0.30 | Desbalanceamento extremo (0.17%) derruba a precisão |

**Observado no smoke-test local (annthyroid, config do paper, threshold=55):**
AUC-ROC ≈ 0.53–0.61, AUC-PR ≈ 0.10–0.14 por run individual.

> Nota de discussão: o AUC-ROC observado no annthyroid ficou **abaixo** da faixa típica de
> iForest. Isso é esperado e relevante para o relatório — o threshold ε=55 (amostragem
> pequena, regra do paper para datasets ≤10k) prejudica datasets de baixa dimensão e
> estrutura sutil como o annthyroid. Vale comparar com uma variação de threshold no estudo
> de ablação (Requisito 3, opcional).

### Fontes das faixas de referência
- **ADBench** (Han et al., NeurIPS 2022): https://github.com/Minqi824/ADBench
- **ODDS** (Outlier Detection DataSets), Stony Brook: https://odds.cs.stonybrook.edu
- **PyOD** benchmark: https://github.com/yzhao062/pyod

## Observação sobre `paper_shape_match`

Como estes datasets **não** estão no paper, o campo `paper_shape_match` no CSV compara
contra as estatísticas do próprio ADBench (registradas em `PaperStats`), não contra o
artigo. Ele serve apenas como checagem de integridade do download, não como validação de
reprodução.

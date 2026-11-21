# AI-TestGuard

[![tests](https://github.com/ibrahima-2/AI-TestGuard/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ibrahima-2/AI-TestGuard/actions/workflows/tests.yml)

Projet de Master 2 (Test logiciel) : stratégie de test et qualification d'un modèle de détection de fraude par carte bancaire, à partir du jeu de données Credit Card Fraud Detection (Kaggle).

Le but est d'établir, à l'aide de tests reproductibles, dans quelles conditions le modèle peut être considéré comme acceptable.

## Résultats principaux

- Modèle retenu : régression logistique (`class_weight="balanced"`, C = 0,01), avec un seuil réglé sur la validation.
- Sur le test (95 fraudes) : précision 0,875, rappel 0,737 pour un critère de 0,75, FPR 0,018 %, PR-AUC 0,681, ROC-AUC 0,961.
- 190 tests automatisés, couverture de 98 %, aucune erreur Ruff.
- Latence P95 de 1,4 ms par transaction, modèle de 4,6 Ko.
- Décision : la v1.0.0 n'est pas qualifiée pour un blocage automatique ; un pilote en mode observation est proposé pour la v1.1.0 (sans la variable `Time`). Voir le [rapport de qualification](reports/08_rapport_qualification.md).

## Organisation du dépôt

```
AI-TestGuard/
├── app/
│   ├── config.py          contrat de données et critères d'acceptation
│   ├── preprocessing.py   chargement, validation, dédoublonnage, découpage, prétraitement
│   ├── training.py        pipelines, réglage, seuil de décision
│   ├── prediction.py      prédiction, point d'entrée de type API, sauvegarde et chargement
│   ├── evaluation.py      métriques, critères, bootstrap, graphiques
│   ├── robustness.py      perturbations et mesure de sensibilité
│   ├── regression.py      référence et comparaison de non-régression
│   ├── monitoring.py      PSI, Kolmogorov-Smirnov, plages, taux d'alertes
│   ├── performance.py     latence, débit, mémoire
│   ├── security.py        recherche de secrets
│   └── cli.py             chaîne complète : python -m app.cli <commande>
├── tests/                 190 tests (pytest, Hypothesis)
├── data/
│   ├── README.md              source, récupération et caractéristiques du jeu
│   └── sample_creditcard.csv  échantillon réel de 3 150 lignes utilisé par la CI
├── reports/               rapports, résultats JSON, figures, références, rapport de tests
├── .github/workflows/tests.yml
├── pyproject.toml         configuration de pytest, coverage et Ruff
└── requirements.txt       versions des dépendances
```

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le jeu complet (144 Mo) n'est pas versionné. Les instructions de téléchargement sont dans [data/README.md](data/README.md) :

```bash
curl -fsSL -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

## Utilisation

```bash
# tests rapides (ceux exécutés par la CI à chaque Pull Request)
pytest -m "not slow" --cov=app

# tests lents, qui nécessitent le jeu complet
pytest -m slow

# chaîne complète : entraînement, robustesse, performance, dérive, non-régression
python -m app.cli all

# étapes séparées
python -m app.cli train
python -m app.cli robustness
python -m app.cli performance
python -m app.cli drift
python -m app.cli regression
python -m app.cli baseline-create --force   # recrée la référence (décision explicite)
```

## Rapports

| Partie de l'énoncé | Document |
|---|---|
| I | [Plan de test et matrice de traçabilité](reports/01_plan_de_test.md) |
| II | [Rapport d'exécution des tests](reports/tests/pytest_rapport.txt), [JUnit](reports/tests/junit.xml) |
| III | [Évaluation du modèle et analyse des erreurs](reports/02_evaluation_modele.md) |
| IV et VI | [Validation hors échantillon et non-régression](reports/04_non_regression.md) |
| V | [Robustesse et sensibilité](reports/03_robustesse.md) |
| VII | [Audit des biais](reports/05_audit_biais.md) |
| VIII | [Performance et sécurité opérationnelle](reports/06_performance_securite.md) |
| X | [Surveillance](reports/07_surveillance.md), [rapport final de qualification](reports/08_rapport_qualification.md) |

Les mesures brutes sont dans `reports/results/`, les figures dans `reports/figures/` et les références de non-régression dans `reports/baseline/`.

## Intégration continue

Le workflow [.github/workflows/tests.yml](.github/workflows/tests.yml) comporte deux jobs.

Le job `quality` s'exécute à chaque push sur `main` et à chaque Pull Request. Il lance `ruff check`, `ruff format --check` et `pytest -m "not slow"` avec une couverture minimale de 90 %, puis publie le rapport JUnit et la couverture. Un échec de lint, de test ou de couverture fait échouer le job.

Le job `full-evaluation` se lance manuellement ou chaque lundi. Il télécharge le jeu complet, vérifie son empreinte SHA-256, exécute les tests lents et la commande `python -m app.cli all`, puis conserve les résultats.

La branche `main` est protégée : une modification doit passer par une Pull Request, et le job « Lint + tests rapides + couverture » doit réussir avant la fusion.

La branche `demo/echec-volontaire` (PR #1) introduit volontairement une erreur : `np.log1p` est remplacé par `np.log` dans le prétraitement de `Amount`. Un montant nul produit alors une valeur infinie, et le job de CI échoue.

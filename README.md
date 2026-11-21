# AI-TestGuard

Stratégie de test et qualification d'un modèle de Machine Learning de détection de fraude par carte bancaire (jeu de données *Credit Card Fraud Detection*, Kaggle).

L'objectif n'est pas seulement d'obtenir un bon score. Il s'agit de **démontrer, par des tests reproductibles et des résultats mesurables, dans quelles conditions le modèle est acceptable**.

## Résultat en bref

| | |
|---|---|
| Modèle retenu | Régression logistique (`class_weight="balanced"`, C = 0,01), seuil réglé sur la validation |
| Test (95 fraudes) | Précision **0,875** · Rappel **0,737** (critère 0,75 ✗) · FPR 0,018 % · PR-AUC 0,681 · ROC-AUC 0,961 |
| Tests automatisés | **190 réussis**, couverture **98 %**, Ruff sans erreur |
| Exécution | Latence P95 de 1,4 ms · modèle de 4,6 Ko |
| Décision | v1.0.0 **non qualifiée pour un blocage automatique** ; pilote en **mode observation** recommandé pour la v1.1.0, sans `Time` → [rapport de qualification](reports/08_rapport_qualification.md) |

## Structure

```
AI-TestGuard/
├── app/
│   ├── config.py          # contrat de données, critères d'acceptation (fixés avant entraînement)
│   ├── preprocessing.py   # chargement, validation, dédoublonnage, découpage, prétraitement
│   ├── training.py        # pipelines, réglage, seuil de décision
│   ├── prediction.py      # prédiction, point d'entrée de type API, sauvegarde et chargement vérifiés
│   ├── evaluation.py      # métriques, critères, bootstrap, graphiques
│   ├── robustness.py      # perturbations et mesure de sensibilité
│   ├── regression.py      # référence et comparaison de non-régression
│   ├── monitoring.py      # PSI, Kolmogorov-Smirnov, plages, taux d'alertes
│   ├── performance.py     # latence, débit, mémoire
│   ├── security.py        # recherche de secrets
│   └── cli.py             # chaîne complète : python -m app.cli <commande>
├── tests/                 # 190 tests (pytest + Hypothesis)
│   ├── test_preprocessing.py  test_prediction.py  test_integration.py
│   ├── test_robustness.py     test_regression.py
│   └── test_performance.py    test_security.py    conftest.py
├── data/
│   ├── README.md              # source, licence, récupération, caractéristiques
│   └── sample_creditcard.csv  # échantillon réel versionné (3 150 lignes) pour la CI
├── reports/               # rapports, résultats JSON, figures, références, rapport de tests
├── .github/workflows/tests.yml
├── pyproject.toml         # configuration pytest, coverage, Ruff
└── requirements.txt       # versions figées
```

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le jeu complet (144 Mo) n'est pas versionné. Pour le récupérer, voir [data/README.md](data/README.md) :

```bash
curl -fsSL -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

## Utilisation

```bash
# Tests rapides (exécutés par la CI à chaque Pull Request)
pytest -m "not slow" --cov=app

# Tests lents (jeu complet requis)
pytest -m slow

# Chaîne complète : entraînement, robustesse, performance, dérive, non-régression
python -m app.cli all

# Étapes séparées
python -m app.cli train          # réglage sur la validation, gel du modèle, évaluation unique sur le test
python -m app.cli robustness
python -m app.cli performance
python -m app.cli drift
python -m app.cli regression     # compare des versions candidates à reports/baseline/baseline_full.json
python -m app.cli baseline-create --force   # (re)crée la référence : décision explicite uniquement
```

## Rapports

| Partie | Rapport |
|---|---|
| I | [Plan de test et matrice de traçabilité](reports/01_plan_de_test.md) |
| II | [Rapport d'exécution des tests](reports/tests/pytest_rapport.txt) · [JUnit](reports/tests/junit.xml) |
| III | [Évaluation du modèle et analyse des erreurs](reports/02_evaluation_modele.md) |
| IV, VI | [Validation hors échantillon et non-régression](reports/04_non_regression.md) |
| V | [Robustesse et sensibilité](reports/03_robustesse.md) |
| VII | [Audit des biais](reports/05_audit_biais.md) |
| VIII | [Performance et sécurité opérationnelle](reports/06_performance_securite.md) |
| X | [Surveillance](reports/07_surveillance.md) · [**Rapport final de qualification**](reports/08_rapport_qualification.md) |

Résultats bruts : `reports/results/*.json` · Figures : `reports/figures/` · Références : `reports/baseline/`.

## Intégration continue (Partie IX)

Le workflow [.github/workflows/tests.yml](.github/workflows/tests.yml) comporte deux jobs.

**`quality`** s'exécute à chaque push sur `main` et à chaque Pull Request. Il lance :
- `ruff check` et `ruff format --check` ;
- `pytest -m "not slow"` avec une couverture minimale de 90 % ;
- la publication du rapport JUnit et de la couverture, avec un résumé dans l'interface GitHub.

Tout échec de lint, de test ou de couverture fait échouer le job.

**`full-evaluation`** se déclenche manuellement (`workflow_dispatch`) ou chaque lundi. Il télécharge le jeu complet, vérifie son empreinte SHA-256, puis exécute les tests lents et la chaîne `python -m app.cli all`. Les résultats sont conservés comme artefacts.

### Publication et protection de la branche

```bash
git remote add origin https://github.com/<compte>/AI-TestGuard.git
git push -u origin main
```

Pour protéger la branche : **Settings → Branches → Add branch ruleset** sur `main`, cocher *Require a pull request before merging* et *Require status checks to pass*, puis choisir **« Lint + tests rapides + couverture »**.

### Preuve d'un échec volontaire correctement détecté

La branche `demo/echec-volontaire` introduit une régression réaliste : `np.log1p` est remplacé par `np.log` dans le prétraitement de `Amount`. Un montant nul produit alors −∞, ce que les tests de contrat détectent.

```bash
git push origin demo/echec-volontaire
# puis ouvrir une Pull Request vers main : le job « quality » doit échouer
```

# 4. Validation hors échantillon et non-régression

> Sources : `reports/baseline/baseline_full.json` (référence), `reports/results/regression.json` (`python -m app.cli regression`).
> Tests automatisés : `tests/test_regression.py`.

## 4.1 Validation hors échantillon

| Règle | Mise en œuvre | Preuve |
|---|---|---|
| Séparer avant toute transformation apprise | Dédoublonnage, puis découpage, puis ajustement du pipeline sur l'entraînement uniquement | `prepare_splits` dans `app/cli.py` ; `test_scaler_is_fitted_on_training_data_only` |
| Entraînement / validation / test | 60 / 20 / 20, stratifié, graine 42 | `test_split_sizes_and_disjoint_indices`, `test_split_preserves_class_proportion` |
| Hyperparamètres et seuil sur les données autorisées | Grille et seuil évalués sur la validation | `tune_model`, `select_threshold` ; `test_threshold_is_derived_from_validation_only_and_reproducible` |
| Test réservé à l'évaluation finale | Modèle sauvegardé (SHA-256) *avant* le calcul des métriques de test | `cmd_train`, étape 3 puis étape 4 |
| Détecter le surapprentissage | Écart de PR-AUC entre entraînement et test ≤ 0,10 | Régression logistique : 0,047 (OK) ; arbre : 0,130 (surapprentissage) |

**Constat.** Pour la régression logistique, les PR-AUC valent 0,728 sur l'entraînement, 0,791 sur la validation et 0,681 sur le test. Elles varient de 0,11 d'une partition à l'autre, sans surapprentissage (le modèle est très régularisé, C = 0,01). Cette variabilité vient du faible nombre de fraudes par partition (94 ou 95) : un écart de quelques fraudes déplace sensiblement chaque métrique.

## 4.2 Référence (baseline)

La référence est créée **manuellement** avec `python -m app.cli baseline-create`. La fonction `save_baseline` **refuse d'écraser** une référence existante sans l'option `--force`, qui correspond à une décision explicite (`test_baseline_is_never_overwritten_without_force`). Elle n'est donc jamais recalculée automatiquement à partir d'une nouvelle version, ce qui masquerait les régressions.

Contenu enregistré (`reports/baseline/baseline_full.json`) :

| Élément | Valeur |
|---|---|
| Version du code | commit Git (`code_version`) et version applicative 1.0.0 |
| Paramètres du modèle | régression logistique, C = 0,01, `class_weight="balanced"`, 30 variables |
| Bibliothèques | Python 3.13.5, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, joblib 1.6.0 |
| Graine | 42 |
| Seuil de décision | 0,999992 |
| Métriques sur le jeu de validation fixe | PR-AUC 0,7906 ; ROC-AUC 0,9797 ; rappel 0,8191 ; précision 0,8851 ; FPR 0,0177 % |
| Matrice de confusion | TP 77, FP 10, FN 17, TN 56 641 |
| Jeu de validation | 56 745 lignes, 94 fraudes, empreinte SHA-256 du contenu |

Une seconde référence, `baseline_sample.json`, porte sur l'échantillon versionné. Elle permet à la CI d'exécuter la non-régression à chaque Pull Request sans télécharger le jeu complet.

## 4.3 Règles de décision

Pour chaque métrique, on calcule la dégradation par rapport à la référence :

| Métrique | Seuil « investiguer » | Seuil « rejeter » |
|---|---|---|
| PR-AUC | > 0,01 | > 0,03 |
| ROC-AUC | > 0,005 | > 0,01 |
| Rappel | > 0,02 (environ 2 fraudes sur 94) | > 0,05 |
| Précision | > 0,03 | > 0,08 |
| FPR (hausse) | > 0,05 point | > 0,10 point |

La décision globale est la plus sévère des décisions par métrique. Un jeu de validation différent de celui de la référence (empreinte modifiée) entraîne au minimum « investiguer », car la comparaison n'est alors plus probante.

## 4.4 Rapport comparatif : référence contre nouvelles versions

Trois versions candidates sont entraînées sur les mêmes données d'entraînement, puis mesurées sur le **même jeu de validation** que la référence. Le test n'est pas utilisé.

| Métrique | Référence v1.0.0 | v1.0.1 — réentraînement identique | v1.1.0 — sans `Time` | v1.1.1 — sans pondération des classes |
|---|---|---|---|---|
| PR-AUC | 0,7906 | 0,7906 (=) | 0,7912 (+0,0006) | 0,7949 (+0,0043) |
| ROC-AUC | 0,9797 | 0,9797 (=) | 0,9809 (+0,0012) | 0,9788 (−0,0010) |
| Rappel | 0,8191 | 0,8191 (=) | 0,8085 (−0,0106) | 0,8191 (=) |
| Précision | 0,8851 | 0,8851 (=) | 0,8837 (−0,0013) | **0,7778 (−0,1073)** |
| FPR | 0,018 % | 0,018 % (=) | 0,018 % (=) | 0,039 % (+0,021 pt) |
| Seuil | 0,999992 | 0,999992 | 0,999996 | 0,092 |
| TP / FP / FN | 77 / 10 / 17 | 77 / 10 / 17 | 76 / 10 / 18 | 77 / 22 / 17 |
| **Décision** | — | **ACCEPTER** | **ACCEPTER** | **REJETER** |

### Décisions motivées

- **v1.0.1 — réentraînement à l'identique : accepter.** Les métriques et le seuil sont identiques au bit près. C'est la **preuve de reproductibilité** du pipeline : même code, mêmes données, même graine, donc même modèle. Le rechargement du modèle sauvegardé donne aussi un écart maximal de 0,0 sur les probabilités de test (`reload_max_abs_diff` dans `evaluation.json`).
- **v1.1.0 — suppression de `Time` : accepter.** Toutes les dégradations restent sous les seuils : une fraude détectée en moins (rappel −1,1 point, sous la tolérance de 2 points), et une PR-AUC légèrement meilleure. Cette version corrige l'anomalie ANO-03 (dépendance à une variable non stationnaire, voir le rapport 03). **Elle est recommandée comme prochaine version candidate**, sous réserve de sa confirmation sur des données nouvelles.
- **v1.1.1 — suppression de la pondération des classes : rejeter.** La précision perd 10,7 points (au-delà du seuil de rejet de 8 points) : 22 fausses alertes au lieu de 10, à rappel égal. Sa PR-AUC est pourtant légèrement meilleure. Cet exemple montre qu'**une métrique globale peut masquer une dégradation au point de fonctionnement réel**. C'est pour cela que la non-régression contrôle aussi les métriques au seuil.

## 4.5 Automatisation

| Contrôle | Données | Exécution |
|---|---|---|
| Logique de décision (8 tests) | Synthétiques | Chaque Pull Request |
| Contenu de la référence versionnée | `baseline_full.json`, `baseline_sample.json` | Chaque Pull Request |
| Non-régression du code courant | Échantillon et `baseline_sample.json` ; la CI échoue si la décision est « rejeter » | Chaque Pull Request |
| Non-régression complète | Jeu Kaggle et `baseline_full.json` ; exige « accepter » | Manuel ou hebdomadaire (`full-evaluation`) |

Sur l'échantillon, le test exige seulement une décision différente de « rejeter ». En effet, la validation de l'échantillon ne contient que 30 fraudes, et de légères différences d'arrondi entre plateformes (macOS en local, Linux en CI) peuvent déplacer le seuil d'une fraude. Cela suffit pour passer en « investiguer », sans qu'il s'agisse d'une régression.

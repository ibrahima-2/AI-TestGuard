# 4. Validation hors échantillon et non-régression

Sources : `reports/baseline/baseline_full.json` (référence) et `reports/results/regression.json` (`python -m app.cli regression`). Les tests correspondants sont dans `tests/test_regression.py`.

## 4.1 Validation hors échantillon

| Règle de l'énoncé | Mise en œuvre | Preuve |
|---|---|---|
| Séparer les données avant toute transformation apprise | Dédoublonnage, puis découpage, puis ajustement du pipeline sur l'entraînement seulement | `prepare_splits` dans `app/cli.py`, `test_scaler_is_fitted_on_training_data_only` |
| Utiliser des jeux d'entraînement, de validation et de test | 60 / 20 / 20, stratifié, graine 42 | `test_split_sizes_and_disjoint_indices`, `test_split_preserves_class_proportion` |
| Régler les hyperparamètres et le seuil sur les données autorisées | Grille et seuil évalués sur la validation | `tune_model`, `select_threshold`, `test_threshold_is_derived_from_validation_only_and_reproducible` |
| Réserver le test à l'évaluation finale | Le modèle est sauvegardé avec son empreinte avant le calcul des métriques de test | `cmd_train`, étapes 3 et 4 |
| Comparer entraînement et test pour détecter le surapprentissage | Écart de PR-AUC limité à 0,10 | 0,047 pour la régression logistique, 0,130 pour l'arbre |

Pour la régression logistique, la PR-AUC vaut 0,728 sur l'entraînement, 0,791 sur la validation et 0,681 sur le test. L'écart entre partitions atteint 0,11 sans qu'il y ait surapprentissage, le modèle étant fortement régularisé (C = 0,01). Cette variabilité s'explique par le faible nombre de fraudes dans chaque partition (94 ou 95) : quelques fraudes de plus ou de moins suffisent à déplacer les métriques.

## 4.2 Référence

La référence est créée à la main avec `python -m app.cli baseline-create`. La fonction `save_baseline` refuse d'écraser une référence existante, sauf si l'option `--force` est passée explicitement (`test_baseline_is_never_overwritten_without_force`). La référence n'est donc jamais recalculée automatiquement à partir d'une nouvelle version, ce qui masquerait les régressions.

Contenu de `reports/baseline/baseline_full.json` :

| Élément | Valeur |
|---|---|
| Version du code | Commit Git (`code_version`) et version applicative 1.0.0 |
| Modèle | Régression logistique, C = 0,01, `class_weight="balanced"`, 30 variables |
| Bibliothèques | Python 3.13.5, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, joblib 1.6.0 |
| Graine | 42 |
| Seuil | 0,999992 |
| Métriques sur le jeu de validation fixe | PR-AUC 0,7906, ROC-AUC 0,9797, rappel 0,8191, précision 0,8851, FPR 0,0177 % |
| Matrice de confusion | TP 77, FP 10, FN 17, TN 56 641 |
| Jeu de validation | 56 745 lignes, 94 fraudes, empreinte SHA-256 du contenu |

Une seconde référence, `baseline_sample.json`, est calculée sur l'échantillon versionné. Elle permet d'exécuter la non-régression à chaque Pull Request sans télécharger le jeu complet.

## 4.3 Règles de décision

Pour chaque métrique, on mesure la dégradation par rapport à la référence :

| Métrique | Seuil « investiguer » | Seuil « rejeter » |
|---|---|---|
| PR-AUC | baisse > 0,01 | baisse > 0,03 |
| ROC-AUC | baisse > 0,005 | baisse > 0,01 |
| Rappel | baisse > 0,02 (environ 2 fraudes sur 94) | baisse > 0,05 |
| Précision | baisse > 0,03 | baisse > 0,08 |
| FPR | hausse > 0,05 point | hausse > 0,10 point |

La décision globale est la plus défavorable des décisions par métrique. Si le jeu de validation n'a pas la même empreinte que celui de la référence, la décision est au minimum « investiguer », car la comparaison n'est plus valable.

## 4.4 Comparaison avec trois versions candidates

Les trois versions sont entraînées sur les mêmes données d'entraînement et mesurées sur le même jeu de validation que la référence. Le test n'est pas utilisé.

| Métrique | Référence v1.0.0 | v1.0.1 : réentraînement à l'identique | v1.1.0 : sans `Time` | v1.1.1 : sans pondération des classes |
|---|---|---|---|---|
| PR-AUC | 0,7906 | 0,7906 | 0,7912 | 0,7949 |
| ROC-AUC | 0,9797 | 0,9797 | 0,9809 | 0,9788 |
| Rappel | 0,8191 | 0,8191 | 0,8085 | 0,8191 |
| Précision | 0,8851 | 0,8851 | 0,8837 | 0,7778 |
| FPR | 0,018 % | 0,018 % | 0,018 % | 0,039 % |
| Seuil | 0,999992 | 0,999992 | 0,999996 | 0,092 |
| TP / FP / FN | 77 / 10 / 17 | 77 / 10 / 17 | 76 / 10 / 18 | 77 / 22 / 17 |
| Décision | | accepter | accepter | rejeter |

Version v1.0.1, acceptée. Les métriques et le seuil sont identiques à la référence. Le pipeline est donc reproductible : avec le même code, les mêmes données et la même graine, on obtient le même modèle. Le rechargement du modèle sauvegardé ne change pas non plus les probabilités de test (`reload_max_abs_diff` = 0,0 dans `evaluation.json`). La même évaluation exécutée sur GitHub Actions (Linux) donne des résultats identiques à ceux obtenus en local (macOS).

Version v1.1.0, acceptée. Toutes les dégradations restent sous les seuils : une fraude détectée de moins (−1,1 point de rappel, pour une tolérance de 2 points) et une PR-AUC légèrement meilleure. Cette version supprime la dépendance à `Time` relevée dans le rapport 03. Elle est proposée comme prochaine version candidate, sous réserve d'une confirmation sur des données nouvelles.

Version v1.1.1, rejetée. La précision baisse de 10,7 points, au-delà du seuil de rejet de 8 points : 22 fausses alertes au lieu de 10 pour le même nombre de fraudes détectées. Sa PR-AUC est pourtant un peu meilleure que celle de la référence. Cet exemple montre qu'il faut contrôler les métriques au seuil de décision, et pas seulement une mesure globale comme la PR-AUC.

## 4.5 Automatisation

| Contrôle | Données | Exécution |
|---|---|---|
| Logique de décision (8 tests) | Synthétiques | À chaque Pull Request |
| Contenu des références versionnées | `baseline_full.json`, `baseline_sample.json` | À chaque Pull Request |
| Non-régression du code courant | Échantillon et `baseline_sample.json` ; échec si la décision est « rejeter » | À chaque Pull Request |
| Non-régression complète | Jeu Kaggle et `baseline_full.json` ; la décision doit être « accepter » | Manuellement ou chaque lundi |

Sur l'échantillon, le test n'exige qu'une décision différente de « rejeter ». Le jeu de validation de l'échantillon ne contient que 30 fraudes, et de faibles écarts d'arrondi entre plateformes pourraient déplacer le seuil d'une fraude, ce qui suffit à passer en « investiguer » sans qu'il s'agisse d'une régression.

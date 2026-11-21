# 1. Plan de test

## 1.1 Objet

Ce plan décrit comment le système de détection de fraude est testé : ce qui est testé, pourquoi, avec quels outils, et selon quels critères le modèle est accepté ou rejeté. L'objectif est de qualifier le système à partir de tests reproductibles, et pas seulement de mesurer un score.

## 1.2 Périmètre

| Composant | Contenu | Module |
|---|---|---|
| Données | Chargement du CSV, contrôle du schéma, des types et des valeurs, suppression des doublons | `app/preprocessing.py` |
| Prétraitement | `log1p` puis standardisation de `Amount`, standardisation des autres variables ; ajusté sur l'entraînement uniquement | `app/preprocessing.py` |
| Modèle | Pipeline scikit-learn (prétraitement et modèle), réglage sur la validation | `app/training.py` |
| Prédiction | Probabilité, seuil de décision, point d'entrée de type API, sauvegarde et chargement vérifiés | `app/prediction.py` |
| Évaluation | Métriques, critères d'acceptation, intervalles de confiance, graphiques | `app/evaluation.py` |
| Intégration | Chaîne complète en ligne de commande, non-régression, surveillance, CI | `app/cli.py`, `app/regression.py`, `app/monitoring.py`, `.github/workflows/tests.yml` |

Sont hors périmètre : une API HTTP déployée, les données de production actuelles, l'interface des analystes et les données démographiques des clients, qui ne figurent pas dans le jeu de données.

## 1.3 Exigences

Exigences fonctionnelles :

- EF1. Rejeter toute donnée qui ne respecte pas le contrat : colonnes, types, valeurs manquantes ou infinies, montant ou temps négatif, cible différente de 0 et 1.
- EF2. Produire pour chaque transaction valide une probabilité finie comprise entre 0 et 1, et une décision cohérente avec le seuil.
- EF3. Évaluer le modèle sur des données qui n'ont servi ni à l'entraînement, ni au réglage, ni au choix du seuil.
- EF4. Sauvegarder et recharger le pipeline sans modifier les prédictions.
- EF5. Comparer chaque nouvelle version à une référence figée.

Exigences non fonctionnelles :

- ENF1. Performance prédictive : rappel ≥ 0,75, précision ≥ 0,30, FPR ≤ 0,5 %, PR-AUC ≥ 0,60, ROC-AUC ≥ 0,90, écart de PR-AUC entre entraînement et test ≤ 0,10.
- ENF2. Reproductibilité : les mêmes données, le même code et la même graine donnent les mêmes résultats.
- ENF3. Robustesse : chaque type d'entrée perturbée est rejeté, traité ou signalé, selon une règle définie à l'avance.
- ENF4. Coût d'exécution : latence P95 ≤ 50 ms par transaction, débit ≥ 1 000 transactions/s, modèle ≤ 10 Mo, surcoût mémoire ≤ 200 Mo.
- ENF5. Sécurité opérationnelle : pas de secret dans le dépôt, pas de donnée de transaction dans les journaux, refus d'un modèle modifié.
- ENF6. Automatisation : les tests s'exécutent à chaque Pull Request et un échec est signalé.

Les seuils sont définis dans `app/config.py`. Ils ont été fixés avant le premier entraînement et n'ont pas été modifiés ensuite.

| Critère | Valeur | Justification |
|---|---|---|
| Rappel | ≥ 0,75 | Intercepter au moins trois fraudes sur quatre |
| Précision | ≥ 0,30 | Au plus 2,3 fausses alertes par fraude détectée environ, ce qui reste traitable par une équipe d'analystes |
| FPR | ≤ 0,5 % | Au plus un client légitime sur deux cents dérangé à tort |
| PR-AUC | ≥ 0,60 | Mesure adaptée à une classe rare ; un modèle naïf obtient 0,0017 |
| ROC-AUC | ≥ 0,90 | Capacité de discrimination indépendante du seuil |
| Seuil de décision | rappel ≥ 0,80 sur la validation | Marge de cinq points au-dessus du critère de rappel |
| Latence P95 | ≤ 50 ms | Le scoring n'est qu'une étape d'une autorisation de paiement, qui dure quelques centaines de millisecondes |

## 1.4 Matrice de traçabilité

Les lignes T01 à T07 reprennent la matrice de l'énoncé. Les lignes T08 à T25 ont été ajoutées. Pour chaque exigence, la dernière colonne donne le test ou le fichier de résultats qui sert de preuve.

| ID | Exigence | Risque maîtrisé | Type | Résultat attendu | Preuve |
|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Prédiction sur un schéma incomplet | Données | Rejet, avec le nom de la colonne manquante | `test_missing_column_is_rejected_and_named` |
| T02 | Les types sont conformes | Texte ou booléen converti sans contrôle | Unitaire | Rejet des types non numériques | `test_text_column_is_rejected`, `test_boolean_column_is_rejected` |
| T03 | Le prétraitement ne produit pas de valeur non finie | NaN propagé jusqu'au score | Unitaire | Aucune valeur NaN ou infinie en sortie | `test_preprocessor_output_is_finite_and_keeps_dimensions`, `test_no_non_finite_values_after_preprocessing` |
| T04 | La probabilité est valide | Score inutilisable en aval | Fonctionnel ML | Valeur finie entre 0 et 1 | `test_probabilities_are_finite_and_between_0_and_1`, `test_any_valid_transaction_gets_valid_probability` |
| T05 | L'évaluation porte sur des données indépendantes | Performance surestimée | ML | Test non utilisé pour le réglage | `test_partitions_are_disjoint_and_free_of_shared_duplicates`, `test_threshold_is_derived_from_validation_only_and_reproducible` |
| T06 | Les résultats sont reproductibles | Résultats impossibles à vérifier | Non-régression | Écart dans la tolérance | `test_current_code_does_not_regress_on_sample`, `test_training_is_reproducible` |
| T07 | Les tests s'exécutent sur GitHub | Régression non détectée | CI | Workflow réussi ou échec signalé | `.github/workflows/tests.yml` |
| T08 | Les valeurs manquantes ou infinies sont rejetées sans imputation | Imputation qui masque une anomalie | Unitaire | Rejet, avec le nom de la colonne | `test_missing_value_is_rejected_and_column_named`, `test_infinite_value_is_rejected` |
| T09 | La cible vaut 0 ou 1 | Étiquettes corrompues | Unitaire | Rejet de 2, −1, 0,5 et NaN | `test_target_values_other_than_0_and_1_are_rejected` |
| T10 | Les doublons sont supprimés avant le découpage | Même transaction dans l'entraînement et le test | Intégration | Aucune ligne commune entre partitions | `test_remove_duplicates_keeps_first_occurrence`, `test_without_deduplication_duplicates_leak_across_splits` |
| T11 | Le découpage est stratifié (60/20/20) | Partitions non représentatives | Unitaire | Taux de fraude conservé à 1,5 point près | `test_split_preserves_class_proportion` |
| T12 | Les transformations sont ajustées sur l'entraînement seul | Fuite par les statistiques du test | Intégration | Moyennes du scaler égales à celles de l'entraînement | `test_scaler_is_fitted_on_training_data_only` |
| T13 | Le seuil est choisi sur la validation | Seuil réglé sur le test, ou seuil arbitraire de 0,5 | Unitaire ML | Rappel minimal respecté, précision maximale | `test_threshold_meets_recall_constraint_with_best_precision` |
| T14 | Les critères ENF1 sont respectés sur le test | Modèle inutile ou trop coûteux | ML | Tous les critères satisfaits | `reports/results/evaluation.json` |
| T15 | Pas de surapprentissage | Dégradation en production | ML | Écart de PR-AUC ≤ 0,10 | `test_no_overfitting_gap_on_sample` |
| T16 | Les prédictions sont identiques après rechargement | Modèle déployé différent du modèle validé | Intégration | Écart ≤ 1e-12 | `test_predictions_identical_after_save_and_reload` |
| T17 | Un modèle modifié est refusé | Exécution de code via un fichier pickle | Sécurité | Refus si l'empreinte SHA-256 diffère | `test_tampered_model_is_refused` |
| T18 | La sensibilité au bruit est mesurée | Décisions instables | Robustesse | Effet faible pour un bruit faible, croissant avec l'amplitude | `test_small_noise_has_small_effect`, `test_sensitivity_grows_with_noise_amplitude` |
| T19 | Les montants extrêmes sont traités et signalés | Plantage ou score aberrant | Robustesse | Probabilité finie et alerte hors plage | `test_extreme_amounts_are_scored_and_flagged_out_of_range` |
| T20 | La dérive de distribution est détectée | Dégradation silencieuse | Surveillance | PSI ≥ 0,20 sur les seules variables décalées | `test_drift_report_flags_only_shifted_features` |
| T21 | Chaque version est comparée à une référence figée | Régression acceptée sans le savoir | Non-régression | Décision accepter, investiguer ou rejeter | `test_large_degradation_is_rejected`, `test_baseline_is_never_overwritten_without_force` |
| T22 | Le coût d'exécution respecte ENF4 | Latence incompatible avec le paiement | Performance | Les quatre seuils sont respectés | `tests/test_performance.py` |
| T23 | Le contrat d'API est strict | Champs injectés ou manquants | Sécurité | Requête rejetée | `test_score_records_rejects_malformed_records` |
| T24 | Les journaux ne contiennent pas de données de transaction | Fuite de données sensibles | Sécurité | Seuls les volumes et les noms de colonnes apparaissent | `test_rejected_request_does_not_log_transaction_values` |
| T25 | Le dépôt ne contient pas de secret | Publication d'une clé | Sécurité | Aucun résultat du scanner | `test_repository_contains_no_secret` |

## 1.5 Organisation des tests

| Niveau | Données | Exécution |
|---|---|---|
| Tests unitaires | Données synthétiques générées dans `tests/conftest.py` | À chaque Pull Request |
| Tests d'intégration | Échantillon réel versionné, `data/sample_creditcard.csv` (3 150 lignes) | À chaque Pull Request |
| Tests par propriétés (Hypothesis) | Données générées | À chaque Pull Request |
| Tests lents | Jeu Kaggle complet | Manuellement ou chaque lundi |
| Expériences (`python -m app.cli all`) | Jeu Kaggle complet | Manuellement ou chaque lundi |

Outils utilisés : pytest et pytest-cov (seuil de couverture de 90 % en CI), Hypothesis, scikit-learn, pandas, NumPy et SciPy, joblib, psutil et `time.perf_counter`, Ruff, GitHub Actions.

Les preuves sont conservées dans le dépôt : `reports/results/` (mesures au format JSON), `reports/figures/`, `reports/baseline/` (références de non-régression) et `reports/tests/` (rapport d'exécution des tests).

## 1.6 Critères d'acceptation et de rejet

Le modèle est accepté si tous les critères ENF1 sont respectés sur le test, si les exigences ENF2 à ENF5 sont vérifiées et si aucune anomalie bloquante n'est ouverte.

Il est rejeté si un critère de performance prédictive n'est pas atteint sur le test, si une fuite de données est constatée, si les résultats ne sont pas reproductibles ou si une anomalie de sécurité bloquante est ouverte.

Une investigation est nécessaire lorsque la dégradation mesurée en non-régression se situe entre les seuils « investiguer » et « rejeter », ou lorsqu'un critère est respecté sur la validation mais pas sur le test.

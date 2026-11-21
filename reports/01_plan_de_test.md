# 1. Plan de test — AI-TestGuard

## 1.1 Objet

Qualifier un système de détection de fraude par carte bancaire fondé sur un modèle de Machine Learning. Il s'agit de démontrer, à partir de tests reproductibles et de résultats mesurables, dans quelles conditions ce système est acceptable. Un bon score sur un jeu de données ne suffit pas.

## 1.2 Périmètre du système

| Composant | Contenu | Module |
|---|---|---|
| Données | Chargement CSV, contrôle du schéma, des types et des valeurs, suppression des doublons | `app/preprocessing.py` |
| Prétraitement | `log1p` + standardisation de `Amount`, standardisation des autres variables, ajustés sur l'entraînement uniquement | `app/preprocessing.py` |
| Modèle | Pipeline scikit-learn (prétraitement + régression logistique ou arbre de décision), réglage sur la validation | `app/training.py` |
| Prédiction | Probabilité, seuil de décision, point d'entrée de type API, sauvegarde et chargement vérifiés | `app/prediction.py` |
| Évaluation | Métriques, critères d'acceptation, intervalles de confiance, graphiques | `app/evaluation.py` |
| Intégration | Chaîne complète en ligne de commande, non-régression, surveillance, CI | `app/cli.py`, `app/regression.py`, `app/monitoring.py`, `.github/workflows/tests.yml` |

**Hors périmètre :** API HTTP déployée, base de données de production, interface des analystes, données démographiques des clients (absentes du jeu de données).

## 1.3 Exigences

### Fonctionnelles

- **EF1** Rejeter explicitement toute donnée qui viole le contrat : colonnes, types, valeurs manquantes ou infinies, montant ou temps négatif, cible hors {0, 1}.
- **EF2** Produire, pour chaque transaction valide, une probabilité de fraude finie comprise entre 0 et 1, et une décision cohérente avec le seuil.
- **EF3** Évaluer le modèle sur des données jamais utilisées pour l'entraînement, le réglage ou le choix du seuil.
- **EF4** Sauvegarder puis recharger le pipeline sans modifier les prédictions.
- **EF5** Comparer toute nouvelle version à une référence figée.

### Non fonctionnelles

- **ENF1** Performance prédictive : rappel ≥ 0,75 ; précision ≥ 0,30 ; FPR ≤ 0,5 % ; PR-AUC ≥ 0,60 ; ROC-AUC ≥ 0,90 ; écart de PR-AUC entre entraînement et test ≤ 0,10.
- **ENF2** Reproductibilité : mêmes données, même code et même graine donnent les mêmes résultats.
- **ENF3** Robustesse : comportement contrôlé (rejet, traitement ou alerte) face à des entrées perturbées.
- **ENF4** Coût d'exécution : latence P95 ≤ 50 ms par transaction, débit ≥ 1 000 transactions/s, modèle ≤ 10 Mo, surcoût mémoire ≤ 200 Mo.
- **ENF5** Sécurité opérationnelle : aucun secret dans le dépôt, aucune donnée de transaction dans les journaux, modèle altéré refusé.
- **ENF6** Automatisation : tests exécutés à chaque Pull Request, échec signalé clairement.

### Justification des critères (fixés avant tout entraînement, dans `app/config.py`)

| Critère | Valeur | Raison métier |
|---|---|---|
| Rappel | ≥ 0,75 | Au moins 3 fraudes sur 4 interceptées, pour limiter les pertes |
| Précision | ≥ 0,30 | Au plus environ 2,3 fausses alertes par fraude détectée : charge supportable pour une équipe d'analystes |
| FPR | ≤ 0,5 % | Au plus 1 client légitime sur 200 dérangé à tort |
| PR-AUC | ≥ 0,60 | Mesure adaptée à une classe rare ; un modèle naïf obtient 0,0017 |
| ROC-AUC | ≥ 0,90 | Bonne discrimination, quel que soit le seuil |
| Seuil | rappel ≥ 0,80 sur la validation | Marge de 5 points au-dessus du critère pour absorber la variabilité |
| Latence P95 | ≤ 50 ms | Une autorisation de paiement répond en quelques centaines de ms ; le scoring n'en est qu'une étape |

## 1.4 Matrice de traçabilité

Chaque ligne relie une exigence au risque qu'elle couvre et au test qui en apporte la preuve. Toutes les preuves sont des tests automatisés ou des fichiers de résultats versionnés.

| ID | Exigence | Risque maîtrisé | Type de test | Résultat attendu | Preuve |
|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Prédiction sur un schéma incomplet | Données | Aucune colonne indispensable manquante ; sinon rejet nommant la colonne | `test_preprocessing.py::test_missing_column_is_rejected_and_named` |
| T02 | Les données respectent les types attendus | Texte ou booléen converti silencieusement | Unitaire | Types conformes au contrat ; sinon rejet | `test_text_column_is_rejected`, `test_boolean_column_is_rejected` |
| T03 | Le prétraitement ne produit pas de valeurs non finies | NaN propagé jusqu'au score | Unitaire | Aucune valeur NaN ou infinie en sortie | `test_preprocessor_output_is_finite_and_keeps_dimensions`, `test_preprocessor_handles_zero_and_extreme_amounts`, `test_integration.py::test_no_non_finite_values_after_preprocessing` |
| T04 | Le modèle produit une probabilité valide | Score inutilisable par l'aval | Fonctionnel ML | Probabilité entre 0 et 1, finie | `test_prediction.py::test_probabilities_are_finite_and_between_0_and_1`, `test_any_valid_transaction_gets_valid_probability` (Hypothesis), `test_invalid_model_output_is_detected` |
| T05 | Les performances sont évaluées sur des données indépendantes | Performance surestimée | ML | Jeu de test non utilisé pour le réglage | `test_partitions_are_disjoint_and_free_of_shared_duplicates`, `test_threshold_is_derived_from_validation_only_and_reproducible`, `app/cli.py` (gel du modèle avant le test) |
| T06 | Les résultats sont reproductibles | Résultats impossibles à vérifier | Non-régression | Écart compatible avec la tolérance définie | `test_regression.py::test_current_code_does_not_regress_on_sample`, `test_training_is_reproducible`, `test_split_is_reproducible_with_same_seed` |
| T07 | Les tests s'exécutent sur GitHub | Régression introduite sans être vue | CI | Workflow réussi ou échec correctement signalé | `.github/workflows/tests.yml` |
| **T08** | Valeurs manquantes ou infinies rejetées, sans imputation | Imputation silencieuse qui masque une anomalie ou un contournement | Unitaire | `DataValidationError` nommant la colonne | `test_missing_value_is_rejected_and_column_named`, `test_infinite_value_is_rejected`, `test_nullable_dtype_with_missing_value_is_rejected` |
| **T09** | Cible limitée à {0, 1} | Étiquettes corrompues pendant l'entraînement | Unitaire | Rejet des valeurs 2, −1, 0,5 et NaN | `test_target_values_other_than_0_and_1_are_rejected`, `test_missing_target_is_rejected` |
| **T10** | Doublons supprimés avant le découpage | **Fuite de données** : une même transaction dans le jeu d'entraînement et dans celui de test | Intégration | Aucune ligne commune entre entraînement et test | `test_remove_duplicates_keeps_first_occurrence`, `test_without_deduplication_duplicates_leak_across_splits` (lent) |
| **T11** | Découpage stratifié 60/20/20 | Partitions sans fraude ou non représentatives | Unitaire | Taux de fraude conservé (±1,5 point) | `test_split_preserves_class_proportion`, `test_split_sizes_and_disjoint_indices` |
| **T12** | Transformations ajustées sur l'entraînement uniquement | **Fuite** par les statistiques du test | Intégration | Moyennes du scaler égales à celles de l'entraînement | `test_scaler_is_fitted_on_training_data_only`, `test_scaler_uses_training_statistics_only` |
| **T13** | Seuil choisi sur la validation (rappel ≥ 0,80, précision maximale) | Seuil optimisé sur le test ; seuil arbitraire à 0,5 | Unitaire ML | Contrainte de rappel respectée, précision maximale | `test_threshold_meets_recall_constraint_with_best_precision`, `test_unreachable_recall_falls_back_to_lowest_threshold` |
| **T14** | Critères métier ENF1 respectés sur le test | Modèle inutile ou trop coûteux en fausses alertes | ML | Tous les critères satisfaits | `reports/results/evaluation.json`, `test_acceptance_criteria_pass_and_fail` |
| **T15** | Pas de surapprentissage | Performance qui s'effondre en production | ML | Écart de PR-AUC entre entraînement et test ≤ 0,10 | `evaluation.json`, `test_no_overfitting_gap_on_sample` |
| **T16** | Prédictions identiques après sauvegarde et rechargement | Modèle déployé différent du modèle validé | Intégration | Écart maximal ≤ 1e-12 | `test_predictions_identical_after_save_and_reload` |
| **T17** | Modèle altéré refusé | Exécution de code via un fichier pickle modifié | Sécurité | `ModelLoadError` si l'empreinte SHA-256 diffère | `test_tampered_model_is_refused`, `test_missing_metadata_is_refused`, `test_object_without_predict_proba_is_refused` |
| **T18** | Sensibilité aux perturbations mesurée | Décisions instables face au bruit | Robustesse | Effet faible pour un bruit faible ; effet croissant avec l'amplitude ; bascules près de la frontière | `test_small_noise_has_small_effect`, `test_sensitivity_grows_with_noise_amplitude`, `test_class_changes_concentrate_near_decision_boundary` |
| **T19** | Montants extrêmes traités avec une alerte | Plantage, ou score aberrant non signalé | Robustesse | Probabilité finie et alerte « hors plage » | `test_extreme_amounts_are_scored_and_flagged_out_of_range` |
| **T20** | Dérive de distribution détectée | Dégradation silencieuse en production | Surveillance | PSI ≥ 0,20 sur les variables décalées uniquement | `test_drift_report_flags_only_shifted_features`, `test_psi_is_low_for_same_distribution_and_high_for_shift` |
| **T21** | Nouvelle version comparée à une référence figée | Régression acceptée sans le savoir | Non-régression | Décision « accepter », « investiguer » ou « rejeter » ; la référence n'est jamais écrasée automatiquement | `test_large_degradation_is_rejected`, `test_baseline_is_never_overwritten_without_force` |
| **T22** | Coût d'exécution dans les limites ENF4 | Latence incompatible avec le paiement | Performance | Les quatre seuils sont respectés | `test_performance.py`, `reports/results/performance.json` |
| **T23** | Contrat d'API : champs manquants, supplémentaires ou mal typés refusés | Injection de champs, scoring sur des données partielles | Sécurité | Requête entière rejetée | `test_score_records_rejects_malformed_records`, `test_score_records_rejects_malformed_payload`, `test_score_records_limits_batch_size` |
| **T24** | Aucune donnée de transaction dans les journaux ni les messages d'erreur | Fuite de données sensibles | Sécurité | Seuls les volumes et les noms de colonnes apparaissent | `test_prediction_logs_volumes_without_transaction_values`, `test_rejected_request_does_not_log_transaction_values`, `test_error_message_does_not_leak_values` |
| **T25** | Aucun secret dans le dépôt | Clé d'API ou identifiant Kaggle publié | Sécurité | Scanner sans résultat ; fichiers sensibles ignorés par Git | `test_repository_contains_no_secret`, `test_sensitive_files_are_git_ignored` |

## 1.5 Stratégie d'exécution

| Niveau | Contenu | Données | Déclenchement |
|---|---|---|---|
| Unitaire | Validation, prétraitement, métriques, seuil, sauvegarde | Synthétiques (générées dans `conftest.py`) | Chaque Pull Request |
| Intégration | Chaîne complète sur données réelles, y compris la ligne de commande | Échantillon versionné `data/sample_creditcard.csv` (3 150 lignes) | Chaque Pull Request |
| Propriétés | Hypothesis : tout lot valide reste fini ; toute perturbation bornée respecte le contrat | Générées | Chaque Pull Request |
| Lent | Caractéristiques du jeu complet, preuve de fuite sans dédoublonnage, non-régression complète | Jeu Kaggle complet | Manuel ou hebdomadaire |
| Expériences | Entraînement, robustesse, performance, dérive, non-régression | Jeu Kaggle complet | `python -m app.cli all` |

## 1.6 Outils et preuves conservées

| Outil | Usage |
|---|---|
| pytest, pytest-cov | Exécution des tests, couverture (seuil CI : 90 %) |
| Hypothesis | Génération de cas de test |
| scikit-learn, pandas, NumPy, SciPy | Pipeline, métriques, test de Kolmogorov-Smirnov |
| joblib, SHA-256 | Persistance vérifiée du modèle |
| psutil, `time.perf_counter`, tracemalloc | Mesures de performance |
| Ruff | Style et règles de qualité, dont les règles de sécurité Bandit (`S`) |
| GitHub Actions | Intégration continue |

**Preuves versionnées :** `reports/results/*.json` (toutes les mesures), `reports/figures/*.png`, `reports/baseline/*.json` (références), `reports/tests/` (rapport d'exécution des tests), `reports/results/run_log.txt` (sortie de la chaîne complète).

## 1.7 Critères d'acceptation et de rejet du modèle

- **Accepté pour la suite :** tous les critères ENF1 respectés sur le jeu de test, ENF2 à ENF5 vérifiés, aucune anomalie bloquante ouverte.
- **Rejeté :** un critère de performance prédictive non respecté sur le test, une fuite de données avérée, une non-reproductibilité, ou une anomalie de sécurité bloquante.
- **À investiguer :** dégradation comprise entre les seuils « investiguer » et « rejeter » de la non-régression, ou critère respecté sur la validation mais pas sur le test.

Ces critères ne sont pas modifiés après observation du jeu de test.

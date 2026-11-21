# 6. Performance et sécurité opérationnelle

Source : `reports/results/performance.json` (`python -m app.cli performance`). Tests : `tests/test_performance.py` et `tests/test_security.py`.

## 6.1 Protocole de mesure

| Élément | Valeur |
|---|---|
| Matériel | Apple Silicon (arm64), 12 cœurs, 24 Go de mémoire |
| Système | macOS 15.6.1 |
| Logiciels | Python 3.13.5, scikit-learn 1.9.1, NumPy 2.5.5, pandas 3.1.1 |
| Chronomètre | `time.perf_counter` |
| Latence unitaire | 1 000 appels sur une transaction, après 50 appels de chauffe ; percentiles P50, P95 et P99 |
| Périmètre complet | `predict_fraud` : validation du schéma, prétraitement, modèle et seuil |
| Périmètre modèle seul | `model.predict_proba` sur une transaction, sans validation |
| Débit | `predict_fraud` sur des lots de 100, 1 000, 10 000 et 56 746 transactions ; médiane de 3 à 5 passages |
| Mémoire | Pic d'allocation mesuré par tracemalloc et mémoire résidente mesurée par psutil, pendant un lot de 56 746 transactions |
| Taille | Taille du fichier `fraud_model.joblib` |

Aucune API HTTP n'est déployée. La latence d'une requête réelle comprendrait en plus le réseau, la sérialisation JSON, l'authentification et la journalisation. Elle devra être mesurée séparément, par exemple avec Locust, lorsque l'API existera.

## 6.2 Résultats

| Mesure | Résultat | Critère | Statut |
|---|---|---|---|
| Latence unitaire P95, validation comprise | 1,43 ms (P50 1,25 ; P99 1,66 ; max 2,17) | ≤ 50 ms | conforme |
| Latence unitaire P95, modèle seul | 0,90 ms (P50 0,80 ; P99 1,01) | | |
| Débit, lot de 100 | 78 000 transactions/s | | |
| Débit, lot de 1 000 | 752 000 transactions/s | ≥ 1 000 transactions/s | conforme |
| Débit, lot de 10 000 | 4,7 millions de transactions/s | | |
| Débit, lot de 56 746 | 11 millions de transactions/s (5,1 ms par lot) | | |
| Taille du modèle | 4,6 Ko | ≤ 10 Mo | conforme |
| Surcoût mémoire (lot de 56 746) | 26 Mo | ≤ 200 Mo | conforme |

Sur GitHub Actions, la même mesure donne une latence P95 de 4,19 ms, toujours très en dessous du budget.

La validation des entrées représente environ 35 % de la latence unitaire (1,25 ms au total contre 0,80 ms pour le modèle seul). Ce coût est faible au regard du budget de 50 ms. Le débit dépend beaucoup de la taille des lots, car les appels à pandas et à scikit-learn ont un coût fixe qui domine pour les petits lots. Pour une API qui traite les transactions une par une, la mesure utile est la latence unitaire, soit environ 800 transactions par seconde et par cœur. Avec 30 coefficients, la régression logistique est très légère, et le coût d'exécution n'est pas un facteur limitant.

Justification des seuils. Une autorisation de paiement par carte prend quelques centaines de millisecondes de bout en bout, et le scoring de fraude n'en est qu'une étape ; 50 ms en représentent 10 à 20 %. Le jeu observé compte environ 142 000 transactions par jour, soit 1,6 par seconde en moyenne ; les pointes d'une banque de taille moyenne atteignent plusieurs centaines par seconde, d'où le seuil de 1 000 transactions par seconde. Les limites de 10 Mo et 200 Mo permettent un déploiement dans un conteneur léger avec plusieurs instances.

Les tests de `test_performance.py` vérifient les mêmes seuils sur un modèle de test. Ces seuils sont larges pour que les tests ne deviennent pas instables sur les machines partagées de GitHub Actions.

## 6.3 Sécurité opérationnelle

| Contrôle | Mise en œuvre | Preuve | Résultat |
|---|---|---|---|
| Champs manquants | `score_records` rejette la requête et indique les champs concernés | `test_score_records_rejects_malformed_records` | conforme |
| Champs supplémentaires | Rejet en mode strict (`card_number`, `ip`, etc.) | idem, `test_extra_column_rejected_in_strict_mode` | conforme |
| Types incorrects | Texte, booléen et `None` refusés, sans conversion implicite | idem | conforme |
| Requête mal formée | Liste vide, dictionnaire, texte, éléments qui ne sont pas des objets | `test_score_records_rejects_malformed_payload` | conforme |
| Taille des requêtes | 10 000 enregistrements au maximum | `test_score_records_limits_batch_size` | conforme |
| Fichiers reçus | Extension `.csv`, taille ≤ 500 Mo, erreurs de lecture converties en `DataLoadError` | `test_load_rejects_*` | conforme |
| Journaux | Seuls les volumes sont journalisés ; les erreurs ne citent que les noms de colonnes | `test_prediction_logs_volumes_without_transaction_values`, `test_rejected_request_does_not_log_transaction_values` | conforme |
| Secrets dans le dépôt | Recherche de clés AWS, clés privées, jetons GitHub, clé Kaggle, affectations de mots de passe, à chaque Pull Request | `test_repository_contains_no_secret` ; détection vérifiée sur 5 secrets factices | conforme |
| Fichiers sensibles | `.gitignore` : `data/*.csv`, `.env`, `kaggle.json`, `*.joblib`, `.venv/` | `test_sensitive_files_are_git_ignored` | conforme |
| Intégrité du modèle | Empreinte SHA-256 vérifiée avant `joblib.load` ; extension, métadonnées et type d'objet contrôlés | `test_tampered_model_is_refused` et 5 autres tests | conforme |
| Analyse statique | Règles de sécurité Bandit (`S`) via Ruff | `ruff check` en CI | conforme |
| Droits de la CI | `permissions: contents: read` | `.github/workflows/tests.yml` | conforme |

Le scanner de secrets a signalé un vrai problème en cours de projet : les noms des tests paramétrés reprenaient le contenu des secrets factices, qui se retrouvaient dans le rapport d'exécution des tests. Les tests sont maintenant nommés par type de secret.

## 6.4 Anomalies et limites

| ID | Constat | Gravité | Recommandation |
|---|---|---|---|
| ANO-08 | `joblib` repose sur `pickle`, et un fichier modifié peut exécuter du code au chargement. L'empreinte est vérifiée, mais elle est stockée à côté du modèle : une personne qui contrôle le dossier peut modifier les deux. | Moyenne | Signer le modèle avec une clé stockée ailleurs, ou utiliser un registre de modèles ; envisager un format sans exécution de code (ONNX, skops) |
| ANO-09 | Pas d'API déployée : latence HTTP, authentification, limitation de débit et tenue en charge non mesurées | Moyenne | Tests de charge avec Locust avant le pilote |
| | Le scanner fonctionne par motifs et ne détecte pas les secrets de forme inconnue | Mineure | Ajouter gitleaks ou la détection de secrets de GitHub |

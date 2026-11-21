# 6. Performance et sécurité opérationnelle

> Source : `reports/results/performance.json` (`python -m app.cli performance`). Tests : `tests/test_performance.py`, `tests/test_security.py`.

## 6.1 Protocole de mesure

| Élément | Valeur |
|---|---|
| Matériel | Apple Silicon (arm64), 12 cœurs, 24 Go de mémoire vive |
| Système | macOS 15.6.1 |
| Logiciel | Python 3.13.5, scikit-learn 1.9.1, NumPy 2.5.5, pandas 3.1.1 |
| Chronomètre | `time.perf_counter` (horloge monotone haute résolution) |
| Latence unitaire | 1 000 appels sur 1 transaction, après 50 appels de chauffe ; percentiles P50, P95, P99 |
| Périmètre « complet » | `predict_fraud` : validation du schéma, prétraitement, modèle et seuil |
| Périmètre « modèle seul » | `model.predict_proba` sur 1 transaction, sans validation |
| Débit | `predict_fraud` sur des lots de 100, 1 000, 10 000 et 56 746 transactions ; médiane de 3 à 5 passes |
| Mémoire | Pic d'allocations (tracemalloc) et mémoire résidente du processus (psutil) pendant un lot de 56 746 transactions |
| Taille | Taille du fichier `fraud_model.joblib` |

**Limite :** aucune API HTTP n'est déployée. La latence d'une requête réelle ajouterait le réseau, la sérialisation JSON, l'authentification et la journalisation, et doit être mesurée séparément (par exemple avec Locust) une fois l'API en place.

## 6.2 Résultats

| Mesure | Résultat | Critère | Statut |
|---|---|---|---|
| Latence unitaire P95 (validation incluse) | **1,43 ms** (P50 1,25 ; P99 1,66 ; max 2,17) | ≤ 50 ms | OK |
| Latence unitaire P95 (modèle seul) | 0,90 ms (P50 0,80 ; P99 1,01) | — | — |
| Débit, lot de 100 | 78 000 transactions/s | — | — |
| Débit, lot de 1 000 | 752 000 transactions/s | ≥ 1 000 transactions/s | OK |
| Débit, lot de 10 000 | 4,7 millions de transactions/s | — | — |
| Débit, lot de 56 746 | 11 millions de transactions/s (5,1 ms par lot) | — | — |
| Taille du modèle | **4,6 Ko** | ≤ 10 Mo | OK |
| Surcoût mémoire (lot de 56 746) | **26 Mo** (pic tracemalloc) | ≤ 200 Mo | OK |

### Interprétation

- **La validation des entrées représente environ 35 % de la latence unitaire** : 1,25 ms au total, contre 0,80 ms pour le modèle seul. C'est un coût assumé, qui reste très faible au regard du budget de 50 ms.
- Le débit dépend fortement de la taille du lot, car le coût fixe des appels pandas et scikit-learn domine pour les petits lots. Pour une API transaction par transaction, la mesure pertinente est la **latence unitaire**, environ 800 transactions/s par cœur. Pour un traitement par lots, c'est le débit par lot.
- La régression logistique (30 coefficients) est très légère. Le coût d'exécution **n'est pas un facteur limitant**.

### Justification des seuils

- **50 ms (P95)** : une autorisation de paiement par carte répond en quelques centaines de millisecondes de bout en bout. Le scoring de fraude n'en est qu'une étape parmi d'autres (réseau, émetteur, règles), d'où une part d'environ 10 à 20 %.
- **1 000 transactions/s** : le jeu observé compte environ 142 000 transactions par jour, soit environ 1,6 par seconde en moyenne. Une banque de taille moyenne connaît des pointes de plusieurs centaines par seconde. 1 000 par seconde laisse une marge confortable.
- **10 Mo et 200 Mo** : permettent un déploiement dans un conteneur léger, avec plusieurs répliques.

Les tests automatisés (`test_performance.py`) contrôlent les mêmes seuils sur le modèle de test. Ces seuils restent volontairement larges pour ne pas échouer de manière aléatoire sur les machines partagées de GitHub Actions.

## 6.3 Sécurité opérationnelle

| Contrôle | Mise en œuvre | Preuve | Résultat |
|---|---|---|---|
| Champs manquants | `score_records` rejette toute la requête et nomme les champs | `test_score_records_rejects_malformed_records` | OK |
| Champs supplémentaires | Rejet en mode strict (`card_number`, `ip`…) | idem ; `test_extra_column_rejected_in_strict_mode` | OK |
| Types incorrects | Texte, booléen, `None` refusés (pas de conversion implicite) | idem | OK |
| Requête mal formée | Liste vide, dictionnaire, texte, éléments non-objets refusés | `test_score_records_rejects_malformed_payload` | OK |
| Taille des requêtes | 10 000 enregistrements au maximum par appel | `test_score_records_limits_batch_size` | OK |
| Fichiers reçus | Extension `.csv` obligatoire, taille ≤ 500 Mo, erreurs de lecture transformées en `DataLoadError` | `test_load_rejects_*` | OK |
| Données sensibles dans les journaux | Les journaux ne contiennent que des volumes ; les messages d'erreur ne contiennent que des noms de colonnes | `test_prediction_logs_volumes_without_transaction_values`, `test_rejected_request_does_not_log_transaction_values`, `test_error_message_does_not_leak_values` | OK |
| Secrets dans le dépôt | Scanner (clés AWS, clés privées, jetons GitHub, clé Kaggle, `password = "…"`), exécuté sur tout le dépôt à chaque Pull Request | `test_repository_contains_no_secret` ; détection prouvée sur 5 secrets factices | OK, aucun secret |
| Fichiers sensibles | `.gitignore` : `data/*.csv`, `.env`, `kaggle.json`, `*.joblib`, `.venv/` | `test_sensitive_files_are_git_ignored` | OK |
| Intégrité du modèle | Empreinte SHA-256 vérifiée **avant** `joblib.load` ; extension, métadonnées et type d'objet contrôlés | `test_tampered_model_is_refused` et 5 autres | OK |
| Analyse statique | Règles de sécurité Bandit (`S`) de Ruff | `ruff check` en CI | OK |
| Permissions CI | `permissions: contents: read` dans le workflow | `.github/workflows/tests.yml` | OK |

## 6.4 Anomalies et limites

| ID | Constat | Gravité | Recommandation |
|---|---|---|---|
| ANO-08 | `joblib` repose sur `pickle` : un fichier modifié peut exécuter du code au chargement. L'empreinte est vérifiée, mais elle est stockée à côté du modèle : un attaquant qui contrôle le dossier peut modifier les deux. | Moyenne | Signer le modèle (clé hors du dossier), ou utiliser un registre de modèles avec contrôle d'accès ; envisager un format sans exécution de code (ONNX, skops) |
| ANO-09 | Pas d'API déployée : latence HTTP, authentification, limitation de débit et tests de charge non mesurés | Moyenne | Tests de charge Locust avant le pilote ; mesurer la latence de bout en bout |
| — | Le scanner de secrets repose sur des motifs : il ne détecte pas les secrets de forme inconnue | Mineure | Ajouter `gitleaks` ou la détection de secrets de GitHub sur le dépôt distant |

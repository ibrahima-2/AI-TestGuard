# 8. Rapport final de qualification

| | |
|---|---|
| Système | Détection de fraude par carte bancaire, pipeline scikit-learn avec régression logistique |
| Version qualifiée | v1.0.0 : régression logistique, C = 0,01, `class_weight="balanced"`, 30 variables, seuil 0,999992 |
| Données | Credit Card Fraud Detection (Kaggle, ULB et Worldline), septembre 2013, 284 807 transactions |
| Preuves | `reports/results/`, `reports/figures/`, `reports/baseline/`, `reports/tests/`, `tests/` |
| Décision | La v1.0.0 n'est pas qualifiée pour un pilote avec blocage automatique. Un pilote en mode observation est proposé pour la v1.1.0, sous conditions (section 10). |

## 1. Contexte et périmètre

La banque souhaite savoir si son modèle de détection de fraude peut passer en phase pilote. Deux types d'erreurs sont à considérer : les faux positifs, qui bloquent des clients légitimes, et les faux négatifs, qui laissent passer des fraudes.

L'évaluation porte sur les données, le prétraitement, le modèle, la prédiction, l'intégration (sauvegarde, rechargement, ligne de commande) et l'automatisation des tests. Une API HTTP, les données de production actuelles et les attributs démographiques des clients sont hors périmètre (voir `01_plan_de_test.md`).

## 2. Plan de test et traçabilité

Le plan de test définit 25 exigences de test (T01 à T25). Chacune est reliée à un risque et à un test automatisé ou à un fichier de résultats (`01_plan_de_test.md`, section 1.4). Les critères d'acceptation ont été fixés dans `app/config.py` avant le premier entraînement :

| Critère | Seuil |
|---|---|
| Rappel | ≥ 0,75 |
| Précision | ≥ 0,30 |
| FPR | ≤ 0,5 % |
| PR-AUC | ≥ 0,60 |
| ROC-AUC | ≥ 0,90 |
| Écart de PR-AUC entre entraînement et test | ≤ 0,10 |
| Latence P95 | ≤ 50 ms |
| Débit | ≥ 1 000 transactions/s |
| Taille du modèle | ≤ 10 Mo |
| Surcoût mémoire | ≤ 200 Mo |

## 3. Architecture et choix méthodologiques

Le pipeline enchaîne les étapes suivantes :

1. chargement du fichier CSV (`load_data`), avec contrôle de l'extension, de la taille et de la lisibilité ;
2. validation du schéma, des types et des valeurs (`validate_data`) ;
3. suppression des 1 081 doublons exacts (`remove_duplicates`) ;
4. découpage stratifié 60 / 20 / 20 avec la graine 42 (`split_data`) ;
5. entraînement d'un pipeline scikit-learn, composé d'un `ColumnTransformer` (`log1p` et standardisation pour `Amount`, standardisation pour les autres variables) et du modèle, sur l'entraînement uniquement ;
6. choix des hyperparamètres et du seuil sur la validation ;
7. sauvegarde du modèle avec son empreinte SHA-256 (`save_model`), puis rechargement après vérification de l'empreinte (`load_model`) ;
8. évaluation unique sur le test.

| Choix | Raison |
|---|---|
| Rejet des données invalides, sans imputation | Compléter une variable d'une transaction à scorer pourrait masquer une fraude |
| Dédoublonnage avant le découpage | Éviter qu'une même transaction se retrouve dans deux partitions |
| `log1p` sur `Amount` | Distribution très asymétrique ; meilleure tenue face aux montants extrêmes |
| `class_weight="balanced"` | La classe positive ne représente que 0,17 % des transactions ; sans pondération, la précision baisse de 11 points |
| Seuil réglé sur la validation | Avec le seuil par défaut de 0,5, on obtiendrait 1 479 fausses alertes pour 85 fraudes détectées |
| PR-AUC comme métrique de sélection | L'accuracy et la ROC-AUC sont peu informatives sur une classe rare |
| Bootstrap stratifié | Le test ne contient que 95 fraudes ; il faut quantifier l'incertitude |

## 4. Tests unitaires et d'intégration

| Fichier | Partie de l'énoncé | Nombre de tests | Résultat |
|---|---|---|---|
| `test_preprocessing.py` | II | 55 | réussis |
| `test_prediction.py` | II, III | 50 | réussis |
| `test_integration.py` (dont 2 tests lents) | IV | 19 | réussis |
| `test_robustness.py` | V, X | 30 | réussis |
| `test_regression.py` (dont 1 test lent) | VI | 16 | réussis |
| `test_performance.py` | VIII | 7 | réussis |
| `test_security.py` | VIII | 13 | réussis |
| Total | | 190 | 190 réussis |

Le détail se trouve dans `reports/tests/pytest_rapport.txt` et `reports/tests/junit.xml`. La couverture de code est de 98 % en local et de 97,75 % sur GitHub Actions, où le workflow échoue en dessous de 90 %. Ruff ne signale aucune erreur.

Les tests vérifient des résultats précis plutôt que la simple absence d'exception : le message d'erreur doit nommer la colonne en cause, le nombre de lignes invalides détectées doit être exact, les métriques sont comparées à des valeurs calculées à la main, et les prédictions doivent être identiques après rechargement du modèle.

Sur GitHub Actions, le push sur `main` passe le lint et les tests rapides. La Pull Request de démonstration (PR #1), qui remplace `np.log1p` par `np.log`, fait échouer le job avec 7 tests en échec et 10 en erreur, et la protection de `main` empêche sa fusion. L'évaluation complète, lancée manuellement, télécharge le jeu Kaggle, vérifie son empreinte et passe les tests lents.

## 5. Performance prédictive

Résultats sur le jeu de test (56 746 transactions, 95 fraudes) ; détail dans `02_evaluation_modele.md`.

| Métrique | Régression logistique | IC 95 % | Arbre de décision | Référence naïve |
|---|---|---|---|---|
| Accuracy | 0,9994 | | 0,9908 | 0,9983 |
| Précision | 0,875 | [0,80 ; 0,94] | 0,130 | 0 |
| Rappel | 0,737 | [0,65 ; 0,83] | 0,789 | 0 |
| F1 | 0,800 | [0,73 ; 0,86] | 0,223 | 0 |
| FPR | 0,018 % | [0,009 % ; 0,030 %] | 0,89 % | 0 |
| ROC-AUC | 0,961 | | 0,893 | 0,5 |
| PR-AUC | 0,681 | [0,57 ; 0,79] | 0,628 | 0,0017 |
| TP / FP / FN | 70 / 10 / 25 | | 75 / 503 / 20 | 0 / 0 / 95 |

La régression logistique respecte cinq critères sur six. Le rappel n'atteint pas 0,75 : il manque deux fraudes sur 95. L'intervalle de confiance contient le seuil, ce qui signifie que le test ne permet pas de trancher. Les fraudes manquées sont surtout de petits montants (médiane de 3,76) ; en valeur, 72 % du montant frauduleux est intercepté. L'arbre de décision est rejeté : il produit 503 fausses alertes et surapprend (écart de PR-AUC de 0,13).

## 6. Robustesse et non-régression

Les huit scénarios d'entrées invalides sont rejetés, et le nombre de lignes invalides détectées correspond exactement au nombre de lignes modifiées. Un bruit faible (jusqu'à 10 % de l'écart-type sur les composantes, ±5 % sur le montant, ±60 s sur le temps) fait changer de classe au plus une transaction sur 56 746, sans effet sur le rappel.

Deux résultats sont moins favorables. Un décalage de 48 heures de la variable `Time` fait perdre 4 fraudes détectées, alors qu'en production toutes les valeurs de `Time` seraient hors de la plage d'entraînement. Par ailleurs, en validation temporelle (entraînement sur le jour 1, évaluation sur le jour 2 qui compte 201 fraudes), la précision tombe entre 0,61 et 0,69, contre 0,875 avec le découpage aléatoire, et le rappel se situe entre 0,74 et 0,78.

En non-régression, le réentraînement à l'identique redonne exactement la référence, en local comme sur GitHub Actions. La version sans `Time` est acceptée. La version sans pondération des classes est rejetée, sa précision baissant de 10,7 points (voir `03_robustesse.md` et `04_non_regression.md`).

## 7. Latence et ressources

| Mesure | Résultat | Critère |
|---|---|---|
| Latence P95, validation comprise | 1,43 ms (3,50 ms sur GitHub Actions) | ≤ 50 ms |
| Débit, lots de 1 000 | 752 000 transactions/s | ≥ 1 000 |
| Taille du modèle | 4,6 Ko | ≤ 10 Mo |
| Surcoût mémoire | 26 Mo | ≤ 200 Mo |

Les contrôles de sécurité opérationnelle sont tous satisfaits : contrat d'API strict, journaux sans valeurs de transaction, absence de secret dans le dépôt, refus d'un modèle modifié (`06_performance_securite.md`).

## 8. Anomalies

| ID | Anomalie | Gravité | Statut |
|---|---|---|---|
| ANO-01 | Rappel sur le test de 0,737, pour un critère de 0,75 | Majeure | Ouverte |
| ANO-02 | Seuil (0,999992) placé là où le rappel commence à baisser ; rappel de validation plus favorable que sur l'entraînement (0,819 contre 0,746 au même seuil) | Majeure | Ouverte, nouvelle règle de seuil proposée en section 10 |
| ANO-03 | Dépendance à `Time`, variable qui croît sans limite | Moyenne | Version v1.1.0 acceptée en non-régression, à confirmer |
| ANO-04 | 1 081 doublons exacts pouvant créer une fuite entre partitions | Majeure | Corrigée (dédoublonnage et tests) |
| ANO-05 | Montants extrêmes : scores relevés jusqu'à +0,44 | Mineure | Acceptée, alerte hors plage |
| ANO-06 | Élargissement des composantes : FPR multiplié par 2,3 sans hausse du taux d'alertes | Moyenne | Couverte par le PSI |
| ANO-07 | Arbre de décision : surapprentissage et précision de 0,13 | Majeure | Close, modèle rejeté |
| ANO-08 | Chargement par pickle ; empreinte stockée à côté du modèle | Moyenne | Atténuée, signature recommandée |
| ANO-09 | Pas d'API : latence de bout en bout et tenue en charge non mesurées | Moyenne | Ouverte |
| ANO-10 | Précision en validation temporelle (0,61 à 0,69) inférieure à celle du découpage aléatoire (0,875) | Moyenne | Ouverte |
| ANO-11 | Équité non évaluable faute d'attributs clients | Bloquante pour la production | Ouverte |

## 9. Limites et risques résiduels

Les données datent de 2013 et couvrent deux jours de transactions de porteurs européens. Les fraudes actuelles (paiement mobile, ingénierie sociale, virements instantanés) n'y sont pas représentées, et les résultats ne prouvent rien sur les transactions d'aujourd'hui.

Les composantes PCA empêchent d'expliquer les décisions en termes métier, de construire des perturbations réalistes et d'évaluer l'équité. Un système réel utiliserait les variables d'origine et devrait être qualifié à nouveau.

Chaque partition ne contient que 94 ou 95 fraudes, ce qui donne des intervalles de confiance d'environ ±9 points sur le rappel. Les métriques varient d'ailleurs de 0,11 de PR-AUC d'une partition à l'autre.

Le jeu de test a été utilisé une fois. Toute correction (nouveau seuil, version sans `Time`) doit être vérifiée sur des données nouvelles ; la réévaluer sur ce test en ferait un jeu de réglage.

Les critères reposent sur des hypothèses raisonnables, mais la banque n'a pas fourni de coûts réels pour une fraude manquée ou une fausse alerte. Aucun test adverse n'a été mené : un fraudeur pourrait exploiter la faiblesse observée sur les petits montants.

Le principal risque résiduel est qu'environ une fraude sur quatre n'est pas détectée, soit une soixantaine par jour à l'échelle du jeu observé. Les fausses alertes sont rares (environ 25 par jour), mais elles portent sur des montants élevés.

## 10. Décision

Le système satisfait les exigences techniques testées : contrat de données, prévention des fuites, robustesse aux entrées invalides et au bruit, reproductibilité, non-régression, coût d'exécution, sécurité opérationnelle et intégration continue. En revanche, il ne satisfait pas le critère de rappel sur des données indépendantes (ANO-01), son seuil est fragile (ANO-02), et la validation temporelle donne également un rappel proche de la limite.

La version v1.0.0 n'est donc pas qualifiée pour un pilote dans lequel le modèle bloquerait seul des transactions. Un pilote en mode observation est proposé pour la version v1.1.0 : le modèle score les transactions en parallèle du dispositif existant, ses alertes sont transmises aux analystes, et aucune transaction n'est bloquée sur sa seule décision.

Avant ce pilote, trois corrections sont prévues, sans réutiliser le jeu de test :

1. passer à la version v1.1.0, sans `Time`, déjà acceptée en non-régression ;
2. choisir le seuil par validation croisée stratifiée (5 plis) sur l'entraînement et la validation réunis, en exigeant un rappel d'au moins 0,80 sur la médiane des plis plutôt que sur une seule partition de 94 fraudes ;
3. recréer la référence de non-régression, par une décision explicite.

Le fait de réussir les tests techniques ne vaut pas autorisation de déploiement. Celle-ci suppose aussi une validation par les équipes métier, d'exploitation, de sécurité et de conformité.

## 11. Question de synthèse

*Dans quelles conditions votre modèle peut-il être considéré comme suffisamment fiable pour être testé dans un environnement pilote, et quelles preuves supplémentaires seraient nécessaires avant une mise en production réelle ?*

### Conditions pour un pilote

Au vu des résultats, le modèle peut être testé en pilote à condition de ne pas lui confier seul la décision de bloquer une transaction. Son rappel mesuré (0,737, avec un intervalle de confiance de 0,65 à 0,83) n'atteint pas l'exigence de 0,75, sa précision descend à environ 0,65 en validation temporelle, et ses fausses alertes concernent des montants élevés. Il peut servir à produire des alertes vérifiées par un analyste, mais pas à décider automatiquement.

Les conditions proposées sont les suivantes :

1. Le modèle fonctionne sans effet direct sur les clients. Ses scores sont calculés en parallèle des règles existantes, qui restent seules à décider, et ses alertes sont examinées par un analyste. La charge prévue, environ 200 alertes par jour pour 142 000 transactions, doit être validée avec l'équipe concernée.
2. La version utilisée est la v1.1.0, sans `Time`, avec un seuil fixé par validation croisée. Les contrôles automatiques (CI, contrat de données, empreinte du modèle, latence) doivent être satisfaits.
3. Les transactions rejetées par la validation des entrées sont traitées par les règles existantes.
4. La surveillance est en place dès le début : PSI par variable, calculé sur des fenêtres comparables, taux d'alertes, taux de rejet, latence, et procédure de retour arrière. Les simulations ont montré que le taux d'alertes seul ne suffit pas à détecter une dérive.
5. Le périmètre et les critères de fin sont fixés à l'avance : durée et part du trafic limitées, fin du pilote après au moins 250 fraudes labellisées (ce qui permet d'estimer le rappel à environ ±5 points), et critères de succès identiques à ceux d'ENF1, mesurés sur ces nouvelles données.

### Preuves supplémentaires avant une mise en production

1. Des performances mesurées sur des données récentes et indépendantes, issues du pilote : rappel ≥ 0,75, précision ≥ 0,30 et FPR ≤ 0,5 %, sur au moins 250 fraudes.
2. Une comparaison avec le dispositif existant sur les mêmes transactions : fraudes détectées, fausses alertes, montant intercepté.
3. Un seuil justifié par les coûts réels, validés par la banque : coût moyen d'une fraude manquée et d'une fausse alerte.
4. Un audit d'équité sur un jeu enrichi et licite (âge, zone, canal).
5. Une qualification sur les variables réellement utilisées en production, avec des explications compréhensibles par les analystes et communicables aux clients.
6. Des tests de l'API réelle : latence de bout en bout, tenue en charge aux pointes (Locust), résilience, authentification, test d'intrusion, signature du modèle.
7. Des tests adverses, en priorité sur les petits montants.
8. Une surveillance de production opérationnelle (par exemple Prometheus et Grafana), la collecte des labels différés et une procédure de réentraînement et de requalification.
9. Les validations non techniques : niveau de risque accepté par le métier, capacité des analystes et procédure en cas de blocage, revue de sécurité, et conformité réglementaire à vérifier avec le délégué à la protection des données (RGPD, décisions automatisées) et au regard des exigences des autorités bancaires.

Les tests réalisés montrent que le pipeline est solide sur le plan technique et que le modèle sépare bien les classes. Ils ne montrent pas que le niveau de détection exigé est atteint, et ils reposent sur des données qui ne représentent pas la fraude actuelle. Le pilote doit permettre de réunir ces éléments sans exposer les clients aux erreurs du modèle.

# 8. Rapport final de qualification — AI-TestGuard v1.0.0

| | |
|---|---|
| Système | Détection de fraude par carte bancaire : pipeline scikit-learn avec régression logistique |
| Version qualifiée | v1.0.0 : régression logistique, C = 0,01, `class_weight="balanced"`, 30 variables, seuil 0,999992 |
| Données | Credit Card Fraud Detection (Kaggle, ULB / Worldline), septembre 2013, 284 807 transactions |
| Preuves | `reports/results/*.json`, `reports/figures/`, `reports/baseline/`, `reports/tests/`, `tests/` |
| **Décision** | **v1.0.0 non qualifiée pour un pilote avec blocage automatique. Pilote en mode observation recommandé pour la v1.1.0, sous conditions (§ 10).** |

---

## 1. Contexte et périmètre

La banque veut savoir si un modèle de détection de fraude peut poursuivre vers une phase pilote. Deux types d'erreurs sont en jeu :
- les **faux positifs** : des clients légitimes sont bloqués, ce qui génère des réclamations ;
- les **faux négatifs** : des fraudes ne sont pas détectées, ce qui entraîne des pertes.

L'évaluation couvre les données, le prétraitement, le modèle, la prédiction, l'intégration (sauvegarde, rechargement, ligne de commande) et l'automatisation (CI).

Hors périmètre : une API HTTP déployée, les données de production actuelles, les attributs démographiques des clients. Le détail est dans `01_plan_de_test.md`, § 1.2.

## 2. Plan de test et traçabilité

25 exigences de test (T01 à T25) relient chaque exigence fonctionnelle ou non fonctionnelle à un risque et à une preuve automatisée (`01_plan_de_test.md`, § 1.4). Les critères d'acceptation ont été **fixés dans `app/config.py` avant tout entraînement** et n'ont pas été modifiés depuis.

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

```
creditcard.csv ─► load_data ─► validate_data ─► remove_duplicates ─► split_data (60/20/20 stratifié, graine 42)
                    │ contrôle extension,   │ schéma, types,        │ 1 081 doublons        │
                    │ taille, lisibilité    │ NaN/inf, bornes       │ supprimés             ▼
                                                                         ┌──────── entraînement ───────┐
 Pipeline scikit-learn : ColumnTransformer [log1p+StandardScaler(Amount), StandardScaler(autres)] ─► modèle
                                                                         └─ réglage + seuil : validation ┘
                         save_model (joblib + SHA-256) ─► load_model (vérifie l'empreinte) ─► test (une fois)
```

| Choix | Raison |
|---|---|
| Rejet des données invalides, sans imputation | Compléter une variable d'une transaction à scorer pourrait masquer une fraude |
| Dédoublonnage *avant* le découpage | Empêcher une même transaction d'apparaître dans deux partitions (fuite) |
| `log1p` sur `Amount` | Distribution très asymétrique ; robustesse aux montants extrêmes |
| `class_weight="balanced"` | Classe positive à 0,17 % ; la non-régression montre que la retirer coûte 11 points de précision |
| Seuil réglé sur la validation | Le seuil de 0,5 donnerait 1 479 fausses alertes pour 85 fraudes détectées |
| PR-AUC comme métrique de sélection | La ROC-AUC et l'accuracy sont trompeuses sur une classe rare |
| Bootstrap stratifié | Quantifier l'incertitude avec seulement 95 fraudes dans le test |

## 4. Tests unitaires et d'intégration

| Suite | Tests | Résultat |
|---|---|---|
| `test_preprocessing.py` (Partie II) | 55 | réussis |
| `test_prediction.py` (Parties II et III) | 50 | réussis |
| `test_integration.py` (Partie IV), dont 2 lents | 19 | réussis |
| `test_robustness.py` (Parties V et X) | 30 | réussis |
| `test_regression.py` (Partie VI), dont 1 lent | 16 | réussis |
| `test_performance.py` (Partie VIII) | 7 | réussis |
| `test_security.py` (Partie VIII) | 13 | réussis |
| **Total** | **190** | **190 réussis, 0 échec** |

Les détails sont dans `reports/tests/pytest_rapport.txt` et `reports/tests/junit.xml`.

- **Couverture de code : 98 %** (lignes et branches de `app/`).
- **Ruff : aucune erreur.**
- Les tests vérifient des comportements précis : message nommant la colonne fautive, nombre exact de lignes invalides, valeurs exactes des métriques sur des cas calculés à la main, prédictions identiques après rechargement. L'absence d'exception ne suffit pas à faire passer un test.

## 5. Performance prédictive (`02_evaluation_modele.md`)

| Jeu de test (56 746 transactions, 95 fraudes) | Régression logistique | IC 95 % | Arbre de décision | Référence naïve |
|---|---|---|---|---|
| Accuracy | 0,9994 | — | 0,9908 | 0,9983 |
| Précision | **0,875** | [0,80 ; 0,94] | 0,130 | 0 |
| Rappel | **0,737** ✗ | [0,65 ; 0,83] | 0,789 | 0 |
| F1 | 0,800 | [0,73 ; 0,86] | 0,223 | 0 |
| FPR | 0,018 % | [0,009 ; 0,030 %] | 0,89 % | 0 |
| ROC-AUC | 0,961 | — | 0,893 | 0,5 |
| PR-AUC | 0,681 | [0,57 ; 0,79] | 0,628 | 0,0017 |
| TP / FP / FN | 70 / 10 / 25 | | 75 / 503 / 20 | 0 / 0 / 95 |

- **5 critères sur 6 sont respectés.** Le **rappel échoue de justesse** : il manque 2 fraudes sur 95 pour atteindre 0,75. L'intervalle de confiance contient le seuil : le test ne permet pas de trancher.
- Les fraudes manquées sont surtout de petits montants (médiane 3,76). Le modèle intercepte 72 % de la valeur frauduleuse.
- L'arbre de décision est rejeté : 503 fausses alertes et un surapprentissage (écart de PR-AUC de 0,13).

## 6. Robustesse et non-régression (`03_robustesse.md`, `04_non_regression.md`)

- **8 scénarios d'entrées invalides sur 8 sont rejetés**, et le nombre de lignes invalides détectées est exact.
- **Bruit faible** (jusqu'à 10 % de σ sur les composantes, ±5 % sur le montant, ±60 s sur le temps) : au plus 1 transaction sur 56 746 change de classe, sans effet sur le rappel.
- **Dépendance à `Time`** : décaler de 48 h fait perdre 4 fraudes détectées. `Time` est 100 % hors plage en production.
- **Validation temporelle** (entraînement sur le jour 1, évaluation sur le jour 2, 201 fraudes) : rappel de 0,74 à 0,78 ; **précision de 0,61 à 0,69**, contre 0,875 avec un découpage aléatoire.
- **Non-régression** :
  - réentraînement identique : **reproductible au bit près** ;
  - version sans `Time` : **acceptée** ;
  - version sans pondération des classes : **rejetée** (précision −10,7 points).

## 7. Latence et ressources (`06_performance_securite.md`)

| Mesure | Résultat | Critère |
|---|---|---|
| Latence P95, validation incluse | 1,43 ms | ≤ 50 ms ✓ |
| Débit, lots de 1 000 | 752 000 transactions/s | ≥ 1 000 ✓ |
| Taille du modèle | 4,6 Ko | ≤ 10 Mo ✓ |
| Surcoût mémoire | 26 Mo | ≤ 200 Mo ✓ |

Les contrôles de sécurité opérationnelle passent tous : contrat d'API, absence de valeurs dans les journaux, aucun secret dans le dépôt, modèle altéré refusé.

## 8. Anomalies

| ID | Anomalie | Gravité | Statut |
|---|---|---|---|
| ANO-01 | Rappel sur le test de 0,737, inférieur au critère de 0,75 | **Majeure** | Ouverte |
| ANO-02 | Seuil fragile (0,999992), placé au début de la chute du rappel ; validation optimiste (rappel de 0,819 contre 0,746 sur l'entraînement au même seuil) | **Majeure** | Ouverte : nouvelle règle de seuil proposée (§ 10) |
| ANO-03 | Dépendance à `Time`, variable non stationnaire | Moyenne | Correctif v1.1.0 accepté en non-régression, à confirmer sur des données nouvelles |
| ANO-04 | 1 081 doublons exacts : risque de fuite entre partitions | Majeure | **Corrigée** : dédoublonnage avant découpage, tests T10 |
| ANO-05 | Montants extrêmes : scores relevés jusqu'à +0,44 | Mineure | Acceptée ; alerte « hors plage » |
| ANO-06 | Élargissement des composantes : FPR × 2,3, invisible dans le taux d'alertes | Moyenne | Couverte par la surveillance PSI |
| ANO-07 | Arbre de décision : surapprentissage et précision de 0,13 | Majeure | Close : modèle rejeté |
| ANO-08 | Chargement `pickle` : l'empreinte est stockée à côté du modèle | Moyenne | Atténuée ; signature recommandée |
| ANO-09 | Pas d'API déployée : latence de bout en bout et tests de charge non mesurés | Moyenne | Ouverte |
| ANO-10 | Précision en conditions temporelles (0,61 à 0,69) nettement inférieure à celle du découpage aléatoire (0,875) | Moyenne | Ouverte : le découpage aléatoire surestime la qualité |
| ANO-11 | Équité impossible à évaluer : aucun attribut client | **Bloquante pour la production** | Ouverte : données d'audit nécessaires |

## 9. Limites de l'évaluation et risques résiduels

1. **Données anciennes et particulières.**
   - Les données couvrent deux jours de septembre 2013, chez des porteurs européens. Les fraudes de 2026 (paiement mobile, ingénierie sociale, paiements instantanés) ne sont pas représentées.
   - Les résultats ne prouvent pas la performance sur les transactions actuelles.
2. **Variables anonymisées.**
   - Les composantes PCA empêchent toute explication métier des décisions, toute perturbation réaliste et toute analyse d'équité.
   - Un système réel utiliserait les variables d'origine. Le pipeline devrait alors être requalifié.
3. **Effectifs faibles.**
   - 94 ou 95 fraudes par partition donnent des intervalles de confiance de ±9 points sur le rappel.
   - Les métriques varient de 0,11 de PR-AUC entre partitions, sans surapprentissage.
4. **Le jeu de test a été consulté une fois.**
   - Toute correction (seuil, version sans `Time`) doit être confirmée sur des données **nouvelles**.
   - Réutiliser ce test pour valider la correction le transformerait en jeu de réglage.
5. **Pas de coût métier réel.**
   - Les critères reposent sur des hypothèses raisonnées, faute d'un coût unitaire validé par la banque pour les fraudes manquées et les fausses alertes.
6. **Pas de test adverse.**
   - Un fraudeur qui connaît le modèle pourrait chercher à passer sous le seuil, par exemple avec de petits montants, la zone de faiblesse identifiée.
7. **Risque résiduel principal.**
   - Environ 1 fraude sur 4 n'est pas détectée (environ 60 par jour à l'échelle du jeu observé).
   - Les fausses alertes, bien que rares (environ 25 par jour), portent sur des montants élevés.

## 10. Décision

### Réussite aux tests techniques

Le système **réussit les tests techniques** :
- contrat de données ;
- prévention des fuites ;
- robustesse aux entrées invalides et au bruit ;
- reproductibilité ;
- non-régression ;
- performance d'exécution ;
- sécurité opérationnelle ;
- automatisation en CI.

### Échec au critère de rappel

Il **échoue au critère de rappel** sur des données indépendantes (ANO-01). Son seuil est fragile (ANO-02), et la validation temporelle confirme un rappel à la limite.

### Décision

> **La v1.0.0 n'est pas qualifiée pour un pilote dans lequel le modèle déciderait seul (blocage automatique).**
>
> **Un pilote en mode observation est recommandé** pour la v1.1.0, sous les conditions ci-dessous. Dans ce mode, le modèle note les transactions en parallèle du dispositif existant : ses alertes sont transmises aux analystes, et aucune transaction n'est bloquée sur sa seule décision.

**Actions correctives avant le pilote (sans réutiliser le jeu de test) :**
1. Passer à la version **v1.1.0 sans `Time`** : déjà acceptée en non-régression.
2. Remplacer la règle de seuil par une **règle robuste**. Le seuil serait choisi par validation croisée stratifiée (5 plis) sur l'entraînement et la validation réunis. Il devrait garantir un rappel ≥ 0,80 sur la *médiane* des plis, plutôt que sur une seule partition de 94 fraudes.
3. Mettre à jour la référence de non-régression (`baseline-create --force`, décision explicite et tracée).

Réussir les tests techniques ne vaut pas **autorisation de déploiement**. Celle-ci exige aussi des validations métier, opérationnelle, de sécurité et réglementaire (§ 11).

---

## 11. Question de synthèse

> *Dans quelles conditions votre modèle peut-il être considéré comme suffisamment fiable pour être testé dans un environnement pilote, et quelles preuves supplémentaires seraient nécessaires avant une mise en production réelle ?*

### Conditions pour un pilote

D'après nos résultats, le modèle est assez fiable pour un pilote **à condition que ce pilote ne lui confie pas seul la décision de bloquer une transaction**. Cette condition découle directement de trois constats :
- son rappel démontré (0,737, IC 95 % de 0,65 à 0,83) n'atteint pas l'exigence métier de 0,75 ;
- sa précision chute de 0,875 à environ 0,65 en conditions temporelles ;
- ses fausses alertes portent sur des montants élevés.

Autrement dit, le modèle est fiable pour **signaler**, pas encore pour **décider**.

Les conditions sont les suivantes :

1. **Mode d'usage sans impact direct sur le client.**
   - Les scores sont produits en parallèle des règles existantes, qui restent seules à décider.
   - Les alertes sont revues par un analyste.
   - La charge prévisible est d'environ 200 alertes par jour pour 142 000 transactions : elle doit être validée avec l'équipe.
2. **Version corrigée.**
   - v1.1.0 sans `Time`, avec un seuil établi par validation croisée.
   - Mêmes contrôles automatiques au vert : CI, contrat de données, empreinte du modèle, latence.
3. **Circuit de secours** pour les transactions rejetées par la validation des entrées : les règles existantes s'appliquent.
4. **Surveillance active dès le premier jour** :
   - PSI par variable, calculé sur des fenêtres de même saisonnalité ;
   - taux d'alertes et taux de rejet ;
   - latence ;
   - procédure de retour arrière.

   Nos simulations montrent que le taux d'alertes seul ne détecte pas les dérives : le PSI est indispensable.
5. **Périmètre et critères de sortie fixés à l'avance.**
   - Durée et part du trafic limitées.
   - Le pilote se termine quand **au moins 250 fraudes labellisées** ont été observées : c'est l'effectif nécessaire pour estimer le rappel à environ ±5 points près.
   - Critères de succès : ceux d'ENF1, mesurés sur ces données nouvelles.

### Preuves supplémentaires avant une mise en production

1. **Performance sur des données récentes et indépendantes.**
   - Rappel ≥ 0,75, précision ≥ 0,30 et FPR ≤ 0,5 %, mesurés sur les transactions du pilote, postérieures à l'entraînement.
   - Avec des intervalles de confiance assez étroits pour conclure : au moins 250 fraudes.
2. **Supériorité sur le dispositif existant** : comparer en parallèle, sur les mêmes transactions, le système actuel et le modèle, en fraudes détectées, en fausses alertes et en montant intercepté.
3. **Seuil justifié par le coût.** Choisir le seuil à partir des coûts réels, validés par la banque : coût moyen d'une fraude manquée, coût d'une fausse alerte (traitement et attrition du client).
4. **Audit d'équité.** Mesurer le FPR et le rappel par segment de clientèle (âge, zone, canal), sur un jeu d'audit enrichi et licite. Notre jeu de données ne le permet pas.
5. **Qualification sur les vraies variables.** Le système de production n'utilisera pas des composantes PCA anonymisées. Il faudra le requalifier avec ses variables réelles, et fournir des **explications** exploitables par les analystes et opposables aux clients.
6. **Tests de l'API réelle.**
   - Latence de bout en bout et tests de charge aux pointes attendues (Locust).
   - Résilience aux pannes, authentification, limitation de débit.
   - Test d'intrusion.
   - Signature du modèle (ANO-08).
7. **Robustesse adverse** : rechercher les manipulations minimales qui font passer une fraude sous le seuil, en priorité sur les petits montants.
8. **Surveillance de production opérationnelle** : tableaux de bord et alertes (par exemple Prometheus et Grafana), collecte des labels différés, procédure documentée de réentraînement et de requalification.
9. **Validations non techniques** :
   - métier : niveau de risque accepté ;
   - opérationnelle : capacité des analystes, procédure client en cas de blocage ;
   - sécurité : revue par l'équipe sécurité ;
   - réglementaire : gouvernance des modèles, documentation, et conformité à vérifier avec le délégué à la protection des données et la conformité, notamment pour le RGPD (décisions automatisées) et les exigences des autorités bancaires.

**En résumé :** nos tests montrent que le pipeline est techniquement solide et que le modèle discrimine bien. Ils ne démontrent pas encore que le modèle atteint le niveau de détection exigé, et ils reposent sur des données qui ne représentent pas la fraude actuelle. Le pilote doit servir à produire ces preuves manquantes, sans exposer les clients à ses erreurs.

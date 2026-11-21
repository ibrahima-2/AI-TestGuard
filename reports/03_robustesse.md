# 3. Robustesse et sensibilité

> Source : `reports/results/robustness.json` (`python -m app.cli robustness`). Modèle évalué : régression logistique gelée, seuil 0,999992.
> Jeu utilisé : le test (56 746 transactions, 95 fraudes). Aucun réglage n'est fait à partir de ces résultats.
> Tests automatisés associés : `tests/test_robustness.py`.

## 3.1 Comportement attendu par scénario

| Scénario | Comportement attendu | Justification |
|---|---|---|
| 1. Données manquantes | **Rejet** | Compléter une variable d'une transaction à scorer pourrait masquer une anomalie ou un contournement. La transaction doit passer par un circuit de secours, par exemple les règles existantes. |
| 2. Valeurs infinies ou non numériques | **Rejet** | Le contrat est violé et aucun score fiable n'est possible. |
| 3. `Amount` extrêmement élevé | **Traitement + alerte** | La valeur est techniquement valide. Le `log1p` garde le calcul fini, mais la valeur sort de la plage d'entraînement. |
| 4. Perturbation limitée | **Traitement** ; mesurer la sensibilité | Un bruit faible doit avoir un effet faible. Une bascule de classe n'est acceptable que près de la frontière de décision. |
| 5. Distribution différente | **Traitement + alerte de dérive** | Le modèle extrapole : il faut prévenir l'exploitation (PSI, plages). |
| 6. Schéma invalide | **Rejet** | Colonne absente, en trop, dupliquée ou mal typée. |

## 3.2 Scénarios de rejet (1, 2, 6)

| Scénario | Lignes injectées | Lignes invalides détectées | Rejeté | Message | Résultat |
|---|---|---|---|---|---|
| 1. Valeurs manquantes (1 % des lignes, `V14` et `Amount`) | 567 | 567 | oui | Valeurs manquantes dans : ['V14', 'Amount'] | conforme |
| 2a. Valeurs infinies (0,5 % des lignes, `V4`) | 284 | 284 | oui | Valeurs infinies dans : ['V4'] | conforme |
| 2b. Texte « abc » dans `Amount` (10 lignes) | 10 | 10 | oui | Type non numérique pour : ['Amount'] | conforme |
| 6a. Colonne `V14` absente | — | — | oui | Colonnes manquantes : ['V14'] | conforme |
| 6b. Colonne `card_number` en plus (mode strict de l'API) | — | — | oui | Colonnes non prévues par le contrat : ['card_number'] | conforme |
| 6c. `Amount` transmis en texte | — | — | oui | Type non numérique pour : ['Amount'] | conforme |
| 6d. Colonne `V1` dupliquée | — | — | oui | Colonnes dupliquées : ['V1'] | conforme |
| 6e. Montant négatif | 56 746 | 56 746 | oui | Montant (Amount) négatif interdit | conforme |

- Le comptage ligne par ligne (`invalid_row_mask`) retrouve **exactement** les lignes corrompues. On peut donc isoler les transactions invalides et scorer les autres (`test_valid_rows_remain_scorable_after_filtering`).
- Les messages citent les colonnes concernées, **jamais les valeurs**, pour ne pas exposer de données de transaction.

## 3.3 Scénarios de traitement (3, 4, 5) : sensibilité mesurée

Les perturbations sont propres à chaque type de variable :
- **`V1` à `V28`** : bruit gaussien d'écart-type ε × σ, où σ est l'écart-type d'entraînement de chaque composante. Ce sont des composantes PCA, sans unité métier.
- **`Amount`** : variation *relative*, car un montant ne se bruite pas en absolu.
- **`Time`** : décalage en secondes.

Référence sans perturbation : rappel 0,737 ; précision 0,875 ; FPR 0,018 %.

| Scénario | Δ proba moyen | Δ proba P95 | Δ proba max | Bascules (→ fraude / → normale) | Rappel | Précision | FPR | Hors plage |
|---|---|---|---|---|---|---|---|---|
| 3a. `Amount` × 10 | 0,0099 | 0,040 | 0,052 | 0 (0 / 0) | 0,737 | 0,875 | 0,018 % | 0,3 % |
| 3b. `Amount` = 10⁹ partout | 0,119 | 0,348 | 0,439 | 6 (6 / 0) | 0,758 | 0,837 | 0,025 % | **100 %** (`Amount`) |
| 4a. Bruit V, ε = 1 % σ | 0,0011 | 0,005 | 0,024 | 1 (0 / 1) | 0,737 | 0,886 | 0,016 % | 0 % |
| 4b. Bruit V, ε = 5 % σ | 0,0052 | 0,023 | 0,150 | 1 (0 / 1) | 0,737 | 0,886 | 0,016 % | 0,1 % |
| 4c. Bruit V, ε = 10 % σ | 0,0106 | 0,047 | 0,232 | 1 (1 / 0) | 0,737 | 0,864 | 0,019 % | 0,5 % |
| 4d. `Amount` ± 5 % | 0,0001 | 0,0004 | 0,001 | 0 | 0,737 | 0,875 | 0,018 % | 0 % |
| 4e. `Time` ± 60 s | 0,0000 | 0,0001 | 0,0002 | 0 | 0,737 | 0,875 | 0,018 % | 0 % |
| **5a. `Time` + 48 h** | 0,054 | 0,259 | 0,464 | **5 (0 / 5)** | **0,695** | 0,880 | 0,016 % | **100 %** (`Time`) |
| **5b. V1–V28 × 1,5** | 0,051 | 0,315 | 0,468 | **17 (17 / 0)** | 0,779 | **0,763** | **0,041 %** | 20,6 % (`V1`) |

### Interprétation

- **Bruit faible (4a à 4e) : comportement stable.** Avec 10 % de σ, la probabilité varie en moyenne de 0,01, et une seule transaction sur 56 746 change de classe. La sensibilité augmente avec l'amplitude, ce que vérifie `test_sensitivity_grows_with_noise_amplitude`. Les bascules ont lieu près de la frontière, ce que vérifie `test_class_changes_concentrate_near_decision_boundary`. Il n'y a donc **pas d'instabilité anormale**.
- **Montants (3a, 3b, 4d).** Grâce au `log1p`, le modèle est peu sensible au montant : multiplier tous les montants par 10 ne change aucune décision. Un montant absurde (10⁹) augmente les scores, avec 6 transactions qui basculent vers « fraude ». Ce comportement est prudent et acceptable, et la valeur est **signalée hors plage dans 100 % des cas**. Revers de cette faible sensibilité : le montant pèse peu dans la décision, ce qui explique en partie les fraudes de petit montant manquées (rapport 02, § 2.7).
- **Anomalie — dépendance à `Time` (5a).** `Time` mesure les secondes écoulées depuis le début de la collecte, pas l'heure de la journée. En production, toutes les valeurs dépassent la plage d'entraînement. Décaler `Time` de 48 h **fait perdre 4 fraudes détectées** (rappel 0,737 → 0,695). Le modèle s'appuie donc sur une variable sans signification stable : **anomalie ANO-03**.
- **Changement d'échelle (5b).** Si la distribution des composantes s'élargit (nouveau parc de terminaux, autre population), 17 transactions basculent vers « fraude ». Parmi elles, 4 sont de vraies fraudes et **13 sont de nouvelles fausses alertes** : la précision baisse de 11 points et le FPR est multiplié par 2,3. Ce cas est détecté par le PSI (rapport 07), mais **pas** par le taux d'alertes global.

## 3.4 Validation temporelle (scénario 5, distribution réelle différente)

Pour tester la généralisation à une période future, le même modèle (même configuration) est entraîné sur le **jour 1**, avec 25 % du jour 1 en validation pour le seuil. Il est évalué sur l'intégralité du **jour 2** : 139 490 transactions, dont 201 fraudes.

| Variante | Seuil | Rappel jour 2 | Précision jour 2 | FPR jour 2 | PR-AUC jour 2 | ROC-AUC | TP / FP / FN |
|---|---|---|---|---|---|---|---|
| Avec `Time` | 0,999910 | 0,781 | 0,611 | 0,072 % | 0,732 | 0,970 | 157 / 100 / 44 |
| Sans `Time` | 0,999924 | 0,741 | 0,693 | 0,047 % | 0,733 | 0,968 | 149 / 66 / 52 |

- **La précision baisse nettement en conditions temporelles** : 0,61 à 0,69, contre 0,875 avec un découpage aléatoire. Le découpage aléatoire mélange les deux jours et surestime donc la qualité réelle. Les critères restent néanmoins respectés pour la précision et le FPR.
- **Le rappel temporel (0,74 à 0,78) reste autour du seuil de 0,75.** Ce résultat est cohérent avec le rappel de 0,737 mesuré sur le test : la limite constatée n'est pas un accident du découpage.
- La variante avec `Time` extrapole sur des valeurs jamais vues (tout le jour 2 est hors plage). Son léger avantage de rappel n'est pas interprétable comme un signal fiable.

## 3.5 Tests par propriétés (Hypothesis)

- Toute transaction valide (composantes entre −50 et 50, montant entre 0 et 10¹², temps entre 0 et 10⁶) reçoit une probabilité dans [0, 1] (`test_any_valid_transaction_gets_valid_probability`).
- Toute combinaison de bruit (ε ≤ 2) et de facteur sur le montant (≤ 100) respecte le contrat de sortie (`test_bounded_perturbations_never_break_output_contract`).
- Tout lot valide de 1 à 20 lignes passe la validation et reste fini (`test_any_valid_batch_passes_validation_and_stays_finite`).

## 3.6 Anomalies et limites

| ID | Constat | Gravité | Statut |
|---|---|---|---|
| ANO-03 | Dépendance à `Time`, variable non stationnaire : −4 points de rappel à +48 h, 100 % hors plage en production | Moyenne | Correctif v1.1.0 (sans `Time`) évalué en non-régression : accepté |
| ANO-05 | Montants extrêmes : scores fortement relevés (jusqu'à +0,44) | Mineure | Accepté (comportement prudent) ; alerte « hors plage » en place |
| ANO-06 | Changement d'échelle des composantes : FPR × 2,3 sans hausse visible du taux d'alertes global | Moyenne | Couvert par la surveillance PSI par variable |

**Limites de l'analyse.**
- Les perturbations restent synthétiques. Il n'y a pas d'attaque adverse ciblée, par exemple une recherche des modifications minimales qui font passer une fraude sous le seuil.
- Les composantes PCA empêchent de construire des perturbations métier réalistes, comme un changement de pays ou de commerçant.
- La validation temporelle couvre un seul passage du jour 1 au jour 2 : 48 heures ne suffisent pas pour juger d'une dérive saisonnière.

# 3. Robustesse et sensibilité

Source : `reports/results/robustness.json`, produit par `python -m app.cli robustness`. Le modèle évalué est la régression logistique sauvegardée (seuil 0,999992), sur le jeu de test (56 746 transactions, 95 fraudes). Aucun réglage n'est fait à partir de ces résultats. Les tests automatisés correspondants sont dans `tests/test_robustness.py`.

## 3.1 Comportement attendu

| Scénario | Comportement attendu | Raison |
|---|---|---|
| 1. Données manquantes | Rejet | Compléter une variable d'une transaction à scorer pourrait masquer une anomalie. La transaction est alors traitée par le dispositif existant. |
| 2. Valeurs infinies ou non numériques | Rejet | Le contrat est violé et aucun score fiable ne peut être calculé. |
| 3. Montant très élevé | Traitement et alerte | La valeur est valide et le calcul reste fini grâce à `log1p`, mais elle sort de la plage observée à l'entraînement. |
| 4. Perturbation limitée | Traitement, avec mesure de la sensibilité | Un faible bruit doit avoir un faible effet ; un changement de classe n'est acceptable que près de la frontière de décision. |
| 5. Distribution différente | Traitement et alerte de dérive | Le modèle extrapole, l'exploitation doit en être informée. |
| 6. Schéma invalide | Rejet | Colonne absente, supplémentaire, dupliquée ou mal typée. |

## 3.2 Scénarios de rejet

| Scénario | Lignes modifiées | Lignes invalides détectées | Rejeté | Message | Résultat |
|---|---|---|---|---|---|
| 1. NaN dans `V14` et `Amount` (1 % des lignes) | 567 | 567 | oui | Valeurs manquantes dans : ['V14', 'Amount'] | conforme |
| 2a. Valeur infinie dans `V4` (0,5 % des lignes) | 284 | 284 | oui | Valeurs infinies dans : ['V4'] | conforme |
| 2b. Texte « abc » dans `Amount` (10 lignes) | 10 | 10 | oui | Type non numérique pour : ['Amount'] | conforme |
| 6a. Colonne `V14` absente | — | — | oui | Colonnes manquantes : ['V14'] | conforme |
| 6b. Colonne `card_number` ajoutée (mode strict) | — | — | oui | Colonnes non prévues par le contrat : ['card_number'] | conforme |
| 6c. `Amount` transmis sous forme de texte | — | — | oui | Type non numérique pour : ['Amount'] | conforme |
| 6d. Colonne `V1` dupliquée | — | — | oui | Colonnes dupliquées : ['V1'] | conforme |
| 6e. Montant négatif | 56 746 | 56 746 | oui | Montant (Amount) négatif interdit | conforme |

La fonction `invalid_row_mask` retrouve exactement les lignes modifiées. Il est donc possible d'écarter les transactions invalides et de scorer les autres (`test_valid_rows_remain_scorable_after_filtering`). Les messages d'erreur indiquent les colonnes concernées mais jamais les valeurs, pour ne pas exposer de données de transaction.

## 3.3 Sensibilité aux perturbations

Les perturbations sont adaptées à chaque type de variable. Sur `V1` à `V28`, qui sont des composantes PCA sans unité, on ajoute un bruit gaussien d'écart-type ε × σ, où σ est l'écart-type de la composante sur l'entraînement. Sur `Amount`, la variation est relative. Sur `Time`, il s'agit d'un décalage en secondes.

Sans perturbation, le rappel est de 0,737, la précision de 0,875 et le FPR de 0,018 %.

| Scénario | Δ proba moyen | Δ proba P95 | Δ proba max | Changements de classe (vers fraude / vers normale) | Rappel | Précision | FPR | Hors plage |
|---|---|---|---|---|---|---|---|---|
| 3a. `Amount` × 10 | 0,0099 | 0,040 | 0,052 | 0 (0 / 0) | 0,737 | 0,875 | 0,018 % | 0,3 % |
| 3b. `Amount` = 10⁹ pour toutes les transactions | 0,119 | 0,348 | 0,439 | 6 (6 / 0) | 0,758 | 0,837 | 0,025 % | 100 % (`Amount`) |
| 4a. Bruit sur V, ε = 1 % σ | 0,0011 | 0,005 | 0,024 | 1 (0 / 1) | 0,737 | 0,886 | 0,016 % | 0 % |
| 4b. Bruit sur V, ε = 5 % σ | 0,0052 | 0,023 | 0,150 | 1 (0 / 1) | 0,737 | 0,886 | 0,016 % | 0,1 % |
| 4c. Bruit sur V, ε = 10 % σ | 0,0106 | 0,047 | 0,232 | 1 (1 / 0) | 0,737 | 0,864 | 0,019 % | 0,5 % |
| 4d. `Amount` ± 5 % | 0,0001 | 0,0004 | 0,001 | 0 | 0,737 | 0,875 | 0,018 % | 0 % |
| 4e. `Time` ± 60 s | 0,0000 | 0,0001 | 0,0002 | 0 | 0,737 | 0,875 | 0,018 % | 0 % |
| 5a. `Time` + 48 h | 0,054 | 0,259 | 0,464 | 5 (0 / 5) | 0,695 | 0,880 | 0,016 % | 100 % (`Time`) |
| 5b. V1 à V28 multipliées par 1,5 | 0,051 | 0,315 | 0,468 | 17 (17 / 0) | 0,779 | 0,763 | 0,041 % | 20,6 % (`V1`) |

Avec un bruit faible (scénarios 4a à 4e), le modèle reste stable. Pour ε = 10 % σ, la probabilité varie en moyenne de 0,01 et une seule transaction sur 56 746 change de classe. Les tests automatisés vérifient que la sensibilité augmente avec l'amplitude du bruit et que les changements de classe concernent des transactions proches du seuil.

Le modèle est peu sensible au montant, du fait de la transformation `log1p`. Multiplier tous les montants par 10 ne change aucune décision. Un montant de 10⁹ relève les scores et fait passer 6 transactions en fraude. Ce comportement prudent est acceptable, d'autant que ces valeurs sont toutes signalées hors plage. Cette faible sensibilité explique aussi en partie pourquoi les fraudes de petit montant sont mal détectées (rapport 02, section 2.7).

Le scénario 5a met en évidence un problème. `Time` mesure le nombre de secondes écoulées depuis le début de la collecte, et non l'heure de la journée. En production, toutes ses valeurs seraient supérieures à celles de l'entraînement. Un décalage de 48 heures fait perdre 4 fraudes détectées (rappel de 0,737 à 0,695). Le modèle s'appuie donc sur une variable qui n'a pas de sens stable (anomalie ANO-03).

Dans le scénario 5b, la distribution des composantes s'élargit, comme cela pourrait arriver avec une autre population de clients ou un nouveau type de terminal. 17 transactions passent en fraude : 4 sont de vraies fraudes et 13 sont de nouvelles fausses alertes. La précision baisse de 11 points et le FPR est multiplié par 2,3. Le PSI détecte cette dérive (rapport 07), mais le taux global d'alertes ne la fait pas apparaître.

## 3.4 Validation temporelle

Pour estimer le comportement du modèle sur une période ultérieure, la même configuration est entraînée sur le premier jour (25 % de ce jour servent de validation pour le seuil), puis évaluée sur la totalité du second jour, soit 139 490 transactions dont 201 fraudes.

| Variante | Seuil | Rappel | Précision | FPR | PR-AUC | ROC-AUC | TP / FP / FN |
|---|---|---|---|---|---|---|---|
| Avec `Time` | 0,999910 | 0,781 | 0,611 | 0,072 % | 0,732 | 0,970 | 157 / 100 / 44 |
| Sans `Time` | 0,999924 | 0,741 | 0,693 | 0,047 % | 0,733 | 0,968 | 149 / 66 / 52 |

La précision est nettement plus faible qu'avec le découpage aléatoire (0,61 à 0,69 contre 0,875). Le découpage aléatoire mélange les deux journées et donne donc une estimation trop favorable. Les critères de précision et de FPR restent respectés. Le rappel, entre 0,74 et 0,78, reste proche du seuil de 0,75, ce qui est cohérent avec le résultat obtenu sur le test.

Dans la variante avec `Time`, toutes les valeurs du second jour sont hors de la plage d'entraînement. Son rappel un peu meilleur ne peut donc pas être considéré comme fiable.

## 3.5 Tests par propriétés

Trois propriétés sont vérifiées avec Hypothesis :

- toute transaction valide (composantes entre −50 et 50, montant entre 0 et 10¹², temps entre 0 et 10⁶) reçoit une probabilité comprise entre 0 et 1 ;
- toute combinaison d'un bruit (ε ≤ 2) et d'un facteur sur le montant (≤ 100) respecte le contrat de sortie ;
- tout lot valide de 1 à 20 lignes passe la validation et reste fini après prétraitement.

## 3.6 Anomalies et limites

| ID | Constat | Gravité | Statut |
|---|---|---|---|
| ANO-03 | Dépendance à `Time` : −4 points de rappel pour un décalage de 48 h, valeurs entièrement hors plage en production | Moyenne | Version v1.1.0 sans `Time` acceptée en non-régression |
| ANO-05 | Montants extrêmes : scores relevés jusqu'à +0,44 | Mineure | Accepté ; alerte hors plage en place |
| ANO-06 | Élargissement des composantes : FPR multiplié par 2,3 sans hausse visible du taux d'alertes | Moyenne | Couvert par la surveillance du PSI par variable |

Ces tests ont plusieurs limites. Les perturbations sont synthétiques et aucune attaque ciblée n'a été simulée, par exemple la recherche des plus petites modifications qui feraient passer une fraude sous le seuil. Les composantes PCA ne permettent pas de construire des perturbations réalistes du point de vue métier (changement de pays, de commerçant). Enfin, la validation temporelle ne porte que sur deux journées, ce qui ne permet pas d'étudier des effets saisonniers.

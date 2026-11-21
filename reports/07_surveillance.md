# 7. Surveillance après déploiement

Source des simulations : `reports/results/drift.json` (`python -m app.cli drift`). Code : `app/monitoring.py`. Tests : `tests/test_robustness.py`.

## 7.1 Dérive des données et dérive de concept

| | Dérive des données (data drift) | Dérive de concept (concept drift) |
|---|---|---|
| Définition | La distribution des entrées P(X) change | La relation entre les entrées et la cible P(y \| X) change, par exemple quand les fraudeurs changent de méthode |
| Exemple | Nouveau type de terminal, hausse des montants, autre population de clients | Les fraudes se concentrent sur des montants que le modèle détecte mal |
| Mesurable sans label | Oui, en comparant les distributions | Non, il faut connaître les fraudes confirmées |
| Indicateurs | PSI, test de Kolmogorov-Smirnov, taux hors plage, taux d'alertes | Rappel, précision, FPR et FNR sur les transactions labellisées |

Une dérive des données ne dégrade pas forcément le modèle, et une dérive de concept peut survenir sans changement visible des entrées. Les deux types d'indicateurs sont donc nécessaires.

## 7.2 Indicateurs

Indicateurs disponibles immédiatement, sans label :

| Indicateur | Calcul | Seuil | Fréquence |
|---|---|---|---|
| PSI par variable | Classes définies par les déciles de l'entraînement | < 0,10 stable, 0,10 à 0,20 à surveiller, > 0,20 alerte | Quotidienne |
| Test de Kolmogorov-Smirnov | `scipy.stats.ks_2samp` | Utilisé en complément du PSI ; la p-valeur seule n'est pas exploitable, car sur 100 000 transactions le moindre écart devient significatif | Quotidienne |
| Taux hors plage | Part des transactions en dehors des valeurs minimale et maximale de l'entraînement | > 1 % | Horaire |
| Taux de transactions signalées | Comparé au taux mesuré sur la validation (0,153 %) | Rapport > 2 ou < 0,5 | Horaire |
| Taux de rejet des entrées | Requêtes refusées par la validation | > 0,1 % | Horaire |
| Disponibilité et latence | Erreurs 5xx, latence P95 et P99 | P95 > 50 ms, disponibilité < 99,9 % | Continue |

Indicateurs qui nécessitent les labels, connus avec retard (contestation du client, rétrofacturation, enquête) :

| Indicateur | Source du label | Délai | Seuil |
|---|---|---|---|
| Précision des alertes | Décision de l'analyste | Quelques heures à quelques jours | < 0,30 |
| FPR | Alertes jugées légitimes par les analystes | Quelques heures à quelques jours | > 0,5 % |
| Rappel et FNR | Fraudes signalées après coup par les clients ou par rétrofacturation | Plusieurs semaines | Rappel < 0,75 sur une fenêtre glissante d'au moins 250 fraudes |
| PR-AUC | Les deux sources | Plusieurs semaines | Baisse > 0,03 par rapport à la référence |

Le rappel est l'indicateur le plus long à obtenir, car une fraude manquée n'est connue qu'après coup. Il doit être calculé sur une fenêtre assez large : 250 fraudes donnent un intervalle de confiance d'environ ±5 points.

## 7.3 Simulations

La référence est le jeu d'entraînement. Le taux d'alertes attendu, mesuré sur la validation, est de 0,153 %.

| Lot | Variables en alerte (PSI) | Variables à surveiller | Plus fortes dérives (PSI) | Taux signalé | Statut du taux |
|---|---|---|---|---|---|
| Jeu de test (même distribution) | 0 | 0 | V25 = 0,001 | 0,141 % | normal |
| Transactions du jour 1 | 4 | 1 | Time = 5,11 ; V1 = 2,21 ; V3 = 0,40 ; V28 = 0,30 | 0,171 % | normal |
| Transactions du jour 2 | 4 | 2 | Time = 6,10 ; V1 = 0,50 ; V3 = 0,33 ; V28 = 0,29 | 0,120 % | normal |
| Simulation : `Amount` × 3 | 1 | 0 | Amount = 0,41 | 0,141 % | normal |
| Simulation : V1 à V5 × 1,5 | 4 | 1 | V1 = 0,48 ; V3 = 0,24 ; V4 = 0,21 ; V2 = 0,21 | 0,153 % | normal |
| Simulation : `Amount` × 10 | 1 | 0 | Amount = 1,60 | 0,141 % | normal |

Sur un lot de même distribution que l'entraînement, aucune alerte n'est levée : tous les PSI du jeu de test sont inférieurs à 0,001.

Dans toutes les simulations, le taux d'alertes reste dans la plage normale. Multiplier les montants par 10 ou élargir les composantes ne change presque pas le nombre de transactions signalées, alors que le rapport 03 montre que l'élargissement des composantes multiplie le FPR par 2,3. Le PSI détecte chacune de ces dérives. Suivre uniquement le taux d'alertes ne suffirait donc pas.

`Time` déclenche une alerte permanente (PSI supérieur à 5), ce qui est attendu pour une variable qui croît indéfiniment. C'est une raison supplémentaire de la retirer du modèle (version v1.1.0) ou de la remplacer par l'heure de la journée.

Les composantes varient aussi d'un jour à l'autre (PSI de V1 égal à 2,21 pour le jour 1). Une partie de leur distribution dépend donc du moment de la journée. En production, il faudra comparer des fenêtres de même durée et de même période (même jour de la semaine, même plage horaire), sinon les alertes PSI seront trop fréquentes.

## 7.4 Réponse aux alertes

| Situation | Action |
|---|---|
| PSI > 0,20 sur une variable, métriques labellisées stables | Rechercher la cause (source de données, changement de population) et renforcer la surveillance |
| Hausse du taux de rejet des entrées | Vérifier le contrat avec le système amont ; incident technique probable |
| Taux d'alertes multiplié par 2 | Vérifier la capacité des analystes et une éventuelle dérive des entrées ; envisager le retour au modèle ou au seuil précédent |
| Précision ou FPR hors critère | Investiguer et réévaluer le seuil sur des données de validation récentes |
| Rappel < 0,75 sur une fenêtre glissante | Dérive de concept probable : réentraînement sur des données récentes, non-régression contre la référence, puis nouvelle qualification |

## 7.5 Mise en œuvre

Dans le prototype, les fonctions `drift_report`, `out_of_range_rates` et `alert_rate_status`, appelées par `python -m app.cli drift`, produisent un rapport JSON et simulent les alertes.

En production, les indicateurs pourraient être exportés vers Prometheus (compteurs de requêtes, de rejets et d'alertes, histogrammes de latence, PSI calculé par une tâche planifiée) et suivis dans Grafana avec des règles d'alerte. Les scores et les décisions seraient historisés, sans les données brutes de carte, pour recalculer les métriques à l'arrivée des labels.

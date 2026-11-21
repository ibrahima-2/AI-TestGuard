# 7. Stratégie de surveillance après déploiement

> Source des simulations : `reports/results/drift.json` (`python -m app.cli drift`). Code : `app/monitoring.py`. Tests : `tests/test_robustness.py`.

## 7.1 Deux phénomènes à distinguer

| | Dérive des données (*data drift*) | Dérive de concept (*concept drift*) |
|---|---|---|
| Définition | La distribution des entrées P(X) change | La relation entre les entrées et la cible P(y \| X) change : les fraudeurs changent de méthode |
| Exemple | Nouveau type de terminal, hausse des montants moyens, autre population de clients | Les fraudes se concentrent sur de petits montants qu'on ne savait pas détecter |
| Détectable sans label ? | **Oui**, par comparaison des distributions | **Non**, il faut les labels (fraudes confirmées) |
| Indicateurs | PSI, Kolmogorov-Smirnov, taux hors plage, taux d'alertes | Rappel, précision, FPR et FNR calculés sur les transactions labellisées |

Une dérive des données ne dégrade pas forcément le modèle. À l'inverse, une dérive de concept peut se produire sans aucun changement visible de la distribution des entrées. **Les deux familles d'indicateurs sont donc nécessaires.**

## 7.2 Indicateurs et seuils

### Suivi immédiat (sans label)

| Indicateur | Calcul | Seuil d'alerte | Fréquence |
|---|---|---|---|
| PSI par variable | Classes définies par les déciles de l'entraînement | < 0,10 stable ; 0,10–0,20 à surveiller ; > 0,20 alerte | Quotidienne |
| Test de Kolmogorov-Smirnov | `scipy.stats.ks_2samp` | Complément du PSI. La p-valeur n'est pas utilisée seule : avec 100 000 transactions, tout écart minime devient « significatif ». | Quotidienne |
| Taux hors plage | Part des transactions hors des valeurs min et max d'entraînement, par variable | > 1 % | Horaire |
| Taux de transactions signalées | Alertes rapportées au volume, comparées au taux de validation (0,153 %) | Rapport > 2 ou < 0,5 | Horaire |
| Taux de rejet des entrées | Requêtes rejetées par la validation | > 0,1 % | Horaire |
| Disponibilité et latence de l'API | Taux d'erreur 5xx, latence P95 et P99 | P95 > 50 ms ; disponibilité < 99,9 % | Continue |

### Suivi différé (avec labels)

Les labels arrivent avec retard : contestation du client, rétrofacturation (*chargeback*) ou enquête, soit plusieurs jours à plusieurs semaines.

| Indicateur | Source du label | Délai | Seuil |
|---|---|---|---|
| Précision des alertes | Décision de l'analyste sur chaque alerte | Heures à jours | < 0,30 |
| FPR | Alertes classées « légitimes » par les analystes, rapportées au volume | Heures à jours | > 0,5 % |
| **Rappel / FNR** | Fraudes signalées *a posteriori* par les clients ou par rétrofacturation | **Semaines** | Rappel < 0,75 sur une fenêtre glissante d'au moins 250 fraudes |
| PR-AUC | Combinaison des deux sources | Semaines | Baisse > 0,03 par rapport à la référence |

Le **rappel est la métrique la plus lente à obtenir**, car les fraudes manquées ne sont découvertes qu'après coup. Il faut l'estimer sur des fenêtres glissantes assez larges : au moins 250 fraudes pour un intervalle de confiance d'environ ±5 points.

## 7.3 Simulations

Référence : le jeu d'entraînement. Le taux d'alertes attendu, mesuré sur la validation, est de 0,153 %.

| Lot | Variables en alerte PSI | Variables à surveiller | Plus fortes dérives (PSI) | Taux signalé | Statut du taux |
|---|---|---|---|---|---|
| Jeu de test (même distribution) | 0 | 0 | V25 = 0,001 | 0,141 % | normal |
| Transactions du jour 1 | 4 | 1 | **Time = 5,11** ; V1 = 2,21 ; V3 = 0,40 ; V28 = 0,30 | 0,171 % | normal |
| Transactions du jour 2 | 4 | 2 | **Time = 6,10** ; V1 = 0,50 ; V3 = 0,33 ; V28 = 0,29 | 0,120 % | normal |
| Simulation : `Amount` × 3 | 1 | 0 | Amount = 0,41 | 0,141 % | normal |
| Simulation : V1–V5 × 1,5 | 4 | 1 | V1 = 0,48 ; V3 = 0,24 ; V4 = 0,21 ; V2 = 0,21 | 0,153 % | normal |
| Simulation : `Amount` × 10 | 1 | 0 | Amount = 1,60 | 0,141 % | normal |

### Enseignements

1. **Aucune fausse alerte sur un lot de même distribution** : tous les PSI du jeu de test sont inférieurs à 0,001.
2. **Le taux d'alertes seul ne suffit pas.** Dans toutes les simulations, il reste « normal ». Multiplier les montants par 10 ou élargir les composantes ne change presque pas le nombre d'alertes. Pourtant, le rapport 03 montre que l'élargissement des composantes multiplie le FPR par 2,3. **Le PSI par variable détecte ces dérives ; le taux d'alertes, non.**
3. **`Time` déclenche une alerte structurelle** (PSI > 5). C'est attendu, puisque la variable croît indéfiniment. Cela confirme qu'elle doit être retirée du modèle (version v1.1.0) ou remplacée par l'heure de la journée.
4. **Les composantes varient d'un jour à l'autre** : PSI de V1 égal à 2,21 pour le jour 1. Une partie de la distribution dépend du moment de la journée et du jour. En production, la référence doit être comparée à une fenêtre de même durée et de même saisonnalité (même jour de la semaine, même plage horaire), sinon les alertes PSI seront trop fréquentes.

## 7.4 Réponse aux alertes

| Situation | Action |
|---|---|
| PSI > 0,20 sur une variable, sans baisse des métriques labellisées | Analyser la cause (source de données, changement de population) ; surveillance renforcée |
| Hausse du taux de rejet des entrées | Vérifier le contrat avec le système amont ; incident technique probable |
| Taux d'alertes × 2 | Vérifier la capacité des analystes ; vérifier une dérive des entrées ; envisager de revenir au seuil ou au modèle précédent |
| Précision ou FPR hors critère (labels des analystes) | Investigation ; réévaluer le seuil **sur des données de validation récentes** |
| Rappel < 0,75 sur une fenêtre glissante | Dérive de concept probable : réentraînement sur des données récentes, puis non-régression contre la référence et nouvelle qualification |

## 7.5 Mise en œuvre

- **Prototype (ce projet)** : fonctions `drift_report`, `out_of_range_rates`, `alert_rate_status` et commande `python -m app.cli drift`, qui produit un rapport JSON et simule les alertes.
- **Production** : exporter les indicateurs vers **Prometheus** (compteurs de requêtes, rejets et alertes ; histogrammes de latence ; jauges PSI calculées par une tâche planifiée). Construire des tableaux de bord et des règles d'alerte dans **Grafana**. Historiser les scores et les décisions, sans les données brutes de la carte, pour recalculer les métriques quand les labels arrivent.

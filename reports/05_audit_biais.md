# 5. Note d'audit des biais

## 5.1 Objectif

L'audit des biais consiste à vérifier si la performance du modèle varie selon des groupes de clients (âge, sexe, zone géographique, type de carte, segment). Un taux de faux positifs plus élevé pour un groupe signifie davantage de blocages injustifiés pour ce groupe. Un rappel plus faible signifie que ce groupe est moins bien protégé contre la fraude.

## 5.2 Constat

Le jeu Credit Card Fraud Detection ne contient que `Time` (secondes depuis la première transaction), les composantes `V1` à `V28` issues d'une PCA dont les variables d'origine ne sont pas publiées, `Amount` et la cible `Class`. Il ne contient aucune caractéristique du client : ni âge, ni sexe, ni localisation, ni nationalité, ni type de carte, ni canal de paiement.

Ce jeu ne permet donc pas d'évaluer l'équité du système selon ces caractéristiques, et aucune conclusion n'est tirée à ce sujet dans ce projet.

## 5.3 Ce qui n'a pas été fait

Aucun groupe n'a été reconstruit à partir des composantes anonymisées. Découper les clients selon le signe ou les quantiles de `V1`, par exemple, produirait des catégories sans signification, et toute conclusion tirée de ces catégories serait trompeuse.

Le montant et l'heure n'ont pas non plus été utilisés comme substituts d'un groupe de clients. Des écarts de performance existent selon le montant : les fraudes manquées sont plutôt de petits montants (médiane de 3,76 contre 31,99 pour les fraudes détectées) et les fausses alertes concernent plutôt des montants élevés (moyenne de 2 836). Ce sont des constats sur la qualité du service, à suivre, mais pas des constats d'équité.

## 5.4 Données nécessaires

Une analyse d'équité demanderait, pour chaque transaction labellisée, des variables de groupe fiables et dont l'usage est légalement autorisé :

| Variable | Usage | Précaution |
|---|---|---|
| Tranche d'âge du porteur | Écarts de FPR et de rappel selon l'âge | Donnée personnelle soumise au RGPD : minimisation, base légale, durée de conservation |
| Zone géographique ou pays d'émission | Écarts entre régions, transactions nationales ou transfrontalières | Risque de discrimination indirecte |
| Type de carte, segment client | Écarts selon le produit | |
| Canal (en ligne, sans contact, retrait) | Écarts selon le canal | |
| Sexe | Seulement si la collecte est licite et justifiée | Usage à valider par la conformité, même pour un audit |

Ces variables ne serviraient qu'à l'audit et ne seraient pas des entrées du modèle. Leur accès devrait être limité et validé par le délégué à la protection des données.

## 5.5 Protocole proposé

1. Calculer pour chaque groupe le rappel, le FPR et la précision, avec des intervalles de confiance par bootstrap. Les fraudes étant rares, un groupe de 20 fraudes donne un intervalle d'environ ±20 points sur le rappel.
2. Mesurer les écarts entre groupes : différence et rapport des FPR, différence des rappels (critère d'égalité des chances).
3. Ne conclure que pour les groupes dont l'effectif est suffisant, et indiquer « non concluant » pour les autres.
4. Utiliser Fairlearn (`MetricFrame`) ou AI Fairness 360.
5. Fixer avec les équipes métier et conformité des seuils d'investigation, par exemple un rapport de FPR supérieur à 1,25 entre deux groupes.

## 5.6 Recommandations

Le modèle ne doit pas être présenté comme exempt de biais, puisque l'équité n'a pas pu être testée. Il faut demander à la banque un jeu d'audit enrichi avant toute mise en production. Pendant un éventuel pilote, les alertes doivent être revues par un analyste et les réclamations clients suivies par segment lorsque l'information existe. L'audit d'équité fait partie des preuves exigées avant la mise en production (rapport 08).

# 5. Note d'audit des biais et de l'équité

## 5.1 Objectif

Repérer les écarts de performance entre groupes de clients pertinents (âge, sexe, zone géographique, type de carte, segment de clientèle…). L'enjeu est double :
- un taux de faux positifs plus élevé pour un groupe signifie des blocages injustifiés plus fréquents pour ce groupe ;
- un rappel plus faible pour un groupe signifie une moindre protection contre la fraude.

## 5.2 Constat : l'analyse d'équité n'est pas réalisable sur ce jeu de données

Le jeu Credit Card Fraud Detection contient uniquement :
- `Time` : secondes écoulées depuis la première transaction ;
- `V1` à `V28` : composantes issues d'une PCA dont les variables d'origine ne sont **pas publiées** ;
- `Amount` : montant ;
- `Class` : cible.

**Il ne contient aucune caractéristique du client** : ni âge, ni sexe, ni localisation, ni nationalité, ni type de carte, ni canal de paiement. Il ne permet donc pas, à lui seul, de conclure à l'équité du système selon ces caractéristiques. **Aucune conclusion d'équité n'est tirée dans ce projet.**

## 5.3 Ce qui n'a pas été fait, volontairement

- **Aucun attribut démographique n'a été reconstruit à partir des composantes anonymisées.** Partitionner les clients selon le signe ou les quantiles de `V1`, ou en déduire un « groupe », produirait des catégories sans fondement. Interpréter les composantes comme des caractéristiques métier ferait de plus courir le risque d'une fausse conclusion, rassurante ou alarmante.
- **Le montant et l'heure ne sont pas utilisés comme substituts** d'un groupe social. Des écarts de performance selon le montant existent (rapport 02 : les fraudes manquées sont de petits montants). Ce sont des constats de performance, pas des constats d'équité.

## 5.4 Analyse possible dès maintenant : la performance par sous-population technique

On peut néanmoins comparer la performance selon des **critères opérationnels**, sans attribut protégé. Par exemple, la répartition des erreurs par tranche de montant montre que :
- les fraudes de **petit montant** sont plus souvent manquées (montant médian des fraudes manquées : 3,76, contre 31,99 pour les fraudes détectées) ;
- les **fausses alertes** touchent des montants plus élevés (montant moyen : 2 836).

Ces écarts doivent être suivis. Ils renseignent sur la qualité du service, mais ne remplacent pas une analyse d'équité.

## 5.5 Données complémentaires nécessaires

Pour une véritable analyse, il faudrait disposer, pour chaque transaction labellisée, de variables de groupe **fiables et légalement utilisables** :

| Variable | Usage | Précaution |
|---|---|---|
| Tranche d'âge du porteur | Écarts de FPR et de rappel par âge | Donnée personnelle (RGPD) : minimisation, base légale, conservation limitée |
| Zone géographique ou pays d'émission | Écarts entre régions et entre transactions domestiques ou transfrontalières | Risque de discrimination indirecte |
| Type de carte, segment client | Écarts par produit | — |
| Canal (paiement en ligne, sans contact, retrait…) | Écarts par canal | — |
| Sexe | Uniquement si la collecte est licite et justifiée | Donnée souvent non collectée ; la conformité doit valider son usage même pour un audit |

Ces variables serviraient à **l'audit uniquement**, pas comme entrées du modèle. L'accès devrait être restreint et validé par le délégué à la protection des données.

## 5.6 Protocole recommandé lorsque les données seront disponibles

1. Pour chaque groupe : **rappel, FPR, précision**, avec intervalles de confiance par bootstrap. Les fraudes sont rares : un groupe de 20 fraudes donne un intervalle de confiance du rappel d'environ ±20 points.
2. Écarts entre groupes : différence et rapport de FPR, différence de rappel (égalité des chances, *equalized odds*).
3. Ne conclure qu'à partir d'effectifs suffisants. Sinon, signaler « non concluant ».
4. Outils : **Fairlearn** (`MetricFrame`) ou **AI Fairness 360**.
5. Seuils d'alerte à définir avec les équipes métier et conformité. Par exemple : un rapport de FPR entre groupes > 1,25 déclenche une investigation.

## 5.7 Recommandations

1. Ne pas présenter le modèle comme « non biaisé » : l'équité **n'a pas été testée**, faute de données.
2. Demander à la banque un jeu d'audit enrichi (§ 5.5) avant toute mise en production.
3. Pendant un pilote, conserver une revue humaine des alertes, et suivre les réclamations clients par segment, s'il est disponible.
4. Inscrire l'audit d'équité parmi les preuves obligatoires avant la mise en production (rapport 08).

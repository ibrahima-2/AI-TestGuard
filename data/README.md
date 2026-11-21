# Données — Credit Card Fraud Detection

## Source

- Jeu de données : **Credit Card Fraud Detection** (Kaggle, `mlg-ulb/creditcardfraud`)
  — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Origine : collaboration entre Worldline et le Machine Learning Group de l'ULB.
  Transactions de porteurs de cartes européens sur deux jours (septembre 2013).
- Licence : se référer à la page Kaggle (Open Database License) avant toute
  redistribution.

Le fichier `creditcard.csv` **n'est pas versionné** (144 Mo, voir `.gitignore`).

## Échantillon versionné : `sample_creditcard.csv`

- 3 150 transactions réelles tirées du jeu dédoublonné : 3 000 normales et 150 fraudes, graine 42.
- Créé par `python -m app.cli make-sample`.
- Il sert aux tests d'intégration et de non-régression exécutés par la CI, sans télécharger le jeu complet.
- **Il est volontairement rééquilibré** (4,8 % de fraudes, contre 0,17 % dans le jeu complet). Il ne doit donc **pas** servir à estimer la performance réelle du modèle.

## Récupération

Avec un compte Kaggle (API configurée dans `~/.kaggle/kaggle.json`, jamais
commitée) :

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

Sans compte Kaggle, le même fichier est publié par TensorFlow :

```bash
curl -fsSL -o data/creditcard.csv \
  https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Vérifier l'intégrité du fichier obtenu :

```bash
shasum -a 256 data/creditcard.csv
# attendu : 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

## Caractéristiques constatées

| Élément | Valeur |
|---|---|
| Lignes | 284 807 |
| Colonnes | 31 (`Time`, `V1`…`V28`, `Amount`, `Class`) |
| Types | 30 × `float64`, `Class` en `int64` |
| Transactions normales (`Class = 0`) | 284 315 |
| Fraudes (`Class = 1`) | 492 (0,173 %) |
| Valeurs manquantes | 0 |
| Valeurs infinies | 0 |
| Doublons exacts | 1 081 (1 062 normales, 19 fraudes) |
| `Amount = 0` | 1 825 transactions |
| `Time` | de 0 à 172 792 s (48 h) |
| `Amount` max (normales / fraudes) | 25 691,16 / 2 125,87 |

## Points d'attention pour les tests

- **Déséquilibre extrême** : un modèle qui prédit toujours « normale » obtient
  99,83 % d'accuracy en ne détectant aucune fraude. L'accuracy ne peut pas
  servir de critère d'acceptation.
- **Doublons exacts** : s'ils ne sont pas traités avant le découpage, une même
  transaction peut se retrouver à la fois dans le jeu d'entraînement et dans le
  jeu de test, ce qui surestime les performances (fuite de données).
- **Variables `V1`…`V28`** : composantes issues d'une PCA, déjà centrées. Elles
  ne doivent pas être interprétées comme des caractéristiques métier.
- **`Time` et `Amount`** : non transformées par la PCA, sur des échelles très
  différentes des autres variables ; une mise à l'échelle ajustée sur le seul
  jeu d'entraînement est nécessaire.
- **Aucun attribut démographique** (âge, sexe, localisation) : le jeu ne permet
  pas d'évaluer l'équité du modèle selon ces critères (Partie VII).

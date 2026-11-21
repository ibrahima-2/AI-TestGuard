# Données : Credit Card Fraud Detection

## Source

- Jeu de données Credit Card Fraud Detection, publié sur Kaggle (`mlg-ulb/creditcardfraud`) : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Origine : collaboration entre Worldline et le Machine Learning Group de l'ULB. Transactions de porteurs de cartes européens sur deux jours, en septembre 2013.
- Licence : voir la page Kaggle (Open Database License) avant toute redistribution.

Le fichier `creditcard.csv` (144 Mo) n'est pas versionné, comme indiqué dans `.gitignore`.

## Échantillon versionné

Le fichier `sample_creditcard.csv` contient 3 150 transactions réelles tirées du jeu dédoublonné : 3 000 normales et 150 fraudes, avec la graine 42. Il est produit par `python -m app.cli make-sample` et sert aux tests d'intégration et de non-régression exécutés par la CI, sans avoir à télécharger le jeu complet.

L'échantillon est volontairement rééquilibré (4,8 % de fraudes contre 0,17 % dans le jeu complet). Il ne doit pas servir à estimer la performance réelle du modèle.

## Récupération

Avec un compte Kaggle (clé d'API dans `~/.kaggle/kaggle.json`, à ne jamais versionner) :

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

Sans compte Kaggle, le même fichier est disponible sur le stockage de TensorFlow :

```bash
curl -fsSL -o data/creditcard.csv \
  https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Vérification de l'intégrité :

```bash
shasum -a 256 data/creditcard.csv
# attendu : 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

## Caractéristiques constatées

| Élément | Valeur |
|---|---|
| Lignes | 284 807 |
| Colonnes | 31 (`Time`, `V1` à `V28`, `Amount`, `Class`) |
| Types | 30 colonnes `float64`, `Class` en `int64` |
| Transactions normales (`Class = 0`) | 284 315 |
| Fraudes (`Class = 1`) | 492 (0,173 %) |
| Valeurs manquantes | 0 |
| Valeurs infinies | 0 |
| Doublons exacts | 1 081 (1 062 normales, 19 fraudes) |
| `Amount = 0` | 1 825 transactions |
| `Time` | de 0 à 172 792 s (48 h) |
| `Amount` maximal (normales / fraudes) | 25 691,16 / 2 125,87 |

## Points d'attention pour les tests

- Le déséquilibre des classes est extrême : un modèle qui prédit toujours « normale » obtient 99,83 % d'accuracy sans détecter aucune fraude. L'accuracy ne peut donc pas servir de critère d'acceptation.
- Si les doublons exacts ne sont pas supprimés avant le découpage, une même transaction peut se trouver à la fois dans l'entraînement et dans le test, ce qui surestime les performances.
- `V1` à `V28` sont des composantes issues d'une PCA, déjà centrées. Elles ne doivent pas être interprétées comme des caractéristiques métier.
- `Time` et `Amount` n'ont pas été transformées par la PCA et ont des échelles très différentes des autres variables. Leur mise à l'échelle doit être ajustée sur l'entraînement uniquement.
- Le jeu ne contient aucun attribut démographique (âge, sexe, localisation) et ne permet donc pas d'évaluer l'équité du modèle selon ces critères (partie VII).

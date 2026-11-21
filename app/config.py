"""Configuration centrale : chemins, contrat de données et critères d'acceptation.

Les critères d'acceptation sont fixés ici à partir des exigences métier,
AVANT tout entraînement et toute observation du jeu de test. Ils ne doivent
pas être modifiés pour « faire passer » un modèle.
"""

from pathlib import Path

# --- Chemins -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FULL_DATA_PATH = DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_creditcard.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
REPORTS_DIR = ROOT / "reports"
BASELINE_DIR = REPORTS_DIR / "baseline"

# --- Reproductibilité --------------------------------------------------------
SEED = 42

# --- Contrat de données -------------------------------------------------------
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *PCA_FEATURES, "Amount"]
TARGET = "Class"

# --- Découpage : 60 % entraînement / 20 % validation / 20 % test -------------
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# --- Seuil de décision --------------------------------------------------------
# Règle : sur le jeu de VALIDATION, on retient le seuil qui maximise la précision
# parmi ceux qui garantissent un rappel >= 0,80. La marge de 5 points au-dessus
# du critère d'acceptation (0,75) absorbe la variabilité d'échantillonnage.
MIN_RECALL_FOR_THRESHOLD = 0.80

# --- Critères d'acceptation du modèle (opérateur, valeur) ---------------------
# - recall >= 0,75    : au moins 3 fraudes sur 4 détectées (pertes limitées).
# - precision >= 0,30 : au plus ~2,3 fausses alertes par fraude détectée,
#                       charge jugée supportable par une équipe d'analystes.
# - fpr <= 0,5 %      : au plus 1 client légitime sur 200 bloqué à tort.
# - pr_auc >= 0,60    : qualité globale sur la classe rare (référence naïve ~0,0017).
# - roc_auc >= 0,90   : bonne capacité de discrimination tous seuils confondus.
ACCEPTANCE_CRITERIA = {
    "recall": (">=", 0.75),
    "precision": (">=", 0.30),
    "fpr": ("<=", 0.005),
    "pr_auc": (">=", 0.60),
    "roc_auc": (">=", 0.90),
}
# Écart maximal toléré entre PR-AUC d'entraînement et de test (surapprentissage).
MAX_OVERFIT_GAP_PR_AUC = 0.10

# --- Critères opérationnels ---------------------------------------------------
# Une autorisation de paiement doit répondre en quelques centaines de ms ;
# on réserve au plus 50 ms (P95) au scoring, validation des entrées comprise.
PERFORMANCE_CRITERIA = {
    "latency_p95_ms": ("<=", 50.0),
    "throughput_tps": (">=", 1000.0),
    "model_size_mb": ("<=", 10.0),
    "memory_increase_mb": ("<=", 200.0),
}

# --- Non-régression : (sens, seuil « investiguer », seuil « rejeter ») --------
# Les écarts sont exprimés en valeur absolue de dégradation.
REGRESSION_TOLERANCES = {
    "pr_auc": ("higher", 0.01, 0.03),
    "roc_auc": ("higher", 0.005, 0.01),
    "recall": ("higher", 0.02, 0.05),
    "precision": ("higher", 0.03, 0.08),
    "fpr": ("lower", 0.0005, 0.001),
}

# --- Surveillance -------------------------------------------------------------
PSI_WARNING = 0.10
PSI_ALERT = 0.20
ALERT_RATE_MAX_RATIO = 2.0

# --- Sécurité des entrées -----------------------------------------------------
MAX_FILE_SIZE_MB = 500
MAX_BATCH_RECORDS = 10_000

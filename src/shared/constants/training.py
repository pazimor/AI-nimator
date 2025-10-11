"""Training-related constant definitions."""

from pathlib import Path


LOCAL_CLIP_RECORD_LIMIT = 100  # Utiliser 0 pour charger l'intégralité du dataset.

DEFAULT_SAVE_DIRECTORY = Path("./ckpts")  # Dossier des checkpoints par défaut.
DEFAULT_EXPERIMENT_NAME = "exp"  # Préfixe court pour nommer les expériences.
DEFAULT_EPOCH_COUNT = 5  # Nombre d'époques d'apprentissage initial.
DEFAULT_BATCH_SIZE = 2  # Taille mini-batch par défaut.
DEFAULT_LEARNING_RATE = 1e-4  # Taux d'apprentissage de départ.
DEFAULT_SEQUENCE_FRAMES = 240  # Longueur des séquences utilisées.
DEFAULT_CONTEXT_HISTORY = 0  # Nombre de clips de contexte pendant l'entraînement.
DEFAULT_CONTEXT_TRAIN_MODE = "alt"  # Stratégie d'activation du contexte.
DEFAULT_CONTEXT_TRAIN_RATIO = 0.5  # Probabilité de contexte en mode ratio.
DEFAULT_MODEL_DIMENSION = 256  # Taille des embeddings du modèle diffusion.
DEFAULT_LAYER_COUNT = 6  # Nombre de blocs temporels empilés.
DEFAULT_MOE_EXPERTS = 8  # Quantité d'experts dans le MoE.
DEFAULT_MOE_TOPK = 2  # Nombre d'experts activés par token.
DEFAULT_CHECKPOINT_INTERVAL = 500  # Fréquence de sauvegarde des checkpoints.
DEFAULT_MAX_TOKENS = 128  # Budget de tokens pour les prompts.
DEFAULT_TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Encodeur texte HF.
DEFAULT_QAT_READY = False  # Préparation quantization-aware training désactivée.
DEFAULT_RANDOM_SEED = 42  # Graine de reproductibilité.
DEFAULT_VALIDATION_SPLIT = 0.1  # Ratio de données réservé à la validation.
DEFAULT_VALIDATION_INTERVAL = 1000  # Fréquence des évaluations validation.
DEFAULT_SUCCESS_DEGREES = 5.0  # Seuil d'erreur angulaire considéré réussi.
DEFAULT_TARGET_SUCCESS_RATE = 0.90  # Taux de succès visé pour early stop.
DEFAULT_CACHE_ON_DEVICE = False  # Mise en cache VRAM désactivée par défaut.
DEFAULT_MAX_VALIDATION_SAMPLES = 256  # Taille max du lot d'évaluation.
DEFAULT_RECACHE_EVERY_EPOCH = False  # Pas de recache systématique.
DEFAULT_DEVICE_BACKEND = "auto"  # Sélection automatique du backend.

DEFAULT_SAMPLING_FRAME_COUNT = 240  # Nombre d'images générées en inférence.
DEFAULT_SAMPLING_STEPS = 12  # Étapes de diffusion en génération.
DEFAULT_SAMPLING_GUIDANCE = 2.0  # Échelle de guidance classifier-free.
DEFAULT_CONTEXT_JSON_LIST = ""  # Pas de contexte additionnel par défaut.
DEFAULT_SAMPLING_TEXT_MODEL = None  # Conserver l'encodeur défini par le checkpoint.

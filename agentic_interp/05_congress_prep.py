"""Stage 5: prep Congress dataset — subsample, embed, train SAE, save artifacts."""
import os, sys, json
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)

from hypothesaes.embedding import get_openai_embeddings
from hypothesaes.quickstart import train_sae

EMBEDDER = "text-embedding-3-small"
CACHE_NAME = f"congress_agentic_{EMBEDDER}"
OUT_DIR = "agentic_interp/artifacts_congress"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN, N_VAL = 20_000, 2_000

train_df = pd.read_json("data/congress/train.json", lines=True).sample(N_TRAIN, random_state=0)
val_df = pd.read_json("data/congress/val.json", lines=True).sample(N_VAL, random_state=0)
texts = train_df["speech_text"].tolist()
val_texts = val_df["speech_text"].tolist()
labels = train_df["republican"].astype(float).tolist()
print(f"train {len(texts)}, val {len(val_texts)}, republican share {np.mean(labels):.3f}")

text2embedding = get_openai_embeddings(texts + val_texts, model=EMBEDDER, cache_name=CACHE_NAME)
train_embeddings = np.stack([text2embedding[t] for t in texts])
val_embeddings = np.stack([text2embedding[t] for t in val_texts])
print("embeddings:", train_embeddings.shape)

sae = train_sae(
    embeddings=train_embeddings,
    val_embeddings=val_embeddings,
    M=256, K=8, matryoshka_prefix_lengths=[32, 256],
    checkpoint_dir=f"checkpoints/{CACHE_NAME}",
)

activations = sae.get_activations(train_embeddings)
np.save(os.path.join(OUT_DIR, "activations_train.npy"), activations)
np.save(os.path.join(OUT_DIR, "embeddings_train.npy"), train_embeddings)
json.dump([{"text": t, "label": lab} for t, lab in zip(texts, labels)],
          open(os.path.join(OUT_DIR, "train_texts.json"), "w"))

alive = (activations.max(axis=0) > 0).sum()
print(f"Done. Activations {activations.shape}, alive neurons: {alive}/256")

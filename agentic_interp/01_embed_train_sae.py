"""Stage 1: embed Yelp demo data, train SAE (notebook config), save activations."""
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)

from hypothesaes.embedding import get_openai_embeddings
from hypothesaes.quickstart import train_sae

EMBEDDER = "text-embedding-3-small"
CACHE_NAME = f"yelp_agentic_{EMBEDDER}"
OUT_DIR = "agentic_interp/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

train_df = pd.read_json("demo_data/yelp-demo-train-10K.json", lines=True)
val_df = pd.read_json("demo_data/yelp-demo-val-2K.json", lines=True)
texts = train_df["text"].tolist()
val_texts = val_df["text"].tolist()

text2embedding = get_openai_embeddings(texts + val_texts, model=EMBEDDER, cache_name=CACHE_NAME)
train_embeddings = np.stack([text2embedding[t] for t in texts])
val_embeddings = np.stack([text2embedding[t] for t in val_texts])
print("embeddings:", train_embeddings.shape, val_embeddings.shape)

sae = train_sae(
    embeddings=train_embeddings,
    val_embeddings=val_embeddings,
    M=256, K=8, matryoshka_prefix_lengths=[32, 256],
    checkpoint_dir=f"checkpoints/{CACHE_NAME}",
)

activations = sae.get_activations(train_embeddings)
np.save(os.path.join(OUT_DIR, "activations_train.npy"), activations)
np.save(os.path.join(OUT_DIR, "embeddings_train.npy"), train_embeddings)
train_df[["text", "stars"]].to_json(os.path.join(OUT_DIR, "train_texts.json"), orient="records")

alive = (activations.max(axis=0) > 0).sum()
print(f"Done. Activations {activations.shape}, alive neurons: {alive}/256")
print("Mean nonzero per doc:", (activations > 0).sum(axis=1).mean())

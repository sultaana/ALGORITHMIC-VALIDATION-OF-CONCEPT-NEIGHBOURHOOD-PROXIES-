from utils import load_embeddings
from baseline import run_baseline
from optimised import run_optimised

EMBEDDINGS_PATH = "/Users/mac/tinyneigh-project/outputs/features.npy"
REFERENCE_INDEX = 25

embeddings = load_embeddings(EMBEDDINGS_PATH)

print("Running baseline pipeline...")
run_baseline(embeddings, REFERENCE_INDEX)

print("Running optimised pipeline...")
run_optimised(embeddings, REFERENCE_INDEX)


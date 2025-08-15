import pandas as pd

# 1. Load metadata
meta = pd.read_csv("input/hidden/metadata.tsv", sep="\t")

# 2. Load normalized counts, index by the gene name
counts = pd.read_csv(
    "input/hidden/GSE165512_normalized_counts.tsv", sep="\t", index_col="Geneid"
)

counts = counts[~counts.index.isna()]

# 3. Identify the treatments you care about
treatments = ["UC", "CD", "Control"]

for tr in treatments:
    # 4a. Get list of samples with this treatment
    samp_list = meta.loc[meta["group"] == tr, "sample"].tolist()

    # 4b. Keep only those columns that actually exist in counts
    samp_list = [s for s in samp_list if s in counts.columns]

    if not samp_list:
        print(f"No samples found for treatment {tr}, skipping.")
        continue

    # 4c. Subset and transpose so rows = samples, cols = genes
    df = counts[samp_list].T

    # 5. Write to CSV
    out_fname = f"input/GSE165512_{tr}.csv"
    df.to_csv(out_fname)
    print(f"  → wrote {df.shape[0]} samples × {df.shape[1]} genes to {out_fname}")

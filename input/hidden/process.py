import pandas as pd

# df = pd.read_csv("TCGA_HNSC_rawcount_data.csv", index_col=0)
# df_t = df.transpose()
# df_t.to_csv("TCGA_HNSC_rawcount_data_t.csv")


# Load the CSV file
df = pd.read_csv("gene_MTB_caries.csv")

df = df.drop(df.columns[0], axis=1)
df = df.set_index("gene_family")
df_t = df.transpose()
df_t.to_csv("gene_MTB_caries_cleaned_t.csv")

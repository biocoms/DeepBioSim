import pandas as pd

# Load the CSV
df = pd.read_csv("TCGA_HNSC_rawcount_data.csv", index_col=0)

# Transpose the DataFrame
df_t = df.transpose()

# Save the transposed DataFrame
df_t.to_csv("TCGA_HNSC_rawcount_data_t.csv")

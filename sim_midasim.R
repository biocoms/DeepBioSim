library(MIDASim)
library(tidyverse)
library(parallel)

# 1. Specify your input CSV (in input/).  
#    Change this to whatever file you want to simulate from.
input_file <- "input/TCGA_HNSC_rawcount_data_t.csv"

# 2. Derive a base name for your output
input_name  <- tools::file_path_sans_ext(basename(input_file))

# 3. Read it in (assuming counts with rownames in col 1)
count_data <- read.csv(input_file, row.names = 1, check.names = FALSE)

start_time <- Sys.time()
# 4. Run the MIDASim workflow
ds_setup    <- MIDASim.setup(count_data,
                             mode        = "nonparametric",
                             n.break.ties = 10)
ds_modified <- MIDASim.modify(ds_setup,
                              lib.size        = NULL,
                              mean.rel.abund  = NULL,
                              gengamma.mu     = NULL,
                              sample.1.prop   = NULL,
                              taxa.1.prop     = NULL)
simulation  <- MIDASim(ds_modified)

end_time <- Sys.time()

print(end_time-start_time)

# 5. Prepare your output folder
output_folder <- "gen_ms"
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# 6. Write out only the count-simulation as {input_name}_ms_samples.csv
output_file <- file.path(output_folder, paste0(input_name, "_ms_samples.csv"))
write.csv(simulation$sim_count,
          file      = output_file,
          row.names = FALSE)

message("Wrote MS samples to ", output_file)
import pandas as pd

# set parameters for abs, t0norm, sort_by_start & sort_by_ch algorithm
mad_cutoff = 4  # for COG screen: 4
c_high_mad_cutoff = 4
c_low_mad_cutoff = 4
n_high_mad_cutoff = 4
n_low_mad_cutoff = 4
cn_high_mad_cutoff = 4
cn_low_mad_cutoff = 4
signal_max = 0.965  # for COG screen: 0.965
signal_min = 0.005  # for COG screen: 0.005
pre_div_time = 100
post_div_time = 100
gap_filling_size = 1  # for COG screen: 10
lineplot_max = 1  # for COG screen: 0.8
lineplot_min = 0  # for COG screen: 0.1

source_path = input("source_path: ")
plate_map = pd.read_csv(f"{source_path}/plate_map.csv")
treatment_list = plate_map[["treatment_list"]].dropna()
treatment_order = treatment_list["treatment_list"].tolist()

heatmap_col = 4

clustering_smooth_win_size = 16
system_correction = "untreated"

import logging
import os
import numpy as np
from matplotlib.colors import ListedColormap
import math
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Set_parameters import lineplot_max, lineplot_min
from Set_parameters import treatment_order
from Set_parameters import source_path
from Set_parameters import heatmap_col
from Set_parameters import clustering_smooth_win_size

# ---------------------------------------------- SET UP / OPTIONS-------------------------------------------------------
# Size of table to display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# # Input cell & reporter
# cell_reporter_list = cr_list.cell_reporter
# list_index = 0
# print("""
# cell_reporter_list:""")
# for cr in cr_list.cell_reporter:
#     print(f"     {list_index} = {cr}")
#     list_index += 1
#
# cell_reporter_input = input("input code for cell_reporter(s): ")
# if int(cell_reporter_input) > len(cell_reporter_list) - 1:
#     print("Warning: wrong cell_reporter input")
#     input("please RE-RUN")
#
# cell_reporter = cell_reporter_list[int(cell_reporter_input)]

clover_reporter = input("""
Signaling pathway of clover reporter: """)
mscarlet_reporter = input("""
Signaling pathway of mScarlet reporter: """)

data_type = "abs_sys_corr"
folder_name = "abs_sys_corr"

# Create output folder and subfolders
parent_directory = f"{source_path}/output_path_coord_{folder_name}"
plot_directory = "plots"
plot_path = os.path.join(parent_directory, plot_directory)
quadrant_dir = "quadrant"
quandrant_plot_output_path = os.path.join(plot_path, quadrant_dir)
kmeans_dir = "k_mean_clustering"
k_mean_plot_output_path = os.path.join(plot_path, kmeans_dir)
h_cluster_directory = "h_cluster"
h_cluster_output_path = os.path.join(plot_path, h_cluster_directory)
pearson_corr_directory = "pearson_corr"
pearson_corr_output_path = os.path.join(plot_path, pearson_corr_directory)
timeserieskmean_directory = "time_series_k"
timeseriesKmean_output_path = os.path.join(plot_path, timeserieskmean_directory)
os.makedirs(plot_path, exist_ok=True)
os.makedirs(quandrant_plot_output_path, exist_ok=True)
os.makedirs(k_mean_plot_output_path, exist_ok=True)
os.makedirs(h_cluster_output_path, exist_ok=True)
os.makedirs(pearson_corr_output_path, exist_ok=True)
os.makedirs(timeseriesKmean_output_path, exist_ok=True)


# Read clover and mScarlet reporter activity data from abs folder
hmap_df_c = pd.read_csv(f"{source_path}/output_{data_type}/clover/clover_all_cell.csv", index_col="track_index")
hmap_df_m = pd.read_csv(f"{source_path}/output_{data_type}/mscarlet/mscarlet_all_cell.csv", index_col="track_index")


# Melt pivot table to put all the reporter activity value in one single column
def unpivot_df(dataframe, val_name):
    dataframe2 = dataframe.iloc[:, 0: len(dataframe.columns) - 3]
    dataframe2["track_index"] = dataframe2.index
    dataframe3 = pd.melt(dataframe2, id_vars=["track_index"], var_name=["time"], value_name=val_name)
    return dataframe3


# Merge Clover and mScarlet reporter activity together using unique identifier "track_index_time"
# Drop the row with missing reporter data
# Divide experiment time into 16 slices
def merge_df(dataframe1, dataframe2):
    dataframe1["track_index_time"] = dataframe1["track_index"] + "_" + dataframe1["time"]
    dataframe2["track_index_time"] = dataframe2["track_index"] + "_" + dataframe2["time"]
    dataframe3 = pd.merge(dataframe1, dataframe2.iloc[:, 2:4], how="inner", on="track_index_time", validate="1:1")
    dataframe4 = dataframe3.drop(["track_index_time"], axis=1)
    before_drop = dataframe4.shape[0]
    dataframe4.dropna(axis=0, how="any", subset=["clover", "mscarlet"], inplace=True)
    dataframe4.rename(columns={"clover": clover_reporter, "mscarlet": mscarlet_reporter}, inplace=True)
    after_drop = dataframe4.shape[0]
    num_of_drops = before_drop - after_drop
    percentage_of_drops = round(((num_of_drops / before_drop) * 100), 2)
    logging.info(f"clover_mScarlet_merged_number of drops: {num_of_drops}")
    logging.info(f"clover_mScarlet_merged_percentage of drops: {percentage_of_drops}%")
    dataframe4[["site", "track", "cell__treatment"]] = dataframe4["track_index"].str.split(pat="_", n=2, expand=True)
    dataframe4["time"] = dataframe4["time"].astype("float64")
    # Divide time into 16 time slices
    time_increments = dataframe4["time"].max(axis=0) / 2
    # time_increments = 1
    time_list = list(range(dataframe4["time"].apply(np.ceil).astype(int).max(axis=0)))
    time_slice_list = [i * time_increments for i in time_list]
    dataframe4["time_range"] = pd.cut(dataframe4["time"], time_slice_list, include_lowest=True)
    # Standardize scaling of data for k-means, tsne and pca clustering
    scaled_data = StandardScaler().fit_transform(dataframe4.loc[:, [clover_reporter, mscarlet_reporter, "time"]])
    dataframe5 = pd.DataFrame(scaled_data, index=dataframe4.index,
                              columns=[f"{clover_reporter}_scaled", f"{mscarlet_reporter}_scaled", "time_scaled"])
    dataframe6 = dataframe4.merge(dataframe5, how="inner", left_index=True, right_index=True, validate="1:1")
    dataframe6[["cell", "treatment"]] = dataframe6["cell__treatment"].str.split(pat="__", n=1, expand=True)
    dataframe6.reset_index(drop=True, inplace=True)
    dataframe6['treatment'] = pd.Categorical(dataframe6['treatment'], categories=treatment_order, ordered=True)
    export_csv(dataframe6, "clover_mscarlet_merged", False)
    return dataframe6


# Divide dataframe into groups based on "cell__treatment" info
def groups(dataframe):
    df_groups = dataframe.groupby("cell__treatment", sort=False)
    return df_groups


# Export divided dataframe in groups
def export_gps(df_groups):
    for name, group in df_groups:
        group.to_csv(f"{parent_directory}/{name}.csv", index=False)
    return


# Export dataframe
def export_csv(dataframe, filename, index_export):
    dataframe.to_csv(f"{parent_directory}/{filename}.csv", index=index_export)
    return


############ Hierarchical clustering ###################################################################################
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def hierarchical_clustering(dataframe, reporter2):
    lead_reporter = ""
    sec_reporter = ""
    lead_color = ""
    sec_color = ""
    if reporter2 == clover_reporter:
        lead_reporter = clover_reporter
        sec_reporter = mscarlet_reporter
        lead_color = "Greens"
        sec_color = "Reds"
    elif reporter2 == mscarlet_reporter:
        lead_reporter = mscarlet_reporter
        sec_reporter = clover_reporter
        lead_color = "Reds"
        sec_color = "Greens"
    else:
        print("PLEASE define reporter!")
    dataframe.sort_values(by="treatment", inplace=True)
    lead_reporter_pivot = dataframe.pivot(index="track_index", columns="time", values=lead_reporter)
    lead_reporter_pivot_smooth = lead_reporter_pivot.rolling(window=clustering_smooth_win_size, min_periods=1,
                                                             axis=1).mean()
    z = linkage(y=lead_reporter_pivot_smooth, method="ward", metric='euclidean', optimal_ordering=True)
    c, coph_dists = cophenet(z, pdist(lead_reporter_pivot_smooth))
    print(c)
    plt.figure(figsize=(25, 10), tight_layout=True)
    sns.set_context("paper")
    plt.xlabel("track_index")
    plt.ylabel("distance")
    fancy_dendrogram(z, leaf_rotation=90, leaf_font_size=6, labels=lead_reporter_pivot_smooth.index,
                     above_threshold_color="#AAAAAA", truncate_mode='lastp', p=30, show_contracted=True,
                     annotate_above=0)
    plt.show()
    plt.close()
    max_d = float(input(f"{lead_reporter}_cluster cutoff: "))
    fancy_dendrogram(z, leaf_rotation=90, leaf_font_size=6, labels=lead_reporter_pivot_smooth.index,
                     above_threshold_color="#AAAAAA", truncate_mode='lastp', p=15, show_contracted=True,
                     annotate_above=0, max_d=max_d)
    plt.savefig(f"{h_cluster_output_path}/{lead_reporter}_dendrogram.png")  # , dpi=300)
    plt.close()
    clusters = fcluster(z, max_d, criterion='distance')
    cluster_df = pd.DataFrame({"track_index": lead_reporter_pivot.index, "cluster_id": clusters})
    lead_pivot_cluster = lead_reporter_pivot.merge(cluster_df, how="inner", left_index=True, right_on="track_index",
                                                   validate="1:1")
    lead_pivot_cluster[["site", "track", "cell__treatment"]] = lead_pivot_cluster["track_index"].str.split(pat="_",
                                                                                                           n=2,
                                                                                                           expand=True)
    lead_pivot_cluster[["cell", "treatment"]] = lead_pivot_cluster["cell__treatment"].str.split(pat="__", n=1,
                                                                                                expand=True)
    lead_melt_cluster = pd.melt(lead_pivot_cluster,
                                id_vars=["track_index", "cluster_id", "site", "track", "cell", "treatment",
                                         "cell__treatment"],
                                var_name="time", value_name=lead_reporter)
    lead_melt_cluster['treatment'] = pd.Categorical(lead_melt_cluster['treatment'], categories=treatment_order,
                                                    ordered=True)
    melt_cluster_merged = lead_melt_cluster.merge(dataframe[["track_index", "time", sec_reporter]], how="inner",
                                                  on=["track_index", "time"], validate="1:1")
    # Sort cell tracks based on 1) treatment, 2) cluster_id at t0, 3) most frequent cluster_id afterwards
    lead_melt_cluster.sort_values(by=["treatment", "cluster_id"], ascending=True, inplace=True)
    cluster_n = melt_cluster_merged["cluster_id"].nunique()
    if cluster_n < heatmap_col:
        col_n = cluster_n
    else:
        col_n = heatmap_col
    sns.set_context("paper")
    ax = sns.relplot(data=melt_cluster_merged, x="time", y=lead_reporter, kind="line", col="cluster_id",
                     hue="track_index",
                     col_wrap=col_n, legend=None, ci=None, palette=lead_color)
    ax.set(ylim=(lineplot_min, lineplot_max))
    plt.savefig(f"{h_cluster_output_path}/{lead_reporter}_cluster_patterns.png")  # , dpi=300)
    plt.close()
    ax2 = sns.relplot(data=melt_cluster_merged, x="time", y=sec_reporter, kind="line", col="cluster_id",
                      hue="track_index",
                      col_wrap=col_n, legend=None, ci=None, palette=sec_color)
    ax2.set(ylim=(lineplot_min, lineplot_max))
    plt.savefig(f"{h_cluster_output_path}/{lead_reporter}_clustered_{sec_reporter}_patterns.png")  # , dpi=300)
    plt.close()
    sns.set_context("talk")
    ax3 = sns.relplot(data=melt_cluster_merged, x="time", y=lead_reporter, kind="line", col="cluster_id",
                      hue=None,
                      col_wrap=col_n, legend="full", ci=None, palette=lead_color)
    ax3.set(ylim=(lineplot_min, lineplot_max))
    plt.savefig(f"{h_cluster_output_path}/represented_{lead_reporter}_cluster_patterns.png")  # , dpi=300)
    plt.close()
    ax4 = sns.relplot(data=melt_cluster_merged, x="time", y=sec_reporter, kind="line", col="cluster_id",
                      hue=None,
                      col_wrap=col_n, legend="full", ci=None, palette=sec_color)
    ax4.set(ylim=(lineplot_min, lineplot_max))
    plt.savefig(
        f"{h_cluster_output_path}/represented_{lead_reporter}_clustered_{sec_reporter}_patterns.png")  # , dpi=300)
    plt.close()
    cluster_counts = (lead_melt_cluster.groupby(["treatment"], sort=False)['cluster_id']
                      # cluster_counts = (lead_melt_cluster.groupby(["cluster_id"], sort=False)['treatment']
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index())
    cluster_counts[["mutation", "drug"]] = cluster_counts["treatment"].str.split(pat="_", n=1, expand=True)
    cluster_counts['treatment'] = pd.Categorical(cluster_counts['treatment'], categories=treatment_order, ordered=True)
    cluster_counts.sort_values(by=["treatment", "cluster_id"], ascending=True, inplace=True)
    mutation_n = cluster_counts["mutation"].nunique()
    drug_n = cluster_counts["drug"].nunique()
    ax2 = sns.catplot(data=cluster_counts, x="cluster_id", y="percentage", kind="bar", hue="mutation", col="drug",
                      col_wrap=drug_n)
    ax2.set(ylim=(0, 100))
    plt.savefig(f"{h_cluster_output_path}/{lead_reporter}_cluster_population_per_treatment.png")
    plt.close()
    if mutation_n > 1:
        df_groups1 = cluster_counts.groupby("drug", sort=False)
        for name, group in df_groups1:
            ax3 = sns.catplot(data=group, x="cluster_id", y="percentage", kind="bar", col="mutation",
                              col_wrap=mutation_n)
            ax3.set(ylim=(0, 100))
            plt.savefig(f"{h_cluster_output_path}/{name}_{lead_reporter}_cluster_population_per_mutation.png")
            plt.close()
    else:
        pass
    export_csv(lead_pivot_cluster, f"{lead_reporter}_hierarchical_cluster_df", False)
    return lead_pivot_cluster, melt_cluster_merged


def h_cluster_heatmap(dataframe, path_reporter, clustering_method):
    num_of_subplots = dataframe["cluster_id"].nunique()
    n_rows = math.ceil(num_of_subplots / heatmap_col)
    fig_width = ""
    fig_height = ""
    n_columns = ""
    if num_of_subplots < heatmap_col:
        fig_width = num_of_subplots * heatmap_col * 1.2
        fig_height = heatmap_col * 1.5
        n_columns = num_of_subplots
    elif num_of_subplots > heatmap_col:
        fig_width = heatmap_col * heatmap_col * 1.2
        fig_height = n_rows * heatmap_col * 1.5
        n_columns = heatmap_col
    elif num_of_subplots == heatmap_col:
        fig_width = heatmap_col * heatmap_col * 1.2
        fig_height = n_rows * heatmap_col * 1.5
        n_columns = heatmap_col
    dataframe.sort_values(by="cluster_id", ascending=True, inplace=True)
    df_group = dataframe.groupby("cluster_id", sort=False)
    i = 1
    fig = plt.figure(figsize=(fig_width, fig_height), tight_layout=True)  # width x height
    for name, group in df_group:
        work_df = group.iloc[:, :len(dataframe.columns) - 7]
        current_palette = sns.color_palette("RdBu_r", n_colors=1000)
        cmap = ListedColormap(sns.color_palette(current_palette).as_hex())
        sns.set_context("paper", font_scale=2)
        fig.add_subplot(n_rows, n_columns, i)
        title = f"cluster_{name}"
        total_cell_number = len(work_df.index)
        ax = sns.heatmap(work_df, cmap=cmap, vmin=0, vmax=1, yticklabels=False)
        ax.set_title(f"{title}\nn={total_cell_number}", fontsize="small")
        i += 1
    out_path = ""
    if clustering_method == "hierarchical":
        out_path = h_cluster_output_path
    if clustering_method == "timeserieskmean":
        out_path = timeseriesKmean_output_path
    fig.savefig(f"{out_path}/clustered_{path_reporter}_heatmap.png")  # , dpi=300)
    plt.close(fig)
    return


###################################### hierarchical clustering correlation #############################################
def hierarchical_cluster_correlation_heatmap(selected_reporter, sort1):
    if selected_reporter == clover_reporter:
        reporter2 = mscarlet_reporter
    else:
        reporter2 = clover_reporter
    dataframe1 = pd.read_csv(f"{parent_directory}/{selected_reporter}_hierarchical_cluster_df.csv")
    dataframe1.rename(columns={"cluster_id": f"{selected_reporter}_cluster_id"}, errors="raise", inplace=True)
    dataframe2 = pd.read_csv(f"{parent_directory}/{reporter2}_hierarchical_cluster_df.csv")
    dataframe2.rename(columns={"cluster_id": f"{reporter2}_cluster_id"}, errors="raise", inplace=True)
    reporter_ids_df = dataframe1.merge(
        dataframe2[["track_index", f"{reporter2}_cluster_id"]], how="inner", on="track_index",
        validate="1:1")
    ############### heatmap #####################
    num_of_subplots = reporter_ids_df["treatment"].nunique()
    n_rows = math.ceil(num_of_subplots / heatmap_col)
    fig_width = ""
    fig_height = ""
    n_columns = ""
    if num_of_subplots < heatmap_col:
        fig_width = num_of_subplots * heatmap_col * 1.2
        fig_height = heatmap_col * 1.5
        n_columns = num_of_subplots
    elif num_of_subplots > heatmap_col:
        fig_width = heatmap_col * heatmap_col * 1.2
        fig_height = n_rows * heatmap_col * 1.5
        n_columns = heatmap_col
    elif num_of_subplots == heatmap_col:
        fig_width = heatmap_col * heatmap_col * 1.2
        fig_height = n_rows * heatmap_col * 1.5
        n_columns = heatmap_col
    i = 1
    fig = plt.figure(figsize=(fig_width, fig_height), tight_layout=True)  # width x height
    reporter_ids_df["median"] = reporter_ids_df.median(axis=1)
    reporter_ids_df['treatment'] = pd.Categorical(reporter_ids_df['treatment'], categories=treatment_order,
                                                  ordered=True)
    reporter_ids_df.sort_values(by=["treatment", f"{sort1}_cluster_id", "median"],
                                ascending=[True, False, True], inplace=True, ignore_index=True)
    df_ids__group = reporter_ids_df.groupby("treatment", sort=False)
    for name, group in df_ids__group:
        work_df = group.iloc[:, :-9]
        current_palette = sns.color_palette("RdBu_r", n_colors=1000)
        cmap = ListedColormap(sns.color_palette(current_palette).as_hex())
        sns.set_context("paper", font_scale=2)  # rc={"font.size":2,"axes.labelsize":2})
        fig.add_subplot(n_rows, n_columns, i)  # row, column, position
        title = f"cluster_{name}"
        total_cell_number = len(work_df.index)
        ax = sns.heatmap(work_df, cmap=cmap, vmin=0, vmax=1, yticklabels=False)
        ax.set_title(f"{title}\nn={total_cell_number}", fontsize="small")
        i += 1
    fig.savefig(f"{h_cluster_output_path}/{selected_reporter}_mc_clustered_heatmap.png")  # , dpi=300)
    plt.close(fig)
    export_csv(reporter_ids_df, f"{reporter}_clustered_ids.csv", False)
    return


cluster_color_palette = "muted"
reporter_list = [clover_reporter, mscarlet_reporter]
unpivot_clover = unpivot_df(hmap_df_c, "clover")
unpivot_mscarlet = unpivot_df(hmap_df_m, "mscarlet")
unpivot_merge = merge_df(unpivot_clover, unpivot_mscarlet)
unpivot_gps = groups(unpivot_merge)

############## hierarchical clustering ##############################################################################
pivot_cluster_df_all = {}
for reporter in reporter_list:
    pivot_cluster, clustered_2nd_reporter_matrix = hierarchical_clustering(unpivot_merge, reporter)
    pivot_cluster_df_all[reporter] = pivot_cluster
    h_cluster_heatmap(pivot_cluster, reporter, "hierarchical")

for reporter in reporter_list:
    hierarchical_cluster_correlation_heatmap(reporter, mscarlet_reporter)

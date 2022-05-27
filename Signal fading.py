import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

nuc_signal = pd.read_csv(
    "C:/Users/tsangsa/Box/Work/GM Lab/My Publications/Bioreporter paper/Figures/Figure data/H210416_k562_erk_akt/Nuc_signal_fading/H190626nuclei.csv")
clover_signal = pd.read_csv(
    "C:/Users/tsangsa/Box/Work/GM Lab/My Publications/Bioreporter paper/Figures/Figure data/H210416_k562_erk_akt/Nuc_signal_fading/H190626Clover_Cell.csv")
mscarlet_signal = pd.read_csv(
    "C:/Users/tsangsa/Box/Work/GM Lab/My Publications/Bioreporter paper/Figures/Figure data/H210416_k562_erk_akt/Nuc_signal_fading/H190626mScarlet_cell.csv")


# cleanup column header
def clean_header(dataframe):
    dataframe2 = dataframe
    dataframe2.columns = dataframe2.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
                                                                                                      '').str.replace(
        ')', '')
    return dataframe2


def signal_fading(dataframe, channel):
    if channel == "farred":
        signal_col = "intensity_meanintensity_farred"
        color = "violet"
    elif channel == "clover":
        signal_col = "intensity_meanintensity_clover"
        color="green"
    else:
        signal_col = "intensity_meanintensity_mscarlet"
        color="red"

    dataframe["time_min"] = dataframe["metadata_time"] * 15
    drop_time = dataframe[dataframe["time_min"] > 1500].index
    dataframe.drop(drop_time, axis=0, inplace=True)
    mad_c = stats.median_abs_deviation(x=dataframe[signal_col], scale="normal")
    c_median = dataframe[signal_col].median()
    c_median_high = c_median + (mad_c * 3)
    c_median_low = c_median - (mad_c * 3)
    sns.set_context("talk")
    sns.displot(dataframe[signal_col])
    plt.yscale('log')
    plt.axvline(x=c_median_high, ymax=(1 - 0.05), ls='--', linewidth=2, color='red')
    plt.axvline(x=c_median_low, ymax=(1 - 0.05), ls='--', linewidth=2, color='red')
    plt.close()
    outliers_c = dataframe[(abs(
        dataframe[signal_col] - c_median) / mad_c > 3)].index
    dataframe.drop(outliers_c, axis=0, inplace=True)
    maximum = dataframe[signal_col].max()
    minimum = dataframe[signal_col].min()
    dataframe[signal_col] = (dataframe[signal_col] - minimum) / (maximum - minimum)
    # Create helper consecutive groups series with Series.ne, Series.shift and Series.cumsum and then filter by boolean indexing
    n = 800
    dataframe = dataframe[dataframe["time_min"].ne(dataframe["time_min"].shift()).cumsum() <= n]
    sns.set_context("talk")
    sns.relplot(data=dataframe, x="time_min", y=signal_col, kind="line", color=color, ci="sd")
    plt.ylim(0, 1)
    plt.savefig(f"C:/Users/tsangsa/Box/Work/GM Lab/My Publications/Bioreporter paper/Figures/Figure data/H210416_k562_erk_akt/Nuc_signal_fading/{signal_col}_lineplot.png")
    plt.close()
    sns.set_context("paper")
    sns.catplot(data=dataframe, x="time_min", y=signal_col, kind="box", color=color, showfliers=False)
    plt.ylim(0, 1)
    plt.savefig(
        f"C:/Users/tsangsa/Box/Work/GM Lab/My Publications/Bioreporter paper/Figures/Figure data/H210416_k562_erk_akt/Nuc_signal_fading/{signal_col}_boxplot.png")
    plt.close()
    return



nuc_clean = clean_header(nuc_signal)
clover_clean = clean_header(clover_signal)
mscarlet_clean = clean_header(mscarlet_signal)
signal_fading(nuc_clean, "farred")
signal_fading(clover_clean, "clover")
signal_fading(mscarlet_clean, "mscarlet")
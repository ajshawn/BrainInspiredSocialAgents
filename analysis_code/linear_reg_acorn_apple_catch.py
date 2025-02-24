import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def plot_linear_regression(
    df,
    x_col,
    y_col,
    x_label=None,
    y_label=None,
    title=None,
    outfile="regplot.png"
):
    """
    Plots a scatter + linear fit (with 95% CI) between df[x_col] and df[y_col].
    Also prints Pearson correlation and p-value.
    Saves the figure to outfile.
    """
    # Drop rows that are NaN in x or y
    sub = df[[x_col, y_col]].dropna()

    # If there's not enough data, skip
    if sub.shape[0] < 2:
        print(f"Not enough data for {x_col} vs {y_col}")
        return

    # Calculate Pearson correlation
    corr, pval = stats.pearsonr(sub[x_col], sub[y_col])
    print(f"{x_col} vs {y_col}: correlation={corr:.3f}, p-value={pval:.3g}")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.regplot(
        x=sub[x_col],
        y=sub[y_col],
        ci=95,  # 95% confidence interval
        scatter_kws={"alpha": 0.6, "label": "Data points"},
        line_kws={"label": f"Best fit (r={corr:.2f})"}
    )

    # Customize labels, title, legend
    plt.xlabel(x_label if x_label else x_col)
    plt.ylabel(y_label if y_label else y_col)
    if not title:
        title = f"{y_col} vs. {x_col}"
    plt.title(title + f"\nPearson r={corr:.2f}, p={pval:.3g}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

def main():
    # 1) Read your special comparison CSV
    df = pd.read_csv("./results/special_comparison_non_perturbed.csv")

    # EXAMPLE A: cp9651_AppSP
    #   Show that catch correlates with apple but not with acorn, etc.
    x_col_apple_cp   = "apple cp9651_AppSP"
    y_col_apple_cp   = "catch cp9651_AppSP"
    plot_linear_regression(
        df,
        x_col=x_col_apple_cp,
        y_col=y_col_apple_cp,
        x_label="Apple (cp9651 Apple specialist)",
        y_label="Catch (cp9651 Apple specialist)",
        title="cp9651: Apple vs. Catch",
        outfile="cp9651_apple_vs_catch.png"
    )

    # If you want the acorn scenario for cp9651:
    x_col_acorn_cp = "acorn cp9651_AcoSP"
    y_col_acorn_cp = "catch cp9651_AcoSP"
    plot_linear_regression(
        df,
        x_col=x_col_acorn_cp,
        y_col=y_col_acorn_cp,
        x_label="Acorn (cp9651 acorn specilaist)",
        y_label="Catch (cp9651 acorn specilaist)",
        title="cp9651: Acorn vs. Catch",
        outfile="cp9651_acorn_vs_catch.png"
    )

    # EXAMPLE B: AH_AppSP
    #   If you have "apple AH_AppSP" and "catch AH_AppSP" columns:
    x_col_apple_ah   = "apple AH_AppSP"
    y_col_apple_ah   = "catch AH_AppSP"
    plot_linear_regression(
        df,
        x_col=x_col_apple_ah,
        y_col=y_col_apple_ah,
        x_label="Apple (AH AppSP)",
        y_label="Catch (AH AppSP)",
        title="AH: Apple vs. Catch",
        outfile="AH_apple_vs_catch.png"
    )

    # If you also have "acorn AH_goodpred_acorn" in your CSV:
    # (If it doesn't exist, you'll need to define that scenario in your data pipeline)
    # x_col_acorn_ah = "acorn AH_goodpred_acorn"
    # y_col_acorn_ah = "catch AH_goodpred_acorn"
    # plot_linear_regression(
    #     df,
    #     x_col=x_col_acorn_ah,
    #     y_col=y_col_acorn_ah,
    #     x_label="Acorn (AH goodpred_acorn)",
    #     y_label="Catch (AH goodpred_acorn)",
    #     title="AH: Acorn vs. Catch",
    #     outfile="AH_acorn_vs_catch.png"
    # )

if __name__ == "__main__":
    main()

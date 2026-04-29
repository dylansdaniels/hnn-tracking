# issues_viewer.py

# %% [markdown] -----------------------------------------------------------
# Setup
# -------------------------------------------------------------------------

import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display

sns.set_palette("Set2")

# function to style html tables
# ------------------------------
def render_html_table(df):
    display(HTML(df.to_html(classes="wrapped-table", index=False)))
    display(
        HTML("""
    <style>
    .wrapped-table {
        table-layout: fixed;
        width: auto;
        word-wrap: break-word;
    }
    .wrapped-table td, .wrapped-table th {
        white-space: normal;
    }
    </style>
    """)
    )


# function to peek at head and tail
# ------------------------------
def peek(df, n=1):
    head = df.head(n)
    tail = df.tail(n)
    out = pd.concat([head, tail])
    return out

def build_report_tables_from_pickle(
    report_data,
    display_tables=True,
    style_displayed_tables=True,
):
    """
    Build report tables from saved metrics
    """
    df = report_data.copy()

    tables = {}

    for metric, group in df.groupby("metric"):
        rename_val = group["value_type"].iloc[0]
        rename_subval = group["sub_value_type"].iloc[0]
        rename_indicator = group["indicator_name"].iloc[0]

        rename_val = rename_val.replace("_", " ").title()
        rename_subval = rename_subval.replace("_", " ").title()
        rename_indicator = rename_indicator.replace("_", " ").title()

        title = group["metric"].iloc[0]
        title = title.replace("_", " ").title()

        # drop columns not needed for display
        table = group.drop(
            columns=[
                "report_date",
                "start_date",
                # "metric",
                "value_type",
                "sub_value_type",
                "grant_year",
            ]
        )
        table = table.rename(
            columns={
                "value": rename_val,
                "sub_value": rename_subval,
                "indicator_value": rename_indicator,
            }
        )

        drops = ["NA", "Na", "indicator_name"]

        for col in drops:
            if col in table.columns:
                table = table.drop(columns=col)

        tables[metric] = table.reset_index(drop=True)

        table = table.drop(columns=["metric"])

        if display_tables:
            print(f"{title}")
            if style_displayed_tables:
                render_html_table(table)
            else:
                display(table)

    return tables

# %% ----------------------------------------
# Define report-specific visualizations
# -------------------------------------------


# barplot of counts
# ----------------------------
def barplot_counts(
    report_data,
    metrics=None,
    value_col="value",
):
    df = report_data.copy()

    if metrics is None:
        metrics = ["issues_status"]

    for metric in metrics:
        yearly_data = df[df["metric"] == metric].copy()
        # yearly_data = yearly_data[yearly_data["grant_year"] != "all_time"]

        yearly_data["grant_year"] = yearly_data["grant_year"].replace(
            {"all_time": "Overall"}
        )

        if yearly_data.empty:
            continue

        pivot_table = yearly_data.pivot_table(
            index="grant_year",
            columns="indicator_value",
            values=value_col,
            aggfunc="first",
        ).sort_index()

        # change bar order to: new issues, closed issues, outstanding issues
        if metric == "issues_status":
            pivot_table = pivot_table.reindex(
                columns=[
                    "New Issues",
                    "Closed Issues",
                    "Outstanding Issues",
                ]
            )

        ax = pivot_table.plot(
            kind="bar",
            figsize=(9, 5),
            width=0.8,
        )
        ax.set_title(
            f"{metric.replace('_', ' ').title()}",
            fontsize=14,
        )
        ax.set_xlabel(
            "",
            fontsize=12,
        )
        ax.set_ylabel(
            "Count",
            fontsize=12,
        )
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )
        ax.legend(
            # bbox_to_anchor=(1.05, 1),
            loc="upper right",
        )
        plt.xticks(
            rotation=0,
        )

        # data labels
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.annotate(
                    f"{int(height)}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.show()


# Stacked bar charts for TTR metrics
# ---------------------------------
def barplot_stacked(
    report_data,
    metrics=None,
    value_col="value",
):
    df = report_data.copy()

    if metrics is None:
        metrics = ["opened_by_dev_status"]

    for metric in metrics:
        metric_data = df[df["metric"] == metric]
        # remove "all_time" and "total" rows
        yearly_data = metric_data[
            (metric_data["grant_year"] != "all_time")
            & (metric_data["indicator_value"].str.lower() != "total")
        ]

        if yearly_data.empty:
            continue

        # pivot table to hold the counts
        pivot_table = yearly_data.pivot_table(
            index="grant_year",
            columns="indicator_value",
            values=value_col,
            aggfunc="first",
        ).sort_index()

        # table to hold percents
        pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

        ax = pivot_table_percent.plot(
            kind="bar",
            stacked=True,
            figsize=(9, 5),
            colormap="Set2",
        )

        title_text = metric.replace(
            "_",
            " ",
        ).title()
        title_text = title_text.replace(
            "Dev",
            "Developer",
        )
        ax.set_title(
            f"Percent Issues {title_text}",
            fontsize=14,
        )
        ax.set_xlabel(
            "",
            fontsize=12,
        )
        ax.set_ylabel(
            "Percent",
            fontsize=12,
        )
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )
        ax.legend(
            # bbox_to_anchor=(1.05, 1),
            loc="lower right",
        )
        plt.xticks(rotation=0)

        # add data labels, zipping the percents with the counts
        for i, (row_pct, row_count) in enumerate(
            zip(
                pivot_table_percent.values,
                pivot_table.values,
            )
        ):
            cumulative = 0
            for j, (pct, count) in enumerate(
                zip(
                    row_pct,
                    row_count,
                )
            ):
                if not pd.isna(pct) and pct > 0:
                    ax.text(
                        i,
                        cumulative + pct / 2,
                        f"{pct:.1f}%\n(n={int(count)})",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )
                    cumulative += pct

        plt.tight_layout()
        plt.show()


# lineplot for time-to-response metrics
# ---------------------------------
def lineplot_fast_response(
    report_data,
    grant_years=[
        "year 1",
        "year 2",
        "year 3",
    ],
):
    df = report_data.copy()

    # filter to TTR metrics
    ttr_metrics = df[
        df["metric"].isin(
            [
                "overall_time_to_respond",
                "nondev_time_to_respond",
            ]
        )
    ]
    if ttr_metrics.empty:
        print("No TTR data available.")
        return

    # get counts from opened_by_dev_status
    counts_df = df[df["metric"] == "opened_by_dev_status"]

    plot_data = []

    for metric, label in zip(
        ["overall_time_to_respond", "nondev_time_to_respond"],
        ["All Issue", "Non-Developer Issues"],
    ):
        subset = ttr_metrics[ttr_metrics["metric"] == metric]

        # keep only the specified grant_years
        subset = subset[subset["grant_year"].isin(grant_years)]

        for _, row in subset.iterrows():
            if row["indicator_value"] == "< 02 days":
                percent = row["value"]

                indicator_value_map = {
                    "overall_time_to_respond": "Total",
                    "nondev_time_to_respond": "Non-Developer",
                }
                count_row = counts_df[
                    (counts_df["grant_year"] == row["grant_year"])
                    & (counts_df["indicator_value"] == indicator_value_map[metric])
                ]
                count = count_row["value"].values[0] if not count_row.empty else None

                plot_data.append(
                    {
                        "grant_year": row["grant_year"],
                        "percent_fast_response": float(percent),
                        "count": int(count) if pd.notna(count) else None,
                        "group": label,
                    }
                )

    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values("grant_year")

    fig, ax = plt.subplots(figsize=(8, 5))
    for group, group_df in plot_df.groupby("group"):
        x = group_df["grant_year"]
        y = group_df["percent_fast_response"]
        ax.plot(
            x,
            y,
            marker="o",
            label=group,
        )

        # add data labels
        for xi, yi, count in zip(x, y, group_df["count"]):
            if count is not None:
                ax.text(
                    xi,
                    yi + 9,
                    f"{yi:.0f}%\nn={count}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="black",
                )

    ax.set_title(
        "% Issues with Response Time < 2 Business Days",
        fontsize=14,
        pad=40,
    )
    ax.set_ylabel(
        "Percent",
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )
    ax.legend()
    plt.tight_layout()
    plt.show()




# longitudinal counts
# -------------------------------------------
def plot_longitudinal_counts(
    report_data,
    metric_period="monthly",  # accepts 'monthly' or 'rolling_monthly'
    value_col="value",
):
    """ """
    df = report_data.copy()

    # filter 'issue_staus' for specific period
    df = df[(df["metric_period"] == metric_period) & (df["metric"] == "issues_status")]

    if df.empty:
        print(f"No data found for period: {metric_period}")
        return

    # ensure value column is numeric
    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    # determine date column
    # - use 'report_date' for rolling
    # - use 'start_date' for monthly
    date_col = "report_date" if "rolling" in metric_period else "start_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    pivot_table = df.pivot_table(
        index=date_col,
        columns="indicator_value",
        values=value_col,
        aggfunc="first",
    )

    desired_order = ["New Issues", "Closed Issues", "Outstanding Issues"]
    cols = [c for c in desired_order if c in pivot_table.columns]
    pivot_table = pivot_table[cols]

    import matplotlib.dates as mdates

    ax = pivot_table.plot(
        kind="line",
        figsize=(12, 6),
        linewidth=2,
    )

    title_prefix = "Rolling " if "rolling" in metric_period else "Monthly "

    # set ticks for start, end, and each year in between
    dates = pivot_table.index
    start_date = dates.min()
    end_date = dates.max()
    years = pd.date_range(
        start=start_date,
        end=end_date,
        freq="YS",
    )
    ticks = sorted(list(set([start_date, end_date] + list(years))))

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # add data labels at the ticks
    for col in pivot_table.columns:
        series = pivot_table[col]

        # find closest indices to the ticks
        nearest_idxs = series.index.get_indexer(ticks, method="nearest")

        for i, idx in enumerate(nearest_idxs):
            date_val = series.index[idx]
            y_val = series.iloc[idx]

            # only label if date is reasonably close to the tick
            if abs((ticks[i] - date_val).days) < 45:
                ax.annotate(
                    f"{int(y_val)}",
                    (date_val, y_val),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )

    ax.set_title(
        f"{title_prefix}Issue Volume",
        fontsize=16,
    )
    ax.set_xlabel(
        "",
        fontsize=12,
    )
    ax.set_ylabel(
        "Count",
        fontsize=12,
    )
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    ax.legend()

    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.show()


def plot_longitudinal_ttr(
    report_data,
    metric_period="monthly",  # accepts 'monthly' or 'rolling_monthly'
    target_bin="< 02 days",
):
    """ """
    df = report_data.copy()

    # filter for TTR metrics and specific period
    ttr_metrics = [
        "overall_time_to_respond",
        "nondev_time_to_respond",
    ]
    df = df[
        (df["metric_period"] == metric_period)
        & (df["metric"].isin(ttr_metrics))
        & (df["indicator_value"] == target_bin)
    ]

    if df.empty:
        print(f"No TTR data found for period: {metric_period}")
        return

    # determine date column based on metric_period
    date_col = "report_date" if "rolling" in metric_period else "start_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    pivot_df = df.pivot_table(
        index=date_col,
        columns="metric",
        values="value",
    )

    col_map = {
        "overall_time_to_respond": "All Issues",
        "nondev_time_to_respond": "Non-Dev Issues",
    }
    pivot_df = pivot_df.rename(columns=col_map)

    # set ticks for start, end, and each year in between
    dates = pivot_df.index
    start_date = dates.min()
    end_date = dates.max()
    years = pd.date_range(
        start=start_date,
        end=end_date,
        freq="YS",
    )
    ticks = sorted(list(set([start_date, end_date] + list(years))))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
    )

    targets = [
        "All Issues",
        "Non-Dev Issues",
    ]
    colors = [
        "#1f77b4",
        "#ff7f0e",
    ]

    for i, (target, color) in enumerate(zip(targets, colors)):
        if target in pivot_df.columns:
            ax = axes[i]

            ax.plot(
                pivot_df.index,
                pivot_df[target],
                # marker="o",
                linewidth=2,
                label=target,
                color=color,
            )

            # add data labels
            series = pivot_df[target]
            # find closest indices to the ticks
            nearest_idxs = series.index.get_indexer(ticks, method="nearest")

            for tick_i, idx in enumerate(nearest_idxs):
                date_val = series.index[idx]
                y_val = series.iloc[idx]

                # only add label if date is reasonably close to the tick
                if abs((ticks[tick_i] - date_val).days) < 45:
                    ax.annotate(
                        f"{y_val:.1f}%",
                        (date_val, y_val),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=12,
                    )

            ax.set_ylabel(
                "Percent",
                fontsize=12,
            )
            ax.set_ylim(0, 105)
            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.4,
            )
            ax.legend(loc="upper left")

            # set title on top plot only
            if i == 0:
                title_prefix = "Rolling " if "rolling" in metric_period else "Monthly "
                ax.set_title(
                    f"{title_prefix}Percent Responded {target_bin}",
                    fontsize=16,
                )

    # set ticks to the bottom axis only
    import matplotlib.dates as mdates

    axes[-1].set_xticks(ticks)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].set_xlabel(
        "Date",
        fontsize=12,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.show()


def view_issues_metrics(
        end_date=False,
        report_name="alltime_issues_report.pkl",
    ):
    if end_date is False:
        end_date = str(pd.Timestamp.now().date())

    # ---------------------------------
    # all time data for the current report date
    # ---------------------------------
    with open(
        os.path.join(
            "issues_metrics",
            "report_data",
            report_name,
        ),
        "rb",
    ) as f:
        report_data = pickle.load(f)

    display(
        report_data[report_data["report_date"] == str(pd.to_datetime(end_date).date())]
    )

    # ---------------------------------
    # U24 report data
    # ---------------------------------
    print(
        "\n"
        "====================================\n"
        "======== U24 Report Outputs ========\n"
        "====================================\n"
    )

    u24_report_path = os.path.join(
        "issues_metrics",
        "report_data",
        "u24_issues_report.pkl"
    )
    with open(u24_report_path, "rb") as f:
        u24_report_data = pickle.load(f)

    # generate tables for each year
    # ---------------------------------
    for year in u24_report_data["grant_year"].unique():
        print(f"\n--- U24 Tables for {year} ---\n")
        year_data = u24_report_data[u24_report_data["grant_year"] == year]
        build_report_tables_from_pickle(
            year_data,
            display_tables=True,
            style_displayed_tables=True,
        )

    # generate U24 plots
    # ---------------------------------
    barplot_counts(u24_report_data)
    barplot_stacked(u24_report_data)
    lineplot_fast_response(u24_report_data)

    print(
        "\n"
        "========================================\n"
        "======== END U24 Report Outputs ========\n"
        "========================================\n"
    )

    # ---------------------------------
    # monthly and rolling visualizations
    # ---------------------------------

    print(
        "\n"
        "========================================\n"
        "======== Monthly Report Outputs ========\n"
        "========================================\n"
    )

    monthly_report_path = os.path.join(
        "issues_metrics",
        "report_data",
        "monthly_issues_report.pkl",
    )

    if os.path.exists(monthly_report_path):
        with open(monthly_report_path, "rb") as f:
            monthly_data = pickle.load(f)

        # monthly data
        plot_longitudinal_counts(
            monthly_data,
            metric_period="monthly",
        )
        plot_longitudinal_ttr(
            monthly_data,
            metric_period="monthly",
        )

        # rolling monthly data
        plot_longitudinal_counts(
            monthly_data,
            metric_period="rolling_monthly",
        )
        plot_longitudinal_ttr(
            monthly_data,
            metric_period="rolling_monthly",
        )

    print(
        "\n"
        "============================================\n"
        "======== End Monthly Report Outputs ========\n"
        "============================================\n"
    )

if __name__ == "__main__":
    # Specify the end_date if viewing a historical report; otherwise uses today
    view_issues_metrics()

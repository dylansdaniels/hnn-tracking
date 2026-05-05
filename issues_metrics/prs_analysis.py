# prs_analysis.py

# %% [markdown] -----------------------------------------------------------
# Setup
# -------------------------------------------------------------------------

# %%

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd

from issues_metrics.issues_analysis import (
    _assign_ttr_date,
    _calc_business_hours_elapsed,
    _identify_self_vs_external_responses,
    _segment_data_on_comments,
    process_datetime,
)

DATAPATH = os.path.join("issues_metrics", "raw_prs_data.pkl")

# %% [markdown] -----------------------------------------------------------
# # Data preprocessing
# -------------------------------------------------------------------------

# %%


def preprocess(
    df,
    dev_usernames,
    start_date=False,
    end_date=False,
):
    """
    Prepare PR data for analysis

    This function formats date fields, labels users as developers or not,
    removes unnecessary rows, and (optionally) filters the data by a date range

    Args:
        df: DataFrame with PRs pulled from download_prs.py
        dev_usernames: list of hnn developers
        start_date: optional start-date filter
        end_date: optional end-date filter

    Returns:
        Preprocessed DataFrame for generating metrics reports
    """
    df = df.copy()

    df = process_datetime(
        df,
        [
            "date_time",  # -> datetime_opened
            "date_closed",  # -> datetime_closed
            "date_merged",  # -> datetime_merged
            "comment_date",  # -> comment_datetime
        ],
    )

    # filter on start_date
    # ------------------------------
    if isinstance(pd.to_datetime(start_date), pd.Timestamp):
        df = df.loc[df["date_opened"] >= pd.to_datetime(start_date).date()]

    def assign_dev(row):
        if row in dev_usernames:
            return "Developer"
        else:
            return "Non-Developer"

    df["opened_by"] = df["username"].apply(lambda x: assign_dev(x))

    # drop rows where username is
    # "github-actions[bot]"
    # ------------------------------
    df = df[df["username"] != "github-actions[bot]"]
    df = df.reset_index(drop=True)

    # adjust report for the specified "end_date",
    # removing any dates after "end_date"
    # ------------------------------
    if isinstance(pd.to_datetime(end_date), pd.Timestamp):
        # Create date object and also timestamp object (for safe comparison)
        end_date = pd.to_datetime(end_date).date()
        end_ts = pd.to_datetime(end_date)

        # remove prs opened after end_date
        # ------------------------------
        # - get pr numbers
        prs_to_remove = df.loc[df["date_opened"] >= end_date]["number"].unique()

        # - remove prs based on number
        df = df.loc[~df["number"].isin(prs_to_remove)].reset_index(drop=True)

        # clear fields for prs closed
        # after the end_date
        # ------------------------------
        # Note: need to compare timestampts to handle NaTs properly
        invalid_dateclosed = df.loc[pd.to_datetime(df["datetime_closed"]) >= end_ts][
            "number"
        ].unique()

        for pr_num in invalid_dateclosed:
            # set date_closed and datetime_closed to NaT
            df.loc[
                df["number"] == pr_num,
                ["date_closed", "datetime_closed"],
            ] = pd.NaT

        # clear fields for prs merged
        # after the end_date
        # ------------------------------
        invalid_datemerged = df.loc[pd.to_datetime(df["datetime_merged"]) >= end_ts][
            "number"
        ].unique()

        for pr_num in invalid_datemerged:
            # set date_merged and datetime_merged to NaT
            df.loc[
                df["number"] == pr_num,
                ["date_merged", "datetime_merged"],
            ] = pd.NaT

        # clear fields for *only* the
        # comments made after the end_date
        # ------------------------------
        # Note: need to compare timestamps to handle NaTs properly
        invalid_commentdate = pd.to_datetime(df["comment_datetime"]) >= end_ts

        # set comment_date and comment_datetime to NaT
        # set comment_username and comment_contents to ""
        df.loc[
            invalid_commentdate,
            [
                "comment_date",
                "comment_datetime",
                "comment_username",
                "comment_contents",
            ],
        ] = [pd.NaT, pd.NaT, "", ""]

    # order columns
    # ------------------------------
    df = df[
        [
            "number",
            # "labels",
            # "milestone",
            "date_opened",
            "datetime_opened",
            "opened_by",
            "username",
            "pr_title",
            "pr_url",
            "state",
            "date_closed",
            "datetime_closed",
            "date_merged",
            "datetime_merged",
            "comment_date",
            "comment_datetime",
            "comment_username",
            "comment_contents",
        ]
    ]

    return df


def process_prs_for_ttr(
    df,
    report_date,
):

    """
    This function segments PRs by response status and calculates
    time-to-response (TTR) measures

    We apply one of the following labels to each PR:
        - "no response": no comments exist on the PR
        - "self comment": only the author has commented
        - "external comment": someone other than the author has commented

    We then identify the first valid response timestamp, determines the appropriate
    cutoff date (response date or report date for open PR), and calculate
    the total business hours elapsed (excluding weekends and holidays)

    Args:
        df (pd.DataFrame): preprocessed PR data
        report_date (datetime.date or str): The cutoff date for the report, used
            as the reference point for open PRs without responses.

    Returns:
        pd.DataFrame: A DataFrame containing segmented PRs with added columns:
            - "status": the response status of the PR as of the report date
            - "ttr_date": the timestamp used for the TTR metric
            - "ttr_hours": business hours elapsed from PR open to response
            - "ttr_days": days elapsed from PR open to response
    """

    # unique non-bot prs
    # ------------------------------
    unique_prs = df.drop_duplicates(
        [
            "pr_title",
            "date_opened",
            "username",
        ]
    )

    # split into dataframes for records with/without responses
    # ------------------------------------------------------------
    no_response, with_response = _segment_data_on_comments(unique_prs)

    # split with_response into dataframes with self/external responses
    # ------------------------------------------------------------
    self_response, external_response = _identify_self_vs_external_responses(
        with_response
    )

    # confirm pr counts are correct after segmentation
    # ------------------------------------------------------------
    if not len(self_response["number"]) + len(external_response["number"]) + len(
        no_response["number"]
    ) == len(unique_prs):
        raise ValueError(
            "Number of unique prs has changed after segmentation,"
            " which indicates a problem with the data processing."
            " Please check the code and try again."
        )
    else:
        pass

    # calculate time-to-respond metric
    # ------------------------------

    prs_segmented = pd.concat(
        [
            df
            for df in [
                no_response,
                self_response,
                external_response,
            ]
            if not df.empty
        ],
        ignore_index=True,
    )

    prs_segmented = prs_segmented.sort_values("number", ascending=False)

    prs_segmented["ttr_date"] = prs_segmented.apply(
        lambda x: _assign_ttr_date(
            x,
            report_date,
        ),
        axis=1,
    )

    prs_segmented = _calc_business_hours_elapsed(prs_segmented)
    prs_segmented["ttr_days"] = round(prs_segmented["ttr_hours"] / 24, 2)

    return prs_segmented


# %% ----------------------------------------
# Metrics
# -------------------------------------------

# %%


def pr_status_counts(
    data,
):
    """
    This function generates a table of PRs based on status
    """
    df = data.copy()
    df = df[
        [
            "number",
            "date_closed",
            "date_merged",
        ]
    ]
    df = df.drop_duplicates().reset_index(drop=True)

    total = len(df["number"])

    if total == 0:
        return pd.DataFrame(
            {
                "PR Status": [
                    "Opened",
                    "Closed",
                    "Merged",
                ],
                "Count": [0, 0, 0],
                "Percent": [0.0, 0.0, 0.0],
            }
        )

    closed_prs = df["date_closed"].notna().sum()
    merged_prs = df["date_merged"].notna().sum()

    table = pd.DataFrame(
        {
            "PR Status": [
                "Opened",
                "Closed",
                "Merged",
            ],
            "Count": [
                total,
                closed_prs,
                merged_prs,
            ],
            "Percent": [
                100,
                round(closed_prs / total * 100, 2),
                round(merged_prs / total * 100, 2),
            ],
        }
    )

    return table


def prs_opened_by_users(
    df,
    by_dev_status=False,
    return_df=False,
):
    """
    This function generates a table of PRs opened by username
    """
    if by_dev_status:
        by_col = "opened_by"
    else:
        by_col = "username"

    prs_by_user = df.drop_duplicates(subset=["number"])[
        [
            "pr_title",
            "date_opened",
            by_col,
        ]
    ]

    prs_by_user = prs_by_user.groupby(by_col).count().reset_index()
    prs_by_user = prs_by_user[
        [
            by_col,
            "pr_title",
        ]
    ].rename(
        columns={
            "pr_title": "prs_opened",
        }
    )

    table = pd.concat(
        [
            prs_by_user,
            pd.DataFrame(
                {
                    by_col: ["Total"],
                    "prs_opened": [prs_by_user["prs_opened"].sum()],
                }
            ),
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    if return_df:
        return prs_by_user

    return table

# %% ---------------------------
# Time To Response Table
# ------------------------------


def generate_ttr_table(data):
    df = data.copy()

    def ttr_indicator(row):
        days = round(row["ttr_hours"] / 24, 2)

        if days <= 2:
            return 0
        elif (days > 2) and (days <= 14):
            return 1
        elif (days > 14) and (days <= 30):
            return 2
        elif (days > 30) and (days <= 90):
            return 3
        elif days > 90:
            return 4
        else:
            raise (
                ValueError(
                    f"Invalid input: {days} cannot be mapped to the specified bins."
                )
            )

    ttr_bins = {
        0: "< 02 days",
        1: "03 - 14 days",
        2: "15 - 30 days",
        3: "31 - 90 days",
        4: "> 90 days",
    }

    df["ttr_indicator"] = df.apply(lambda x: ttr_indicator(x), axis=1)

    def percent_bins_table(df, indicator_column="ttr_indicator"):
        if indicator_column not in df.columns:
            raise ValueError(
                f"Columns {indicator_column} not found in dataframe columns"
            )

        ttr_percent_bins = round(
            df[indicator_column].value_counts() / len(df[indicator_column]) * 100, 2
        )

        ttr_percent_bins = (
            ttr_percent_bins.reset_index()
            .sort_values(indicator_column)
            .reset_index(drop=True)
        )

        ttr_percent_bins = ttr_percent_bins.rename(columns={"count": "percent"})

        ttr_percent_bins["bins"] = ttr_percent_bins[indicator_column].map(ttr_bins)

        ttr_percent_bins = ttr_percent_bins[["bins", "percent"]]

        ttr_percent_bins["cumulative_percent"] = ttr_percent_bins["percent"].cumsum()

        return ttr_percent_bins

    ttr_prs_table = percent_bins_table(df)

    ttr_prs_table = ttr_prs_table.rename(
        columns={
            "bins": "Time Window",
            "percent": "Percent",
            "cumulative_percent": "Cumulative Percent",
        }
    )

    return ttr_prs_table

# %% ---------------------------
# Process report data for saving
# ------------------------------


def prep_alltime_data_for_saving(
    start_date,
    report_date,
    pr_status,
    opened_by_status,
    ttr_prs,
    nondev_ttr_prs,
):
    # pr_status metric
    # ----------------------------------------
    pr_status["report_date"] = f"{report_date}"
    pr_status["start_date"] = f"{start_date}"
    pr_status["metric"] = "pr_status"
    pr_status["indicator_name"] = "open_status"
    pr_status["value_type"] = "count"
    pr_status["sub_value_type"] = "percent"

    pr_status = pr_status.rename(
        columns={
            "PR Status": "indicator_value",
            "Count": "value",
            "Percent": "sub_value",
        }
    )

    # opened_by_dev_status metric
    # ----------------------------------------
    opened_by_status["report_date"] = f"{report_date}"
    opened_by_status["start_date"] = f"{start_date}"
    opened_by_status["metric"] = "opened_by_dev_status"
    opened_by_status["indicator_name"] = "opened_by"
    opened_by_status["value_type"] = "count"
    opened_by_status["sub_value_type"] = "NA"
    opened_by_status["sub_value"] = "NA"

    opened_by_status = opened_by_status.rename(
        columns={
            "opened_by": "indicator_value",
            "prs_opened": "value",
        }
    )

    # alltime_ttr_perc metric
    # ----------------------------------------
    ttr_prs["report_date"] = f"{report_date}"
    ttr_prs["start_date"] = f"{start_date}"
    ttr_prs["metric"] = "overall_time_to_respond"
    ttr_prs["indicator_name"] = "time_window"
    ttr_prs["value_type"] = "percent"
    ttr_prs["sub_value_type"] = "cumulative_percent"

    ttr_prs = ttr_prs.rename(
        columns={
            "Time Window": "indicator_value",
            "Percent": "value",
            "Cumulative Percent": "sub_value",
        }
    )

    # alltime_nondev_ttr_perc metric
    # ----------------------------------------
    nondev_ttr_prs["report_date"] = f"{report_date}"
    nondev_ttr_prs["start_date"] = f"{start_date}"
    nondev_ttr_prs["metric"] = "nondev_time_to_respond"
    nondev_ttr_prs["indicator_name"] = "time_window"
    nondev_ttr_prs["value_type"] = "percent"
    nondev_ttr_prs["sub_value_type"] = "cumulative_percent"

    nondev_ttr_prs = nondev_ttr_prs.rename(
        columns={
            "Time Window": "indicator_value",
            "Percent": "value",
            "Cumulative Percent": "sub_value",
        }
    )

    report_data = pd.concat(
        [
            df
            for df in [
                pr_status,
                opened_by_status,
                ttr_prs,
                nondev_ttr_prs,
            ]
            if not df.empty
        ],
        ignore_index=True,
    )

    report_data = report_data[
        [
            "report_date",
            "start_date",
            "metric",
            "indicator_name",
            "indicator_value",
            "value_type",
            "value",
            "sub_value_type",
            "sub_value",
        ]
    ]

    return report_data


def save_alltime_report_data(
    hist_report_data,
    new_report_data,
    unique_id_cols,
    report_path,
    overwrite_historical_data=False,
):
    """ """

    if overwrite_historical_data:
        if os.path.exists(report_path):
            print("Overwriting previous report with new data")

            with open(report_path, "wb") as f:
                pickle.dump(new_report_data, f)

            print(f"\nReport saved to: {report_path}")
        else:
            with open(report_path, "wb") as f:
                pickle.dump(new_report_data, f)

            print(f"\nReport saved to: {report_path}")

    elif (
        (os.path.exists(report_path))
        and (not overwrite_historical_data)
        and (unique_id_cols)
    ):
        with open(report_path, "rb") as f:
            hist_report_data = pickle.load(f)

        # create unique id column for historical data
        hist_report_data["unique_id"] = (
            hist_report_data[unique_id_cols].astype(str).agg("_".join, axis=1)
        )
        new_report_data["unique_id"] = (
            new_report_data[unique_id_cols].astype(str).agg("_".join, axis=1)
        )

        # identify overlapping unique ids
        overlapping_ids = new_report_data["unique_id"].isin(
            hist_report_data["unique_id"]
        )

        # check if unique id already exists in historical data
        if overlapping_ids.any():
            print(
                f"Unique IDs '{overlapping_ids}' already exists in the historical "
                "data. Set overwrite_historical_data=True to replace the data.\n"
                "Retaining historical data and appending new data only."
            )

            new_report_data = new_report_data[~overlapping_ids]

        combined = pd.concat(
            [
                new_report_data,
                hist_report_data,
            ],
            ignore_index=True,
        )
        combined = combined.drop(columns=["unique_id"])

        with open(report_path, "wb") as f:
            pickle.dump(combined, f)

        print(f"\nReport saved to: {report_path}")

    else:
        if unique_id_cols:
            print(
                "Unable to process historical report data. Check that "
                f"the datapath '{report_path}' is correct."
                "\n\nReport not saved."
            )
        else:
            print(
                "The 'unique_id_cols' parameter must be passed when "
                "'overwrite_historical_data' is False or None. Please "
                "pass a valid list of column names to use as the unique "
                "identifiers"
                "\n\nReport not saved."
            )

    return




def run_alltime_report(
    raw_prs_data=False,
    start_date=False,
    end_date=False,
    save_report_data=True,
    report_name="prs_report.pkl",
    dev_usernames=[
        "stephanie-r-jones",
        "jasmainak",
        "ntolley",
        "rythorpe",
        "asoplata",
        "dylansdaniels",
        "blakecaldwell",
    ],
    verbose=False,
):
    """
    Run PRs report.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing PRs data.
    """
    run_date = datetime.now().date()

    # set report end date
    # ------------------------------
    if end_date is False:
        report_date = run_date
    else:
        report_date = datetime.strptime(str(end_date), "%Y-%m-%d").date()

    if verbose:
        print(f"Using report date of {report_date}\n")

    # If needed, load pickle file of raw pr data
    # generated by download_prs.py
    # ------------------------------
    if not isinstance(raw_prs_data, pd.DataFrame):
        with open(DATAPATH, "rb") as f:
            raw_prs_data = pickle.load(f)

        raw_prs_data = pd.DataFrame(raw_prs_data)

    df = raw_prs_data.copy()

    if verbose:
        print(
            "Date range of opened prs:",
            f"\n   First_pr_opened : {df['date_opened'].min()} UTC",
            f"\n   Last_pr_opened  : {df['date_opened'].max()} UTC",
        )

    # set start date
    # ------------------------------
    if start_date is False:
        # datetime of earliest record in EST
        start_date = pd.to_datetime(df["date_time"].min(), utc=True)
        start_date = start_date.tz_convert("US/Eastern")
        # remove time from record
        start_date = start_date.tz_localize(None)

    start_date = pd.to_datetime(start_date).date()

    if verbose:
        print(
            "\nDate range of report:",
            f"\n   Start : {start_date} EST",
            f"\n   End   : {report_date} EST",
        )

    # preprocess raw data
    # ------------------------------
    df = preprocess(
        df,
        dev_usernames,
        start_date=start_date,
        end_date=report_date,
    )

    # generate table of prs opened, closed
    # ------------------------------
    pr_status_overall = pr_status_counts(df)

    # generate table of prs opened by developer status
    # ------------------------------
    opened_by_status_table = prs_opened_by_users(
        df,
        by_dev_status=True,
    )

    # Generate overall time-to-response table
    # ------------------------------
    if df.empty:
        ttr_prs_table = pd.DataFrame(
            columns=[
                "Time Window",
                "Percent",
                "Cumulative Percent",
            ]
        )
    else:
        prs_segmented = process_prs_for_ttr(
            df,
            report_date,
        )
        ttr_prs_table = generate_ttr_table(prs_segmented)

    # Generate non-developer time-to-response table
    # ------------------------------

    nondev_prs = df.loc[df["opened_by"] != "Developer"].reset_index(drop=True)

    if nondev_prs.empty:
        if verbose:
            print("\nNo prs opened by non-developers in the specified date range.")
        nondev_ttr_prs_table = pd.DataFrame(
            columns=[
                "Time Window",
                "Percent",
                "Cumulative Percent",
            ]
        )
    else:
        nondev_prs_segmented = process_prs_for_ttr(
            df.loc[df["opened_by"] != "Developer"].reset_index(drop=True),
            report_date,
        )
        nondev_ttr_prs_table = generate_ttr_table(nondev_prs_segmented)

    # format report data
    # ------------------------------
    report_data = prep_alltime_data_for_saving(
        start_date,
        report_date,
        pr_status_overall,
        opened_by_status_table,
        ttr_prs_table,
        nondev_ttr_prs_table,
    )

    # optionally save report data
    # ------------------------------

    # --- DEV NOTE --- #
    #  Currently using pickle instead of save_alltime_report_data()

    if save_report_data:
        report_path = os.path.join(
            "issues_metrics",
            report_name,
        )

        report_data.to_pickle(report_path)

        print(f"\nReport data saved to {report_path}")
    # --- END NOTE --- #

    return report_data


def run_monthly_report(
    start_date=False,
    end_date=False,
    save_report_data=True,
    report_name="monthly_prs_report.pkl",
    dev_usernames=[
        "stephanie-r-jones",
        "jasmainak",
        "ntolley",
        "rythorpe",
        "asoplata",
        "dylansdaniels",
        "blakecaldwell",
    ],
    verbose=False,
):
    run_date = datetime.now().date()

    report_path = os.path.join(
        "issues_metrics",
        report_name,
    )
    agg_report_data = []

    # Load pickle file of raw pr data
    # ------------------------------
    with open(DATAPATH, "rb") as f:
        raw_prs_data = pickle.load(f)

    raw_prs_data = pd.DataFrame(raw_prs_data)
    df = raw_prs_data.copy()

    if verbose:
        print(
            "Date range of opened prs:",
            f"\n   First_pr_opened : {df['date_opened'].min()} UTC",
            f"\n   Last_pr_opened  : {df['date_opened'].max()} UTC",
        )

    # set report start / end dates
    # ------------------------------
    if end_date is False:
        end_date = run_date
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    if start_date is False:
        # use datetime of earliest record in EST
        start_date = pd.to_datetime(df["date_time"].min(), utc=True)
        start_date = start_date.tz_convert("US/Eastern")
        start_date = start_date.tz_localize(None)

    start_date = pd.to_datetime(start_date).date()

    # preprocess raw data
    # ------------------------------
    # need to use the processed data to get accurate year-months after
    # timezone conversions
    tmp_df = preprocess(
        df,
        dev_usernames,
        start_date=start_date,
        end_date=end_date,
    )

    # get year-month
    tmp_df["year_month"] = pd.to_datetime(tmp_df["date_opened"]).dt.to_period("M")

    # drop NaT values to prevent the loop from processing null periods,
    # which may default to Unix Epoch (1970) in downstream calculations
    year_months = tmp_df["year_month"].dropna().sort_values().unique()

    # loop through year-months for monthly metrics
    # ------------------------------
    for month in year_months:
        if verbose:
            print(f"Processing data for {month}")

        # get first and last day of month
        month_start = month.to_timestamp().date()
        month_end = (month + 1).to_timestamp().date() - timedelta(days=1)

        metrics_monthly = run_alltime_report(
            raw_prs_data=raw_prs_data,
            start_date=month_start,
            end_date=month_end,
            save_report_data=False,
            dev_usernames=dev_usernames,
            verbose=verbose,
        )

        metrics_monthly["metric_period"] = "monthly"

        agg_report_data.append(metrics_monthly)

    # loop through year-months for rolling metrics
    for month in year_months:
        month_end = (month + 1).to_timestamp().date() - timedelta(days=1)

        metrics_monthly = run_alltime_report(
            raw_prs_data=raw_prs_data,
            start_date=start_date,
            end_date=month_end,
            save_report_data=False,
            dev_usernames=dev_usernames,
            verbose=verbose,
        )

        metrics_monthly["metric_period"] = "rolling_monthly"

        agg_report_data.append(metrics_monthly)

    # combine reports
    combined_report_data = pd.concat(
        agg_report_data,
        ignore_index=True,
    )

    # save to pickle, overwrite data
    if save_report_data:
        save_alltime_report_data(
            hist_report_data=None,
            new_report_data=combined_report_data,
            unique_id_cols=None,
            report_path=report_path,
            overwrite_historical_data=True,
        )

    return


def run_u24_pr_report(
    start_date="2023-08-01",
    end_date=False,
    save_report_data=True,
    report_name="u24_prs_report.pkl",
    overwrite_historical_data=False,
    dev_usernames=[
        "stephanie-r-jones",
        "jasmainak",
        "ntolley",
        "rythorpe",
        "asoplata",
        "dylansdaniels",
        "blakecaldwell",
    ],
    verbose=False,
):
    run_date = datetime.now().date()

    if end_date is False or end_date is True:
        end_date = str(run_date)

    report_path = os.path.join("issues_metrics", report_name)
    all_report_data = []

    # -------------------------------
    # All-time report
    # -------------------------------

    metrics_alltime = run_alltime_report(
        start_date=start_date,
        end_date=end_date,
        save_report_data=False,
        report_name=report_name,
        dev_usernames=dev_usernames,
        verbose=verbose,
    )

    metrics_alltime["grant_year"] = "all_time"

    all_report_data.append(metrics_alltime)

    # -------------------------------
    # Grant year reports
    # -------------------------------
    grant_years = [
        ("2023-08-01", "2024-07-31"),
        ("2024-08-01", "2025-07-31"),
        ("2025-08-01", "2026-07-31"),
        ("2026-08-01", "2027-07-31"),
        ("2027-08-01", "2028-07-31"),
    ]

    # filter to only grant years ending <= end_date
    grant_years = [gy for gy in grant_years if gy[0] <= end_date]

    for i, (gy_start, gy_end) in enumerate(grant_years, start=1):
        metrics_gy = run_alltime_report(
            start_date=gy_start,
            end_date=gy_end,
            save_report_data=False,
            report_name=report_name,
            dev_usernames=dev_usernames,
            verbose=verbose,
        )

        metrics_gy["grant_year"] = f"year {i}"
        all_report_data.append(metrics_gy)

    # combine all reports
    combined_report_data = pd.concat(
        all_report_data,
        ignore_index=True,
    )

    # save to pickle
    if save_report_data:
        # open historical data if it exists
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                hist_report_data = pickle.load(f)
        else:
            hist_report_data = None

        save_alltime_report_data(
            hist_report_data=hist_report_data,
            new_report_data=combined_report_data,
            unique_id_cols=["report_date", "start_date", "grant_year", "metric"],
            report_path=report_path,
            overwrite_historical_data=overwrite_historical_data,
        )

    return combined_report_data


def run_main_reports(dev_usernames=None):
    # set report parameters
    # ------------------------------
    start_date = False
    end_date = False
    save_report_data = True
    overwrite_historical_data = True

    if dev_usernames is None:
        dev_usernames = [
            "stephanie-r-jones",
            "jasmainak",
            "ntolley",
            "rythorpe",
            "asoplata",
            "dylansdaniels",
            "blakecaldwell",
            "katduecker",
            "carolinafernandezp",
            "gtdang",
            "kmilo9999",
            "samadpls",
            "Myrausman",
            "Chetank99",
        ]

    # --> [DEV] for local testing
    if "dylandaniels" in os.getcwd():
        start_date = "2019-01-01"
        end_date = "2026-05-01"
    # --> [END DEV]

    alltime_report_dir = os.path.join(
        "report_data",
        "alltime_prs_report.pkl",
    )
    monthly_report_dir = os.path.join(
        "report_data",
        "monthly_prs_report.pkl",
    )
    u24_report_dir = os.path.join(
        "report_data",
        "u24_prs_report.pkl",
    )

    print("\n" + "# " + "-" * 48)
    print("# Runing 'all time' report")
    print("# " + "-" * 48, "\n")
    processed_df = run_alltime_report(
        start_date=start_date,
        end_date=end_date,
        save_report_data=save_report_data,
        report_name=alltime_report_dir,
        dev_usernames=dev_usernames,
    )

    print("\n" + "# " + "-" * 48)
    print("# Runing 'monthly' report")
    print("# " + "-" * 48, "\n")
    _ = run_monthly_report(
        start_date=start_date,
        end_date=end_date,
        save_report_data=save_report_data,
        report_name=monthly_report_dir,
        dev_usernames=dev_usernames,
    )

    print("\n" + "# " + "-" * 48)
    print("# Runing 'U24' report")
    print("# " + "-" * 48, "\n")
    # Run U24 grant-year report
    _ = run_u24_pr_report(
        end_date=end_date,
        save_report_data=save_report_data,
        report_name=u24_report_dir,
        overwrite_historical_data=overwrite_historical_data,
        dev_usernames=dev_usernames,
    )

    return processed_df


# %% ----------------------------------------
# Run main
# -------------------------------------------

if __name__ == "__main__":
    run_main_reports()

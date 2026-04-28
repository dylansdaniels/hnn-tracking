# issues_analysis.py

# potential enhancements
# - the "business_hours_elapsed" function could be vectorized

# %% ----------------------------------------
# Setup
# -------------------------------------------

import os
import pickle
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from pandas.tseries.holiday import USFederalHolidayCalendar

DATAPATH = os.path.join("issues_metrics", "raw_issues_data.pkl")


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


# %% ----------------------------------------
# Data preprocessing
# -------------------------------------------


def process_datetime(df, date_cols):
    for col in date_cols:
        if "date_time" in col:
            name = col.replace("date_time", "datetime_opened")
        else:
            name = col.replace("date", "datetime")

        df[name] = pd.to_datetime(
            df[col],
            utc=True,
        )
        df[name] = df[name].dt.tz_convert("US/Eastern")
        df[name] = df[name].dt.tz_localize(None)
        df[col] = df[name].dt.date

    # use timezone-corrected datetime_opened for date_opened
    df["date_opened"] = df["datetime_opened"].dt.date

    return df


def preprocess(
    df,
    dev_usernames,
    start_date=False,
    end_date=False,
):
    df = df.copy()

    df = process_datetime(
        df,
        [
            "date_time",  # -> datetime_opened
            "date_closed",  # + datetime_closed
            "comment_date",  # + comment_datetime
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

        # remove issues opened after end_date
        # ------------------------------
        # - get issues numbers
        issues_to_remove = df.loc[df["date_opened"] >= end_date]["number"].unique()

        # - remove issues based on number
        df = df.loc[~df["number"].isin(issues_to_remove)].reset_index(drop=True)

        # clear fields for issues closed
        # after the end_date
        # ------------------------------
        # Note: need to compare timestampts to handle NaTs properly
        invalid_dateclosed = df.loc[pd.to_datetime(df["datetime_closed"]) >= end_ts][
            "number"
        ].unique()

        for issue_num in invalid_dateclosed:
            # set date_closed and datetime_closed to NaT
            df.loc[
                df["number"] == issue_num,
                ["date_closed", "datetime_closed"],
            ] = pd.NaT
            # set closed_by to ""
            df.loc[
                df["number"] == issue_num,
                "closed_by",
            ] = ""

        # clear fields for *only* the
        # comments made after the end_date
        # ------------------------------
        # Note: need to compare timestampts to handle NaTs properly
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
            "issue_name",
            "issue_url",
            "date_closed",
            "datetime_closed",
            "closed_by",
            "comment_date",
            "comment_datetime",
            "comment_username",
            "comment_contents",
        ]
    ]

    return df


# number of issues by by username
# ------------------------------
def issues_by_user(
    df,
    show=True,
    return_df=False,
):
    issues_by_user = df[
        [
            "issue_name",
            "date_opened",
            "username",
        ]
    ].drop_duplicates()

    issues_by_user = issues_by_user.groupby("username").count().reset_index()

    issues_by_user = issues_by_user[
        [
            "username",
            "issue_name",
        ]
    ].rename(
        columns={
            "issue_name": "issues_opened",
        }
    )

    if show:
        display(
            pd.concat(
                [
                    issues_by_user,
                    pd.DataFrame(
                        {
                            "username": ["Total"],
                            "issues_opened": [issues_by_user["issues_opened"].sum()],
                        }
                    ),
                ],
                ignore_index=True,
            ).reset_index(drop=True)
        )

    if return_df:
        return issues_by_user

    return


# issues_by_user(df)

# %% -------------------------------------
# Get issues closed / merged
# ----------------------------------------


def issue_status_counts(
    data,
):
    df = data.copy()
    df = df[
        [
            "number",
            "date_closed",
        ]
    ]
    df = df.drop_duplicates().reset_index(drop=True)

    total = len(df["number"])

    if total == 0:
        return pd.DataFrame(
            {
                "Issue Status": [
                    "New Issues",
                    "Outstanding Issues",
                    "Closed Issues",
                ],
                "Count": [0, 0, 0],
                "Percent": [0.0, 0.0, 0.0],
            }
        )

    closed_issues = df["date_closed"].notna().sum()
    outstanding_issues = total - closed_issues

    table = pd.DataFrame(
        {
            "Issue Status": [
                "New Issues",
                "Outstanding Issues",
                "Closed Issues",
            ],
            "Count": [
                total,
                outstanding_issues,
                closed_issues,
            ],
            "Percent": [
                100,
                round(outstanding_issues / total * 100, 2),
                round(closed_issues / total * 100, 2),
            ],
        }
    )

    return table


# %% -------------------------------------
# Get issues opened by users
# ----------------------------------------


def issues_opened_by_users(
    df,
    by_dev_status=False,
    return_df=False,
):
    if by_dev_status:
        by_col = "opened_by"
    else:
        by_col = "username"

    issues_by_user = df[
        [
            "issue_name",
            "date_opened",
            by_col,
        ]
    ].drop_duplicates()

    issues_by_user = issues_by_user.groupby(by_col).count().reset_index()
    issues_by_user = issues_by_user[
        [
            by_col,
            "issue_name",
        ]
    ].rename(
        columns={
            "issue_name": "issues_opened",
        }
    )

    table = pd.concat(
        [
            issues_by_user,
            pd.DataFrame(
                {
                    by_col: ["Total"],
                    "issues_opened": [issues_by_user["issues_opened"].sum()],
                }
            ),
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    if return_df:
        return issues_by_user

    return table


# %% -------------------------------------
# Get Time to Response Metric
# ----------------------------------------


def process_issues_for_ttr(
    df,
    report_date,
):
    # unique non-bot issues
    # ------------------------------
    unique_issues = df.drop_duplicates(
        [
            "issue_name",
            "date_opened",
            "username",
        ]
    )

    # issues w/o comments
    # ------------------------------
    no_response = (
        unique_issues[
            unique_issues["comment_datetime"].apply(
                lambda x: not isinstance(
                    x,
                    pd.Timestamp,
                )
            )
        ]
        .reset_index(drop=True)
        .copy()
    )

    if no_response.empty:
        no_response = pd.DataFrame(columns=unique_issues.columns)

    no_response["status"] = "no response"

    # get issues w/ comments
    # ------------------------------
    with_response = (
        unique_issues[
            unique_issues["comment_datetime"].apply(
                lambda x: isinstance(
                    x,
                    pd.Timestamp,
                )
            )
        ]
        .reset_index(drop=True)
        .copy()
    )

    # ensure DataFrame has the right columns if empty
    if with_response.empty:
        with_response = pd.DataFrame(columns=unique_issues.columns)

    unique_issues_with_response = list(with_response["number"].unique())

    # create indicator var for when username != comment_username
    # ------------------------------
    def assign_external_response(row):
        """
        Function to compare username and comment_username rows
        of a DataFrame and return 1 if they are not equal, else 0.
        """
        if row["username"] != row["comment_username"]:
            return 1
        else:
            return 0

    with_response["ext_response"] = with_response.apply(
        lambda x: assign_external_response(x), axis=1
    )

    # check that all records have valid ext_response value
    # ------------------------------
    if not set(with_response["ext_response"].unique()).issubset({0, 1}):
        raise ValueError("ext_response column should only contain 0 and 1")
    else:
        pass

    # split into two dataframes on ext_response indicator
    # ------------------------------
    external_responses_all = with_response[with_response["ext_response"] == 1].copy()
    without_ext = with_response[with_response["ext_response"] == 0].copy()

    # sort by id, date with oldest dates first
    external_responses_all = external_responses_all.sort_values(
        ["number", "comment_date"],
        ascending=[False, True],
    )
    external_responses_all = external_responses_all.reset_index(drop=True)

    # unique issues with an external response, keeping the first response instance
    # ------------------------------
    external_response = (
        external_responses_all.drop_duplicates(["number"]).reset_index(drop=True).copy()
    )
    external_response["drop"] = 1

    # unique issues with only self comments
    # ------------------------------
    self_response = without_ext.join(
        external_response[["number", "drop"]].set_index("number"),
        on="number",
        how="left",
    )

    # drop rows where top is 1
    # ------------------------------
    self_response = self_response[self_response["drop"] != 1].copy()
    self_response = self_response.drop_duplicates(["number"])
    self_response = self_response.reset_index(drop=True)

    cols_to_remove = ["ext_response", "drop"]

    self_response = self_response.drop(
        columns=cols_to_remove,
    )
    external_response = external_response.drop(
        columns=cols_to_remove,
    )

    self_response["status"] = "self comment"
    external_response["status"] = "external comment"

    # check that all records are accounted for after manipulations
    # ------------------------------
    if not len(self_response["number"]) + len(external_response["number"]) == len(
        unique_issues_with_response
    ):
        raise ValueError(
            "Number of unique issues with a response has changed,"
            " which indicates a problem with the data processing."
            " Please check the code and try again."
        )
    else:
        pass

    # confirm issue counts are correct after segmentation
    # ----------------------------------------
    if not len(self_response["number"]) + len(external_response["number"]) + len(
        no_response["number"]
    ) == len(unique_issues):
        raise ValueError(
            "Number of unique issues has changed after segmentation,"
            " which indicates a problem with the data processing."
            " Please check the code and try again."
        )
    else:
        pass

    # Determine time-to-respond metric
    # ------------------------------

    issues_segmented = pd.concat(
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

    issues_segmented = issues_segmented.sort_values("number", ascending=False)

    def assign_ttr_date(row, report_date):
        """
        Function to assign a date to use for time-to-respond metric based on
        the status of the issue.
        """
        # format report date
        report_date = pd.to_datetime(report_date)

        # assign ttr_date based on status
        if row["status"] == "no response":
            if pd.notnull(row["datetime_closed"]):
                return row["datetime_closed"]
            else:
                return report_date
        elif row["status"] == "self comment":
            if pd.notnull(row["datetime_closed"]):
                return row["datetime_closed"]
            else:
                return report_date
        elif row["status"] == "external comment":
            if pd.notnull(row["datetime_closed"]):
                # return whichever is earliest between datetime_closed
                # and comment_datetime
                return min(
                    pd.to_datetime(row["datetime_closed"]),
                    pd.to_datetime(row["comment_datetime"]),
                )
            else:
                if not isinstance(row["comment_datetime"], pd.Timestamp):
                    print("\n--- BAD TYPE DETECTED ---")
                    print(f"number: {row['number']}, status: {row['status']}")
                    print("comment_datetime type:", type(row["comment_datetime"]))
                    print("comment_datetime value:", row["comment_datetime"])
                return row["comment_datetime"]
        else:
            raise ValueError(
                "Invalid status value. Expected 'no response', 'self comment',"
                " or 'external comment'."
            )

    issues_segmented["ttr_date"] = issues_segmented.apply(
        lambda x: assign_ttr_date(
            x,
            report_date,
        ),
        axis=1,
    )

    def business_hours_elapsed(df):
        """
        Calculate the business hours elapsed between datetime_opened and ttr_date,
        subtracting time spent on weekends and holidays

        Parameters:
        -----------
        df : pandas.DataFrame

        Returns:
        --------
        pandas.DataFrame
            The initial DataFrame with an additional column for "ttr_hours"
        """

        # build holiday calendar
        start_holiday = df["datetime_opened"].min().floor("D")
        end_holiday = df["ttr_date"].max().floor("D")

        cal = USFederalHolidayCalendar()
        holidays = set(cal.holidays(start=start_holiday, end=end_holiday).date)

        def calc(row):
            # get the issue open date and the response date
            start = row["datetime_opened"]
            end = row["ttr_date"]

            # total time elapsed between issue open and response
            raw_elapsed = end - start

            # get all calendar days
            all_dates = pd.date_range(
                start=start.floor("D"), end=end.floor("D"), freq="D"
            ).date

            # get all holiday/weeksnds
            days_to_exclude = {
                d for d in all_dates if d.weekday() >= 5 or d in holidays
            }

            # calculate the non-business-day hours to subtract from the total elapsed
            # time between when the issue was open and the first response
            total_exclude_time = timedelta(0)
            for d in days_to_exclude:
                if d == start.date() and d == end.date():
                    # start and end day are the same holiday/weekend
                    total_exclude_time += (end - start)
                elif d == start.date():
                    # only the start is on a holiday / weekend
                    # here we get time between start time and mignight of the next day
                    next_day_midnight = start.floor("D") + timedelta(days=1)
                    total_exclude_time += (next_day_midnight - start)
                elif d == end.date():
                    # only the end is on a holiday / weekend
                    # here we get time between midnight and the end time on the same day
                    total_exclude_time += (end - end.floor("D"))
                else:
                    # full holiday / weekends between the start/end dates
                    total_exclude_time += timedelta(days=1)

            business_delta = raw_elapsed - total_exclude_time
            business_delta = max(business_delta, timedelta(0))
            return round(business_delta.total_seconds() / 3600, 1)

        df["ttr_hours"] = df.apply(calc, axis=1)
        return df


    issues_segmented = business_hours_elapsed(issues_segmented)
    issues_segmented["ttr_days"] = round(issues_segmented["ttr_hours"] / 24, 2)

    return issues_segmented


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

    ttr_issues_table = percent_bins_table(df)

    ttr_issues_table = ttr_issues_table.rename(
        columns={
            "bins": "Time Window",
            "percent": "Percent",
            "cumulative_percent": "Cumulative Percent",
        }
    )

    return ttr_issues_table


# %% ----------------------------------------
# Define and run reports
# -------------------------------------------


def build_report_tables_from_records(
    df,
    report_date=None,
    display_tables=True,
    style_displayed_tables=True,
):
    """
    Build report tables from the processed data.

    Parameters
    ----------
    df : pd.DataFrame
    report_date : datetime.date, optional
    display_tables : bool
    style_displayed_tables : bool

    Returns
    -------
    dict
    """
    if report_date is None:
        report_date = datetime.now().date()

    # Issue status metrics
    issues_status_overall = issue_status_counts(df)

    opened_by_status_table = issues_opened_by_users(
        df,
        by_dev_status=True,
    )

    # Time-to-response metrics
    issues_segmented = process_issues_for_ttr(df, report_date)
    ttr_issues_table = generate_ttr_table(issues_segmented)

    nondev_issues = df.loc[df["opened_by"] != "Developer"].reset_index(drop=True)
    if nondev_issues.empty:
        nondev_ttr_issues_table = pd.DataFrame(
            columns=["Time Window", "Percent", "Cumulative Percent"]
        )
    else:
        nondev_issues_segmented = process_issues_for_ttr(nondev_issues, report_date)
        nondev_ttr_issues_table = generate_ttr_table(nondev_issues_segmented)

    tables = {
        "issues_status_overall": issues_status_overall,
        "opened_by_status_table": opened_by_status_table,
        "ttr_issues_table": ttr_issues_table,
        "nondev_ttr_issues_table": nondev_ttr_issues_table,
    }

    if display_tables:
        if style_displayed_tables:
            for key, table in tables.items():
                render_html_table(table)
        else:
            for table in tables.values():
                display(table)

    return tables


# %% ---------------------------
# Process report data for saving
# ------------------------------


def prep_alltime_data_for_saving(
    start_date,
    report_date,
    issues_status,
    opened_by_status,
    ttr_issues,
    nondev_ttr_issues,
):
    # issues_status metric
    # ----------------------------------------
    issues_status["report_date"] = f"{report_date}"
    issues_status["start_date"] = f"{start_date}"
    issues_status["metric"] = "issues_status"
    issues_status["indicator_name"] = "open_status"
    issues_status["value_type"] = "count"
    issues_status["sub_value_type"] = "cumulative_percent"

    issues_status = issues_status.rename(
        columns={
            "Issue Status": "indicator_value",
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
            "issues_opened": "value",
        }
    )

    # alltime_ttr_perc metric
    # ----------------------------------------
    ttr_issues["report_date"] = f"{report_date}"
    ttr_issues["start_date"] = f"{start_date}"
    ttr_issues["metric"] = "overall_time_to_respond"
    ttr_issues["indicator_name"] = "time_window"
    ttr_issues["value_type"] = "percent"
    ttr_issues["sub_value_type"] = "cumulative_percent"

    ttr_issues = ttr_issues.rename(
        columns={
            "Time Window": "indicator_value",
            "Percent": "value",
            "Cumulative Percent": "sub_value",
        }
    )

    # alltime_nondev_ttr_perc metric
    # ----------------------------------------
    nondev_ttr_issues["report_date"] = f"{report_date}"
    nondev_ttr_issues["start_date"] = f"{start_date}"
    nondev_ttr_issues["metric"] = "nondev_time_to_respond"
    nondev_ttr_issues["indicator_name"] = "time_window"
    nondev_ttr_issues["value_type"] = "percent"
    nondev_ttr_issues["sub_value_type"] = "cumulative_percent"

    nondev_ttr_issues = nondev_ttr_issues.rename(
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
                issues_status,
                opened_by_status,
                ttr_issues,
                nondev_ttr_issues,
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


def build_report_tables_from_pickle(
    report_data,
    display_tables=True,
    style_displayed_tables=True,
):
    """
    Build report tables from saved metrics data in a generic way.

    Parameters
    ----------
    report_data : pd.DataFrame
    display_tables : bool
    style_displayed_tables : bool

    Returns
    -------
    dict
        Dictionary of tables keyed by metric type.
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


def run_alltime_report(
    raw_issue_data=False,
    start_date=False,
    end_date=False,
    save_report_data=True,
    report_name="basic_report.pkl",
    dev_usernames=[
        "stephanie-r-jones",
        "jasmainak",
        "ntolley",
        "rythorpe",
        "asoplata",
        "dylansdaniels",
        "blakecaldwell",
    ],
):
    """
    Run issues report.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing issues data.
    """
    run_date = datetime.now().date()

    # set report end date
    # ------------------------------
    if end_date is False:
        report_date = run_date
    else:
        report_date = datetime.strptime(str(end_date), "%Y-%m-%d").date()

    print(f"Using report date of {report_date}\n")

    # If needed, load pickle file of raw issue data
    # generated by download_issues.py
    # ------------------------------
    if not isinstance(raw_issue_data, pd.DataFrame):
        with open(DATAPATH, "rb") as f:
            raw_issue_data = pickle.load(f)

        raw_issue_data = pd.DataFrame(raw_issue_data)

    df = raw_issue_data.copy()

    print(
        "Date range of opened issues:",
        f"\n   First_issue_opened : {df['date_opened'].min()} UTC",
        f"\n   Last_issue_opened  : {df['date_opened'].max()} UTC",
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

    # generate table of issues opened, closed
    # ------------------------------
    issues_status_overall = issue_status_counts(df)

    # generate table of issues opened by developer status
    # ------------------------------
    opened_by_status_table = issues_opened_by_users(
        df,
        by_dev_status=True,
    )

    # Generate overall time-to-response table
    # ------------------------------
    if df.empty:
        ttr_issues_table = pd.DataFrame(
            columns=[
                "Time Window",
                "Percent",
                "Cumulative Percent",
            ]
        )
    else:
        issues_segmented = process_issues_for_ttr(
            df,
            report_date,
        )
        ttr_issues_table = generate_ttr_table(issues_segmented)

    # Generate non-developer time-to-response table
    # ------------------------------

    nondev_issues = df.loc[df["opened_by"] != "Developer"].reset_index(drop=True)

    if nondev_issues.empty:
        print("\nNo issues opened by non-developers in the specified date range.")
        nondev_issues_segmented = pd.DataFrame(
            columns=[
                "number",
                "date_opened",
                "datetime_opened",
                "opened_by",
                "username",
                "issue_name",
                "issue_url",
                "date_closed",
                "datetime_closed",
                "closed_by",
                "comment_date",
                "comment_datetime",
                "comment_username",
                "comment_contents",
                "status",
                "ttr_date",
                "ttr_hours",
                "ttr_days",
            ]
        )
        nondev_ttr_issues_table = pd.DataFrame(
            columns=[
                "Time Window",
                "Percent",
                "Cumulative Percent",
            ]
        )
    else:
        nondev_issues_segmented = process_issues_for_ttr(
            df.loc[df["opened_by"] != "Developer"].reset_index(drop=True),
            report_date,
        )
        nondev_ttr_issues_table = generate_ttr_table(nondev_issues_segmented)

    # format report data
    # ------------------------------
    report_data = prep_alltime_data_for_saving(
        start_date,
        report_date,
        issues_status_overall,
        opened_by_status_table,
        ttr_issues_table,
        nondev_ttr_issues_table,
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
    report_name="monthly_issues_report.pkl",
    dev_usernames=[
        "stephanie-r-jones",
        "jasmainak",
        "ntolley",
        "rythorpe",
        "asoplata",
        "dylansdaniels",
        "blakecaldwell",
    ],
):
    run_date = datetime.now().date()

    report_path = os.path.join(
        "issues_metrics",
        report_name,
    )
    agg_report_data = []

    # Load pickle file of raw issue data
    # ------------------------------
    with open(DATAPATH, "rb") as f:
        raw_issue_data = pickle.load(f)

    raw_issue_data = pd.DataFrame(raw_issue_data)
    df = raw_issue_data.copy()

    print(
        "Date range of opened issues:",
        f"\n   First_issue_opened : {df['date_opened'].min()} UTC",
        f"\n   Last_issue_opened  : {df['date_opened'].max()} UTC",
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

    year_months = tmp_df["year_month"].sort_values().unique()

    # loop through year-months for monthly metrics
    # ------------------------------
    for month in year_months:
        print(f"DEV: processing data for {month}")

        # get first and last day of month
        month_start = month.to_timestamp().date()
        month_end = (month + 1).to_timestamp().date() - timedelta(days=1)

        metrics_monthly = run_alltime_report(
            raw_issue_data=raw_issue_data,
            start_date=month_start,
            end_date=month_end,
            save_report_data=False,
            dev_usernames=dev_usernames,
        )

        metrics_monthly["metric_period"] = "monthly"

        agg_report_data.append(metrics_monthly)

    # loop through year-months for rolling metrics
    for month in year_months:
        month_end = (month + 1).to_timestamp().date() - timedelta(days=1)

        metrics_monthly = run_alltime_report(
            raw_issue_data=raw_issue_data,
            start_date=start_date,
            end_date=month_end,
            save_report_data=False,
            dev_usernames=dev_usernames,
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


def run_u24_ttr_report(
    start_date="2023-08-01",
    end_date=False,
    save_report_data=True,
    report_name="u24_issues_report.pkl",
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


# %% ----------------------------------------
# Define report-specific visualizations
# -------------------------------------------

sns.set_palette("Set2")


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


def run_main_reports():
    # set report parameters
    # ------------------------------
    start_date = False
    end_date = False
    # display_tables = True
    # style_displayed_tables = True
    save_report_data = False
    overwrite_historical_data = True
    report_name = "alltime_issues_report.pkl"

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

    # --> [DEV]
    if "dylandaniels" in os.getcwd():
        start_date = "2019-01-01"
        end_date = "2025-12-01"
    # --> [END DEV]

    processed_df = run_alltime_report(
        start_date=start_date,
        end_date=end_date,
        save_report_data=True,
        report_name=report_name,
        dev_usernames=dev_usernames,
    )

    # Run U24 grant-year report
    u24_report_data = run_u24_ttr_report(
        end_date=end_date,
        save_report_data=save_report_data,
        report_name="u24_issues_report.pkl",
        overwrite_historical_data=overwrite_historical_data,
        dev_usernames=dev_usernames,
    )

    _ = run_monthly_report(
        # in progress # --
        start_date=start_date,
        end_date=end_date,
        save_report_data=True,
        report_name="monthly_issues_report.pkl",
        dev_usernames=dev_usernames,
    )

    # ---------------------------------
    # posthoc checks
    # ---------------------------------
    with open(
        os.path.join(
            "issues_metrics",
            report_name,
        ),
        "rb",
    ) as f:
        report_data = pickle.load(f)

    # only show data for the current report date
    # ---------------------------------
    display(
        report_data[report_data["report_date"] == str(pd.to_datetime(end_date).date())]
    )

    # generate tables for each year
    # ---------------------------------
    print(
        "\n"
        "====================================\n"
        "======== U24 Report Outputs ========\n"
        "====================================\n"
    )
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

    print(
        "\n"
        "========================================\n"
        "======== Monthly Report Outputs ========\n"
        "========================================\n"
    )

    # monthly and rolling visualizations
    # ---------------------------------
    monthly_report_path = os.path.join(
        "issues_metrics",
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

    return processed_df


# %% ----------------------------------------
# Run main
# -------------------------------------------

if __name__ == "__main__":
    run_main_reports()

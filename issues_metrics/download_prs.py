# %% ----------------------------------------
# download_prs.py

import os
import pickle
import time
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}  # for GitHub Action
# HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}  # for local testing

OWNER = "jonescompneurolab"
REPO = "hnn-core"
DATAPATH = os.path.join(
    "issues_metrics",
    "raw_prs_data.pkl",
)

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

# %% ----------------------------------------


def get_start_end_dates(
    df,
    rerun_all=False,
    manual_end=False,
):
    """
    Determine START_DATE and END_DATE based on the historical PR
    data

    START_DATE should be the oldest unresolved PR date in the historical data (to
    minimize the number of API calls needed).

    END_DATE should be the day before the GitHub Action is run
    """

    # check that df of historical data exists
    if (rerun_all is False) and (not isinstance(df, pd.DataFrame)):
        warnings.warn(
            "Historical data not found. All data will be fetched \n"
            "anew even through rerun_all is set to False"
        )

    if not rerun_all:
        # ------- TODO ------- #
        # -------------------- #
        # This is just placeholder code for now
        #
        # Below, I'm using the most recently "opened" PR in
        # the historical data as the start date, though this
        # is not what we actually want to do in practice, as
        # PRs will continue to accrue activity over time
        # and their status may change, which we need to keep
        # track of.
        #
        # We would ideally want to make the starting point
        # the oldest "unresolved" PR, which we would need
        # some logic from the analysis script to figure out
        START_DATE = df["date_opened"].max()

    else:
        START_DATE = "2019-01-01"
        START_DATE = datetime.strptime(START_DATE, "%Y-%m-%d").date()

    if manual_end is True:
        END_DATE = datetime.strptime(manual_end, "%Y-%m-%d").date()
    else:
        # Since the GitHub Action will be run on the first of
        # the month, use the previous day as the end date
        END_DATE = date.today() - timedelta(days=1)

    return START_DATE, END_DATE


def safe_request(url):
    """
    Make a get request to the GitHub API, with handling for rate limits
    """
    try:
        resp = requests.get(
            url,
            headers=HEADERS,
        )
        if (
            resp.status_code == 403
            and "X-RateLimit-Remaining" in resp.headers
            and resp.headers["X-RateLimit-Remaining"] == "0"
        ):
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            reset_time = datetime.fromtimestamp(reset_ts).strftime("%Y-%m-%d %H:%M:%S")
            raise RuntimeError(
                f"GitHub rate limit exceeded. Try again after {reset_time}."
            )
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request to {url} failed: {e}")


def get_pull_requests(START_DATE, END_DATE):
    """
    Get pull requests created between START_DATE and END_DATE
    """
    prs = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls?state=all&per_page=100&page={page}"
        resp = safe_request(url)
        data = resp.json()
        if not data:
            break
        for pr in data:
            created = datetime.strptime(pr["created_at"], "%Y-%m-%dT%H:%M:%SZ").date()
            if START_DATE <= created <= END_DATE:
                prs.append(
                    {
                        "number": pr["number"],
                        "title": pr["title"],
                        "created_at": pr["created_at"],
                        "html_url": pr["html_url"],
                        "user": pr["user"]["login"],
                        "state": pr["state"],
                        "closed_at": pr.get("closed_at", ""),
                        "merged_at": pr.get("merged_at", ""),
                    }
                )
        page += 1
        time.sleep(0.5)
    return prs


def get_review_comments(
    pr_number,
    max_comments=5,
):
    """
    Get review comments for a given PR
    """
    comments = []
    page = 1
    while len(comments) < max_comments:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}/comments?per_page=100&page={page}"
        resp = safe_request(url)
        data = resp.json()
        if not data:
            break
        for comment in data:
            comments.append(
                {
                    "created_at": comment["created_at"],
                    "user": comment["user"]["login"],
                    "body": comment["body"],
                }
            )
            if len(comments) >= max_comments:
                break
        page += 1
        time.sleep(0.5)
    return comments


def get_prs_with_comments(
    rerun_all=False,
    max_comments=5,
    manual_end=False,
):
    """
    Get pull requests with specified number of comments
    """

    if rerun_all:
        print(
            "The 'rerun_all' flag is enabled. All historical data \n"
            "will be fetched and the 'raw_prs_data.pkl' file will \n"
            "be entirely overwritten."
        )

    if os.path.exists(DATAPATH):
        with open(DATAPATH, "rb") as f:
            hist_prs_data = pickle.load(f)
    else:
        hist_prs_data = None
        rerun_all = True

    START_DATE, END_DATE = get_start_end_dates(
        hist_prs_data,
        rerun_all=rerun_all,
        manual_end=manual_end,
    )

    print(f"Fetching data between {START_DATE} and {END_DATE}")

    prs = get_pull_requests(START_DATE, END_DATE)
    rows = []

    for pr in prs:
        comments = get_review_comments(pr["number"], max_comments=max_comments)
        if comments:
            for c in comments:
                rows.append(
                    [
                        pr["number"],
                        pr["created_at"],
                        pr["user"],
                        pr["title"],
                        pr["html_url"],
                        pr["state"],
                        pr["closed_at"],
                        pr["merged_at"],
                        c["user"],
                        c["created_at"],
                        c["body"].replace("\n", " ").strip(),
                    ]
                )
        else:
            rows.append(
                [
                    pr["number"],
                    pr["created_at"],
                    pr["user"],
                    pr["title"],
                    pr["html_url"],
                    pr["state"],
                    pr["closed_at"],
                    pr["merged_at"],
                    "",
                    "",
                    "",
                ]
            )

    columns = [
        "number",
        "date_time",
        "username",
        "pr_title",
        "pr_url",
        "state",
        "date_closed",
        "date_merged",
        "comment_username",
        "comment_date",
        "comment_contents",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns,
    )
    df["date_opened"] = pd.to_datetime(df["date_time"]).dt.date

    print(f"Fetched and updated {len(df['number'].unique())} unique PRs.")

    # append historical data to df if not re-running everything
    if (rerun_all is False) and (isinstance(hist_prs_data, pd.DataFrame)):
        # get historical data before the start date, as everything
        # on or after start date has been fetched anew
        hist_data_retained = hist_prs_data.loc[
            hist_prs_data["date_opened"] < START_DATE
        ].copy()

        df = pd.concat(
            [hist_data_retained, df],
            ignore_index=True,
        )

        df = df.sort_values(
            by="date_time",
            ignore_index=True,
        )

    return df

def download_prs(
    rerun_all=False,
    max_comments=5,
    manual_end=False,

):

    # --> [DEV] for local testing
    if "dylandaniels" in os.getcwd():
        manual_end = "2025-12-01"
    # --> [END DEV]

    prs_data = get_prs_with_comments(
        rerun_all=rerun_all,
        max_comments=max_comments,
        manual_end=manual_end,
    )

    prs_data.to_pickle(DATAPATH)
    print(f"PR data saved to {DATAPATH}")


# %% ----------------------------------------

if __name__ == "__main__":
    download_prs()


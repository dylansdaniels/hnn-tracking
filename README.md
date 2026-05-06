# Issue, Pull Request Tracking and Reporting

The `issues_metrics` directory contains scripts to automatically fetch, analyze, and visualize GitHub activity for issues and pull requests (PRs). It focuses on the volume of issues/PRs and responsiveness (e.g., time to respond). 

## Project Structure

```text
├── .github/workflows/
│   ├── issues_prs-tracking.yml # GitHub Action
├── pyproject.toml
│   └── report_data/            # Contains processed data (the calculated metrics)
│      ├── ...
├── issues_metrics/
│   ├── __init__.py
│   ├── download_issues.py      # Fetch issue data using GitHub API
│   ├── download_prs.py         # Fetch PR data using GitHub API
│   ├── issues_analysis.py      # Preprocess data and calculate issue metrics
│   ├── prs_analysis.py         # Preprocess data and calculate PR metrics
│   ├── issues_viewer.py        # Generate issue visualizations for reporting
│   ├── prs_viewer.py           # Generate PR visualizations for reporting
│   ├── raw_issues_data.pkl     # Raw issues data
│   ├── raw_prs_data.pkl        # Raw PRs data
│   └── run_reports.py          # Main orchestrator
```

## Terms and Definitions

*  **Developers:** Refers to members of the core hnn development team, including GSoC contributors, as opposed to community members who have contributed code to hnn. The list of developers used for segmenting contributors is defined in `run_reports.py`. 
*   **TTR (Time-to-Response):** The duration from the "open time" to the first comment by a user that is not the author of the issue/PR.
*   **Status Counts:**
    *   **Opened:** Total issues/PRs created within the window.
    *   **Closed:** Issues/PRs closed (including those not merged).
    *   **Merged:** (PRs only) Successfully merged into the master branch


## Setup

This project uses `pyproject.toml` for dependency management. We recommend using `uv` for fast environment setup. Follow the steps below to set up your environment locally.

1.  **Clone the repository.**
2.  **Create and activate your virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```
    Or, alternatively, you can install with the development dependencies

    ```bash
    uv pip install -e ".[dev]"  # includes ipykernel for running .py files as interactive notebooks
    ```

## Using the Repository

### Automatic Tracking via GitHub Actions
The script runs automatically via GitHub Actions:
*   **Schedule:** Runs on `master` on the 1st of every month at ~ 8 AM Eastern Time
*   **On Push:** Runs on each push to branches that are not `master`
*   **Skip Fetch:** If you only want to test code changes without fetching the latest data, use the `[skip fetch]` keyword in your commit message

### Updating the Data Locally

To fetch the data from the GitHub API, you must have an active GitHub Personal Access Token with `repo` scope. If your token is missing or expired, the script will return a `401` error. You can verify your token is active and recognized by running:
```bash
curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user
```

**Note that you *only* need a Personal Access Token if you intend to fetch new data through the GitHub API**. Alternatively, if you do not need to fetch new data, you can always pull the data from the most recent GitHub Action workflow. You are then able to run the analysis/visualization scripts separately without needing an authentication token. 

If you *do* need to fetch new data, you must create a Personal Access Token with the proper scope and then export your token as an environment variable in your active terminal session. For example, on Mac/Linux, you could run the following command:

```bash
export GITHUB_TOKEN=your_token_here
```

If you have an active authentication token, the easiest way to update the data locally is to run the "orchestrator" script with the following command:

```bash
python ./issues_metrics/run_reports.py
```

This file does the following:
1. Runs the `download_issues.py` and `download_prs.py` modules to fetch any new issues/PRs
2. Run the `issues_analysis.py` and `prs_analysis.py` modules to generate  pickle files in `issues_metrics/report_data/` with up-to-date metrics


### Visualizing Performance Metrics

The "orchestrator" script (`run_reports.py`) is responsible for downloading the raw data from GitHub and preprocessing the data to generate pickle files containing our performance metrics, but it does *not* generate any of the visualizations used in our reports.

Visualizations are handled by the `issues_viewer.py` and `prs_viewer.py` scripts.  These scripts load the latest pickle files from `report_data/` and use those data to generate visualizations for our U24 reports and for tracking monthly performance measures. 

Remember that if you want to visualize the most up-to-date data, you should either pull the latest (updated monthly) pickle files from `master`, or run the entire pipeline locally as described above under the **Updating the Data Locally** header

To pull the latest data from master and generate all report visualizations, run the commands below from your terminal.
```bash
git pull upstream master
python ./issues_metrics/issues_viewer.py
python ./issues_metrics/prs_viewer.py
```

Alternatively, you may open either python file and run the file "cell by cell" in an interactive Jupyter environment (e.g., using an Interactive Window in VS Code), so long as you have installed the optional `ipykernel` dependency. 


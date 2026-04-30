import os

from issues_metrics import download_issues, download_prs, issues_analysis, prs_analysis


def main():

    # Fetch raw data from GitHub
    # ----------------------------------------

    skip_fetch = os.getenv("SKIP_FETCH", "false").lower() == "true"

    if skip_fetch:
        print("Skip data fetching from GitHub")
    else:
        print("\n"+"#"*50)
        print("Fetching issues data from GitHub")
        print("#"*50,"\n")
        download_issues.download_issues()

        print("\n"+"#"*50)
        print("Fetching PR data from GitHub")
        print("#"*50,"\n")
        download_prs.download_prs()

    # Peprocess raw data for reporting
    # ----------------------------------------

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

    print("\n"+"#"*50)
    print("Processing issues data for report")
    print("#"*50,"\n")
    try:
        issues_analysis.run_main_reports(dev_usernames=dev_usernames)
    except Exception as e:
        print(f"Issues Analysis failed: {e}")

    print("\n"+"#"*50)
    print("Processing PR data for report")
    print("#"*50,"\n")
    try:
        prs_analysis.run_main_reports(dev_usernames=dev_usernames)
    except Exception as e:
        print(f"PR Analysis failed: {e}")

if __name__ == "__main__":
    main()

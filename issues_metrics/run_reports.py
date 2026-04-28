from issues_metrics import issues_analysis, prs_analysis

if __name__ == "__main__":
    # set report parameters
    # ------------------------------
    start_date = False
    end_date = False
    display_tables = True
    style_displayed_tables = True
    save_report_data = True
    overwrite_historical_data = True

    # run PR report
    # ------------------------------
    # prs_analysis.run_report(
    #     end_date,
    #     display_tables,
    #     style_displayed_tables,
    #     save_report_data,
    #     overwrite_historical_data,
    # )

    # run issues report
    # ------------------------------
    issues_analysis.run_main_reports(
        start_date,
        end_date,
        display_tables,
        style_displayed_tables,
        save_report_data,
        overwrite_historical_data,
    )

# FlowCast Roadmap (due 15 Dec 2026)

Each step = one GitHub Issue + one feature branch + one Pull Request.

| Step | Branch | Deliverable | PRD ref | Target date |
|------|--------|-------------|---------|-------------|
| 1 | `feature/01-setup` | Repo, environment, config, GitHub | Wk 1 | 12 Nov |
| 2 | `feature/02-ingest` | Load + validate + quarantine (M1) | FR-01, FR-02 | 14 Nov |
| 3 | `feature/03-clean` | Dedup, outliers, missing, labels (M2) | FR-03 | 17 Nov |
| 4 | `feature/04-merge` | Merge weather + calendar, processed dataset | FR-04, FR-06 | 19 Nov |
| 5 | `feature/05-features-eda` | Features (M3), EDA notebook + data-quality report (M4) | FR-05 | 23 Nov |
| 6 | `feature/06-classical-ml` | From-scratch LR, DT, RF, XGB, SVM, scoreboard (M5) | FR-07, FR-08 | 30 Nov |
| 7 | `feature/07-lstm` | LSTM/GRU from scratch + benchmark (M6) | FR-09 | 6 Dec |
| 8 | `feature/08-evaluation` | Error analysis, model cards, confidence | FR-10..12 | 8 Dec |
| 9 | `feature/09-dashboard` | Streamlit, 9 views + upload/predict/report (M7) | FR-13..18 | 12 Dec |
| 10 | `feature/10-final` | README, final report, clean rerun | Wk 4 | 14 Dec |

Dates leave a 1-day buffer before the 15 Dec deadline.

Health Partners Forecasting

Machine Learning for Cash Flow Forecasting: A State-Space Approach to Weekly Appointment Cancellation Losses

A note on the data: everything in it is synthetic. The clients, appointments, prices and nominal codes were all invented because the real data is protected. The file data/DATA_CARD.md covers what is in the dataset and how it was put together.

What the project does

The project has three parts.

The first is a forecasting engine written in C# that predicts how many appointment cancellations will happen in the next one and two weeks. It works by looking at the shape of recent weeks, finding earlier periods in the history that looked similar, and using what happened after those periods to build a forecast.

The second is a step that converts those cancellation counts into income at risk, using each client’s own price for each appointment type.

The third is a Streamlit app with two tabs, Forecast and Database, that presents the results.

Every past forecast in the app was produced by standing at a past Monday and looking only at what was known at that point, then comparing it with what actually happened. The forecast range is worked out from the engine’s own track record: for each appointment type, it takes the middle 40% of the gaps between past forecasts and actual results, calculated separately for one-week and two-week forecasts. Types that are hard to predict get wider ranges; steadier types get narrower ones. The app shows below each chart how often the actual result fell inside the range.

Before every forecast run, the data goes through a set of checks covering duplicates, unrecognised clients and appointment types, prices, dates, the 48-hour cancellation rule, and whether any booking predates the client’s join date. If a check fails the run stops, and the report is written to outputs/data_checks.csv.

How the pieces fit together

data/ holds the database as CSV files.

engine/ is the C# forecasting engine. It reads from data/ and writes its output to outputs/, which holds both the current forecast and a full record of past forecasts alongside what actually happened.

app/dashboard.py is the Streamlit app. It reads from data/ and outputs/ and draws the charts and tables. It does not run any forecasting itself.

Getting it running (Windows, VS Code)

You need Python 3.10 or later and the .NET 8 SDK or later.

Open VS Code, choose File then Open Folder, and select this folder. Then go to Terminal, Run Task, and run the following in order:

Task 1 sets up Python. It creates a .venv environment and installs the required packages.

Task 2 builds the C# engine and runs its built-in checks.

Task 6 opens the app at http://localhost:8510.

The data and forecasts are already included in data/ and outputs/, so those three tasks are all you need to get the app running. Tasks 3 to 5 are for regenerating the forecast and running the test suite.

Commands

All commands follow the pattern python -m hpforecast followed by the command name. Add -h for help on any of them. The VS Code run tasks do the same things.

setup-engine --selftest compiles the C# engine and runs its built-in checks.

backtest validates the data, runs the engine across the full historical period in a walk-forward test, and then produces the live two-week forecast.

forecast re-runs the live two-week forecast on its own, which takes only a few seconds.

status reports what has been built and shows the headline numbers.

dashboard launches the app.

Folder structure

engine/ contains the C# prediction engine, covering data cleaning, fractal dimension calculation, state space construction, SVD, DTW distance and the weighted prediction step.

src/hpforecast/ is the Python layer. checks.py handles the data validation. pipeline.py and cli.py define and run the commands. engine_bridge.py calls the C# engine. evaluate.py computes weekly totals. views.py builds the tables shown in the app. config.py holds the project settings. llm.py, lookalikes.py and lookalike_eval.py cover the look-alike report feature and its evaluation. prompts/ holds the instructions sent to the AI.

app/ is the front end. dashboard.py is the app itself and style.css controls its appearance. Colours and light mode settings are in .streamlit/config.toml.

data/ holds the synthetic CSV database and the data card.

outputs/ holds the current forecast, the historical forecast record the app draws from, and the data check report.

tests/ contains automated tests for the engine, the data checks, the look-alike report and the app.

.vscode/ holds the run task definitions for VS Code.

The look-alike report (optional, needs an AI key)

As part of its method, the engine finds past weeks whose cancellation pattern closely resembles the most recent period. On the Forecast tab, below the chart for each appointment type, there is a Write a Report button. Clicking it sends a request to an AI model, which returns two paragraphs. The first describes what the matched weeks suggest about the forecast. The second considers whether those weeks genuinely share the same kind of real-world influences as the forecast week, such as the time of year, school holidays, bank holidays and seasonal health patterns, or whether they simply match in the numbers without sharing an underlying cause.

Nothing sensitive is sent to the AI. Only dates, match weights and calendar facts leave the system. No client names, client identifiers, prices or appointment records are included. The code checks that every date mentioned in the report is one that was sent to the AI, and that the response is exactly two paragraphs with no tables or percentages.

The button appears in both chart views. In the Next week view the report covers the one-week forecast. In the Next 2 weeks view it covers the combined two-week period. Because the engine builds its week 2 forecast on top of its week 1 prediction rather than observed data, week 2 has a separate set of matched weeks, saved in outputs/diagnostics_week2.json by the forecast command. The AI is told that the second week is forecast-based. The 20 evaluation test cases cover the next-week report only.

To use this feature, create a file called keys.env in the project root and add your API keys. The file keys.env.example shows the format. keys.env is excluded from version control and should be removed from the folder before sharing the project with anyone.

lookalike-check verifies that the keys and model names are valid. It makes no chargeable requests.

lookalike-cases regenerates the 20 test cases in outputs/lookalike_eval/cases.json by re-running the engine at a set of earlier Mondays.

lookalike-eval runs the writer on each test case and has a second OpenAI model act as a judge and score the results. Output goes to outputs/lookalike_eval/. Use --limit 3 to run a small sample first.

The instructions for the writer and the judge are plain text files in src/hpforecast/prompts/. Each ends with a Guardrails section that tells the model to treat all input as data rather than instructions, to stay focused on the task, never to guess or invent facts, to say “not known” when information is absent, and never to predict a specific outcome

Screenshot 1 — Database tab

The database explorer page. The top banner carries the project title. Below it are four summary cards: 200 clients, 10 appointment types, 391,344 cancellations total, and 261 weeks of history (roughly 5 years). The Clients sub-tab is open, showing a searchable, sortable table with columns for client number, name, industry, region, size, join date, total cancellations, 6-month cancellations, and income. A handful of clients are visible — a mix of industries including technology, security, construction and food.

Screenshot 2 — Forecast tab

The main forecasting view. The Health Partners logo sits in the header alongside the project title. The left sidebar lists all 11 appointment types with their forecast income at risk (Neurodiversity at £107k is the largest individual type; all types combined show £369k). The selected view is “All appointment types, next week w/c 2 Nov”.

The four metric cards show: 1,943 cancellations expected, £369k income at risk (highlighted in red), £190 average per cancellation, and £3.86M billed last week. The range shown is labelled “25% range” — that may mean the app was running before the 40% config change was picked up, or the Streamlit process needs a restart.

The backtest chart below shows the purple forecast line and shaded band against the black actual-outcome dots over roughly the past year, with Cancellations/Income and 6 months/1 year/2 years toggles.

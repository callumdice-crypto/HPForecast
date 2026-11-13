Health Partners Forecasting: data card

This is synthetic data. The project couldn’t use the real data because it’s protected, so this was built instead. There are no real patients, employees or clients in it. The client names are made up, and any similarity to an actual business is coincidence.

What’s in it

clients.csv holds 200 fictional clients, each with an industry, region, size, join date and price band.

appointment_types.csv lists the 10 types of appointment used across the project, each given a made-up nominal code.

price_list.csv holds what each client pays for each appointment type. Prices change on 1 April each year, which is when the annual price review happens.

cancellations.csv is the main file. It has 391,344 rows, one per cancelled appointment, covering 2021-11-01 to 2026-10-26. Each row records when the appointment was, when it was booked, when it was cancelled, how many hours’ notice was given, and what the client’s price was. Every row in this file is a cancellation that came in more than 48 hours before the appointment, because cancellations inside 48 hours are chargeable and are not treated as lost income. The income at risk for each row is the full price.

appointments.csv covers the last 26 weeks of the diary, with 450,204 rows in total, both attended and cancelled. The cancelled ones match rows in cancellations.csv and are linked by cancellation_id.

operating_calendar_weekly.csv shows how many working days each week has, based on Monday to Friday minus England and Wales bank holidays. It also flags school holiday weeks and runs 60 weeks beyond today.

Appointment types

T01, Physiotherapy, nominal code 4105, base price £62.00, 80,560 cancellation records
T02, Neurodiversity appointments, nominal code 4120, base price £313.35, 61,259 cancellation records
T03, Health Surveillance, nominal code 4135, base price £418.92, 43,629 cancellation records
T04, Pre-placement questionnaire, nominal code 4150, base price £20.21, 47,003 cancellation records
T05, Workstation assessment, nominal code 4165, base price £70.00, 30,843 cancellation records
T06, Therapy, nominal code 4180, base price £78.00, 41,323 cancellation records
T07, Workplace needs assessment, nominal code 4195, base price £367.90, 23,282 cancellation records
T08, Vaccinations, nominal code 4210, base price £38.00, 19,222 cancellation records
T09, Telephone Triage, nominal code 4225, base price £290.22, 30,114 cancellation records
T10, GP report, nominal code 4240, base price £69.91, 14,109 cancellation records

Volume

The weekly cancellation count averages around 1,529, with a low of 787 and a high of 1,991, spread across 256 weeks of records.

Deliberate problems

A few issues were put in on purpose so the data cleaning has something to find.

Two stretches have no records at all. One runs for four weeks from 2021-12-06. The other is the single week of 2024-06-10. Both represent a data-capture outage. Any week with no records at all is marked as missing rather than counted as zero. The four-week gap exceeds the two-week limit the engine uses for filling gaps, which means it breaks the history into two separate runs. The engine works from the longer, later stretch, which is why the forecast history only goes back about two and a half years.

The client base shifts over time. Around 30% of clients come on board part-way through the five years rather than being present from the start.

Prices rise every April, so the average price per cancellation drifts upward across the dataset.

How it was made

Each appointment type’s weekly counts were built from four ingredients: a growth trend, a seasonal pattern across the year, an adjustment for weeks with fewer working days due to bank holidays or school holidays, and a small amount of week-to-week carry-over with some added noise. It was generated in one pass using a fixed random seed of 2026, by a script that sits outside this project. The CSV files in this folder are the finished output.
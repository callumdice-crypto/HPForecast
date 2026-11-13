using System.Globalization;
using System.Text;
using HpForecast.Data;
using HpForecast.Engine;

namespace HpForecast.Cli;

/// <summary>
/// Walk-forward test. We pretend to stand at a past Monday (the "origin"), show the engine ONLY the data up to
/// then, ask for the next 1 and 2 weeks (the second one uses the first forecast as the new end point), then
/// move one week on and repeat. Nothing after the origin is ever shown.
///
///   dotnet run -c Release -- backtest --data ../../data --out ../../outputs/csharp_backtest.csv
/// Options: --label name  --horizon 2  --first-origin 2024-11-04  --last-origin 2026-10-19  --series T01,ALL
///          --raw-counts  --distance euclidean  --no-rotate  --fixed-window 13
/// </summary>
public static class BacktestCommand
{
    private const string Header =
        "label,series,origin,target_week,h,operating_days,forecast_count,actual_count,avg_price_gbp," +
        "forecast_revenue_gbp,actual_revenue_gbp,window,k,fallback,effective_neighbours,lb_skipped_share";

    public static int Run(Options o)
    {
        string dataDir = o.Get("data", "data");
        string outPath = o.Get("out", "outputs/csharp_backtest.csv");
        string label = o.Get("label", "State-space engine (C#)");
        int horizon = o.GetInt("horizon", 2);
        bool perOpDay = !o.Has("raw-counts");
        var cfg = o.Engine();
        var data = CancellationData.Load(dataDir);
        var series = o.Has("series") ? o.Get("series", "").Split(',') : data.SeriesNames.ToArray();
        var first = DateTime.Parse(o.Get("first-origin", "2024-11-04"), CultureInfo.InvariantCulture);
        var lastOrigin = o.Has("last-origin") ? DateTime.Parse(o.Get("last-origin", ""), CultureInfo.InvariantCulture)
                                              : data.Weeks[^2];
        var forecaster = new StateSpaceForecaster(cfg);

        var perSeries = new Dictionary<string, List<string>>();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Parallel.ForEach(series, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount) }, s =>
        {
            var rows = new List<string>();
            var counts = data.CountOf(s);
            var gross = data.GrossOf(s);
            var y = new double[counts.Length];
            for (int w = 0; w < y.Length; w++)
                y[w] = perOpDay ? counts[w] / data.OperatingDaysOn(data.Weeks[w]) : counts[w];

            for (int o2 = 0; o2 < data.Weeks.Length - 1; o2++)
            {
                var origin = data.Weeks[o2];
                if (origin < first || origin > lastOrigin) continue;
                if (double.IsNaN(y[o2])) continue;                                   // the origin week itself has no data
                ForecastResult[] res;
                try { res = forecaster.ForecastAhead(data.Weeks[..(o2 + 1)], y[..(o2 + 1)], horizon).ToArray(); }
                catch (DataRefusedException) { continue; }

                double price = data.AvgPrice(s, o2);
                for (int h = 1; h <= horizon; h++)
                {
                    int t = o2 + h;
                    if (t >= data.Weeks.Length) break;
                    int op = data.OperatingDaysOn(data.Weeks[t]);
                    double fc = perOpDay ? res[h - 1].Forecast * op : res[h - 1].Forecast;
                    var r = res[h - 1];
                    rows.Add(string.Join(",", new[]
                    {
                        Csv.Quote(label), s, origin.ToString("yyyy-MM-dd"), data.Weeks[t].ToString("yyyy-MM-dd"), h.ToString(),
                        op.ToString(), Csv.Num(fc), Csv.Num(counts[t]), Csv.Num(price), Csv.Num(fc * price),
                        Csv.Num(gross[t]), r.Window.ToString(), r.K.ToString(), r.UsedFallback ? "1" : "0",
                        Csv.Num(r.EffectiveNeighbours), Csv.Num((double)r.LbSkipped / Math.Max(r.NPatterns, 1)),
                    }));
                }
            }
            lock (perSeries) perSeries[s] = rows;
        });

        var sb = new StringBuilder(Header).AppendLine();
        foreach (var s in series) foreach (var l in perSeries[s]) sb.AppendLine(l);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, sb.ToString());
        Console.WriteLine($"[{label}] {perSeries.Values.Sum(l => l.Count):N0} forecast rows in {sw.Elapsed.TotalSeconds:F1}s -> {outPath}");
        return 0;
    }
}

using System.Globalization;
using System.Text;
using System.Text.Json;
using HpForecast.Data;
using HpForecast.Engine;

namespace HpForecast.Cli;

/// <summary>
/// The live forecast: standing at the latest week with data, forecast the next 2 weeks (the second week uses
/// the first forecast as the new end point), for every appointment type and for all types together.
/// Then turn counts into missed income using each client's own price.
///
///   dotnet run -c Release -- forecast --data ../../data --out ../../outputs
/// Writes: forecast_next_weeks.csv, forecast_by_client.csv, diagnostics_latest.json (the look-alikes behind week 1) and
/// diagnostics_week2.json (the look-alikes behind week 2: same layout; nothing in the forecasts changes)
/// </summary>
public static class ForecastCommand
{
    public static int Run(Options o)
    {
        string dataDir = o.Get("data", "data");
        string outDir = o.Get("out", "outputs");
        int horizon = o.GetInt("horizon", 2);
        bool perOpDay = !o.Has("raw-counts");
        var cfg = o.Engine();
        var data = CancellationData.Load(dataDir);
        var forecaster = new StateSpaceForecaster(cfg);
        Directory.CreateDirectory(outDir);

        int origin = data.Weeks.Length - 1;
        while (double.IsNaN(data.CountOf(WeeklyData.All)[origin])) origin--;

        var summary = new StringBuilder(
            "series,series_name,origin,target_week,h,operating_days,forecast_count,avg_price_gbp,forecast_missed_revenue_gbp,window,fallback\n");
        var clientRows = new StringBuilder("client_id,client_name,type_id,type_name,target_week,expected_cancellations,price_gbp,expected_missed_revenue_gbp\n");
        var diag = new Dictionary<string, object>();
        var diag2 = new Dictionary<string, object>();        // the same working for week 2 (the report on the 2-week forecast explains it)
        var counts = new Dictionary<string, double[]>();     // type -> forecast count per h (for client allocation)

        foreach (var s in data.SeriesNames)
        {
            var c = data.CountOf(s);
            var y = new double[origin + 1];
            for (int w = 0; w <= origin; w++) y[w] = perOpDay ? c[w] / data.OperatingDaysOn(data.Weeks[w]) : c[w];
            var res = forecaster.ForecastAhead(data.Weeks[..(origin + 1)], y, horizon);

            double price = data.AvgPrice(s, origin);
            var fcCounts = new double[horizon];
            for (int h = 1; h <= horizon; h++)
            {
                var target = data.Weeks[origin].AddDays(7 * h);
                int op = data.OperatingDaysOn(target);
                double fc = perOpDay ? res[h - 1].Forecast * op : res[h - 1].Forecast;
                fcCounts[h - 1] = fc;
                summary.AppendLine(string.Join(",", new[]
                {
                    s, "\"" + data.NameOf(s) + "\"", data.Weeks[origin].ToString("yyyy-MM-dd"), target.ToString("yyyy-MM-dd"),
                    h.ToString(), op.ToString(), Csv.Num(fc), Csv.Num(price), Csv.Num(fc * price),
                    res[h - 1].Window.ToString(), res[h - 1].UsedFallback ? "1" : "0",
                }));
            }
            if (s != WeeklyData.All) counts[s] = fcCounts;
            diag[s] = Describe(res[0], data, s);
            if (horizon >= 2) diag2[s] = Describe(res[1], data, s);
            if (s == WeeklyData.All) WriteStateSpace(Path.Combine(outDir, "statespace_all.csv"), res[0]);
        }

        // Share each type's forecast out to clients (by their share of the last 26 weeks) at their CURRENT price
        foreach (var (type, fcCounts) in counts)
        {
            double typeTotal = data.ClientMix.Where(k => k.Key.Type == type).Sum(k => k.Value.Count);
            if (typeTotal <= 0) continue;
            string typeName = data.NameOf(type);
            foreach (var (key, v) in data.ClientMix.Where(k => k.Key.Type == type))
            {
                double share = v.Count / typeTotal;
                for (int h = 1; h <= fcCounts.Length; h++)
                {
                    var target = data.Weeks[origin].AddDays(7 * h);
                    double price = data.PriceFor(key.Client, type, target);
                    double exp = fcCounts[h - 1] * share;
                    clientRows.AppendLine(string.Join(",", new[]
                    {
                        key.Client, "\"" + data.ClientNames[key.Client] + "\"", type, "\"" + typeName + "\"",
                        target.ToString("yyyy-MM-dd"), Csv.Num(Math.Round(exp, 4)), Csv.Num(price), Csv.Num(Math.Round(exp * price, 4)),
                    }));
                }
            }
        }

        File.WriteAllText(Path.Combine(outDir, "forecast_next_weeks.csv"), summary.ToString());
        File.WriteAllText(Path.Combine(outDir, "forecast_by_client.csv"), clientRows.ToString());
        File.WriteAllText(Path.Combine(outDir, "diagnostics_latest.json"),
            JsonSerializer.Serialize(diag, new JsonSerializerOptions { WriteIndented = false }));
        if (diag2.Count > 0)
            File.WriteAllText(Path.Combine(outDir, "diagnostics_week2.json"),
                JsonSerializer.Serialize(diag2, new JsonSerializerOptions { WriteIndented = false }));
        Console.WriteLine($"Forecast made from week {data.Weeks[origin]:yyyy-MM-dd} for the next {horizon} weeks -> {outDir}");
        return 0;
    }

    /// <summary>The state-space picture: every earlier pattern on its first two SVD directions, its neighbours and the newest pattern.</summary>
    private static void WriteStateSpace(string path, ForecastResult r)
    {
        var sb = new StringBuilder("kind,end_date,svd1,svd2,outcome_z,weight\n");
        var nb = r.NeighbourPattern.Select((p, i) => (p, i)).ToDictionary(t => t.p, t => r.NeighbourWeight[t.i]);
        for (int i = 0; i < r.PatternEndPos.Length; i++)
            sb.AppendLine(string.Join(",", new[]
            {
                nb.ContainsKey(i) ? "neighbour" : "pattern", r.Cleaned.Dates[r.PatternEndPos[i]].ToString("yyyy-MM-dd"),
                Csv.Num(r.PatternCoords[i][0]), Csv.Num(r.PatternCoords[i][1]), Csv.Num(r.PatternOutcome[i]),
                Csv.Num(nb.TryGetValue(i, out var w) ? w : 0.0),
            }));
        sb.AppendLine(string.Join(",", new[]
        {
            "newest", r.LastDate.ToString("yyyy-MM-dd"), Csv.Num(r.NewestCoords[0]), Csv.Num(r.NewestCoords[1]), "", "0",
        }));
        File.WriteAllText(path, sb.ToString());
    }

    /// <summary>The working behind one forecast, for the teaching charts.</summary>
    private static object Describe(ForecastResult r, WeeklyData data, string series)
    {
        var vals = r.Cleaned.Values;
        int n = r.Window;
        var neigh = new List<object>();
        for (int i = 0; i < r.NeighbourEndPos.Length; i++)
        {
            int end = r.NeighbourEndPos[i];
            var w = new double[n];
            Array.Copy(vals, end - n + 1, w, 0, n);
            var (z, _, _) = StateSpaceForecaster.ZScore(w);
            neigh.Add(new
            {
                end_date = r.Cleaned.Dates[end].ToString("yyyy-MM-dd"), z, outcome = r.NeighbourOutcome[i],
                weight = r.NeighbourWeight[i], distance = r.NeighbourDist[i],
            });
        }
        return new
        {
            series, name = data.NameOf(series), window = n, k = r.K, n_patterns = r.NPatterns,
            n_points = r.Cleaned.Values.Length, n_filled = r.Cleaned.NFilled, n_gap_weeks = r.Cleaned.NGapWeeks,
            latest_run_length = r.Cleaned.LatestRunLength, window_mean = r.WindowMean, window_sd = r.WindowSd,
            forecast_per_day = r.Forecast, fd_used_fallback = r.Fd.UsedFallback, fixed_window = double.IsNaN(r.Fd.FdMid),
            fd_lags = r.Fd.Lags, fd = r.Fd.Fd.Select(v => double.IsNaN(v) ? (double?)null : v).ToArray(),
            elbow_lag = r.Fd.ElbowLag, max_lag = r.Fd.MaxLag, fd_mid = double.IsNaN(r.Fd.FdMid) ? (double?)null : r.Fd.FdMid,
            newest_z = r.NewestZ, z_pred = r.ZPred, used_fallback = r.UsedFallback,
            effective_neighbours = r.EffectiveNeighbours, lb_skipped = r.LbSkipped,
            last_date = r.LastDate.ToString("yyyy-MM-dd"), neighbours = neigh,
        };
    }
}

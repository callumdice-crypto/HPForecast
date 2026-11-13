using System.Globalization;
using System.Text;

namespace HpForecast.Data;

/// <summary>Weekly totals built from the cancellations table (the fake database), by appointment type.</summary>
public sealed class WeeklyData
{
    public DateTime[] Weeks { get; init; } = Array.Empty<DateTime>();
    public string[] TypeIds { get; init; } = Array.Empty<string>();
    public string[] TypeNames { get; init; } = Array.Empty<string>();
    /// <summary>[type][week] cancellations. NaN where the whole week has no records (a data-capture gap).</summary>
    public double[][] Count { get; init; } = Array.Empty<double[]>();
    /// <summary>[type][week] missed income at the client's price (sum of price of each cancelled appointment), GBP.</summary>
    public double[][] Gross { get; init; } = Array.Empty<double[]>();
    /// <summary>Operating days for every week in the calendar file (history AND future weeks).</summary>
    public Dictionary<DateTime, int> OperatingDays { get; init; } = new();
    /// <summary>Last 26 weeks, by client and type: cancellations and missed income (used to share a type forecast out to clients).</summary>
    public Dictionary<(string Client, string Type), (double Count, double Gross)> ClientMix { get; init; } = new();
    public Dictionary<string, string> ClientNames { get; init; } = new();
    /// <summary>Every price change per client and type, oldest first.</summary>
    public Dictionary<(string Client, string Type), List<(DateTime From, double Price)>> Prices { get; init; } = new();

    public const string All = "ALL";

    public IEnumerable<string> SeriesNames => TypeIds.Prepend(All);

    public double[] CountOf(string series) => series == All ? SumAcross(Count) : Count[Array.IndexOf(TypeIds, series)];
    public double[] GrossOf(string series) => series == All ? SumAcross(Gross) : Gross[Array.IndexOf(TypeIds, series)];
    public string NameOf(string series) => series == All ? "All appointment types" : TypeNames[Array.IndexOf(TypeIds, series)];

    private static double[] SumAcross(double[][] a)
    {
        var s = new double[a[0].Length];
        for (int w = 0; w < s.Length; w++)
        {
            double t = 0;
            foreach (var row in a) t += row[w];          // NaN stays NaN (a missing week is missing for every type)
            s[w] = t;
        }
        return s;
    }

    public int OperatingDaysOn(DateTime week) =>
        OperatingDays.TryGetValue(week, out var d) ? d : 5;

    public double PriceFor(string client, string type, DateTime on)
    {
        if (!Prices.TryGetValue((client, type), out var list)) return double.NaN;
        double p = list[0].Price;
        foreach (var (from, price) in list) if (from <= on) p = price; else break;
        return p;
    }

    /// <summary>
    /// Average missed income per cancelled appointment over the trailing `weeks` weeks ending at index `origin`
    /// (weeks with no records are skipped).
    /// </summary>
    public double AvgPrice(string series, int origin, int weeks = 13)
    {
        var c = CountOf(series);
        var g = GrossOf(series);
        double sc = 0, sg = 0;
        for (int w = Math.Max(0, origin - weeks + 1); w <= origin; w++)
        {
            if (double.IsNaN(c[w])) continue;
            sc += c[w]; sg += g[w];
        }
        return sc > 0 ? sg / sc : double.NaN;
    }
}

public static class CancellationData
{
    /// <summary>
    /// Every row of cancellations.csv is a cancellation made MORE THAN 48 HOURS before the appointment (inside 48 hours
    /// it becomes chargeable, so those are not in this data). The slot is lost, so the income lost is the full price.
    /// </summary>
    public static WeeklyData Load(string dataDir)
    {
        var typesRows = Csv.Read(Path.Combine(dataDir, "appointment_types.csv"));
        var typeIds = typesRows.Select(r => r["type_id"]).ToArray();
        var typeNames = typesRows.Select(r => r["type_name"]).ToArray();
        var typeIndex = typeIds.Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i);

        // First pass: find the week range
        var path = Path.Combine(dataDir, "cancellations.csv");
        var (header, lines) = Csv.ReadLines(path);
        int cWeek = Array.IndexOf(header, "week_start"), cType = Array.IndexOf(header, "type_id"),
            cClient = Array.IndexOf(header, "client_id"), cPrice = Array.IndexOf(header, "price_gbp"),
            cNotice = Array.IndexOf(header, "notice_hours");
        if (new[] { cWeek, cType, cClient, cPrice, cNotice }.Any(i => i < 0))
            throw new InvalidDataException("cancellations.csv is missing an expected column.");

        var dateCache = new Dictionary<string, DateTime>();
        DateTime min = DateTime.MaxValue, max = DateTime.MinValue;
        var parsed = new List<string[]>(lines.Count);
        foreach (var line in lines)
        {
            var f = Csv.SplitLine(line);
            if (!dateCache.TryGetValue(f[cWeek], out var d))
                dateCache[f[cWeek]] = d = DateTime.ParseExact(f[cWeek], "yyyy-MM-dd", CultureInfo.InvariantCulture);
            if (d < min) min = d;
            if (d > max) max = d;
            parsed.Add(f);
        }
        int nWeeks = (max - min).Days / 7 + 1;
        var weeks = Enumerable.Range(0, nWeeks).Select(i => min.AddDays(7 * i)).ToArray();

        int nTypes = typeIds.Length;
        var count = Enumerable.Range(0, nTypes).Select(_ => new double[nWeeks]).ToArray();
        var gross = Enumerable.Range(0, nTypes).Select(_ => new double[nWeeks]).ToArray();
        var anyRecords = new int[nWeeks];
        var mix = new Dictionary<(string, string), (double, double)>();
        int mixFrom = nWeeks - 26;

        foreach (var f in parsed)
        {
            int w = (dateCache[f[cWeek]] - min).Days / 7;
            anyRecords[w]++;
            if (double.Parse(f[cNotice], CultureInfo.InvariantCulture) <= 48) continue;     // inside 48 hours = chargeable, not a cancellation here
            int t = typeIndex[f[cType]];
            double price = double.Parse(f[cPrice], CultureInfo.InvariantCulture);
            count[t][w] += 1;
            gross[t][w] += price;
            if (w >= mixFrom)
            {
                var key = (f[cClient], f[cType]);
                mix.TryGetValue(key, out var cur);
                mix[key] = (cur.Item1 + 1, cur.Item2 + price);
            }
        }
        // A week with no records at all is MISSING (a capture gap), not zero
        for (int w = 0; w < nWeeks; w++)
            if (anyRecords[w] == 0)
                for (int t = 0; t < nTypes; t++) { count[t][w] = double.NaN; gross[t][w] = double.NaN; }

        var opDays = new Dictionary<DateTime, int>();
        foreach (var r in Csv.Read(Path.Combine(dataDir, "operating_calendar_weekly.csv")))
            opDays[DateTime.ParseExact(r["week_start"], "yyyy-MM-dd", CultureInfo.InvariantCulture)] = int.Parse(r["operating_days"]);

        var names = Csv.Read(Path.Combine(dataDir, "clients.csv")).ToDictionary(r => r["client_id"], r => r["client_name"]);
        var prices = new Dictionary<(string, string), List<(DateTime, double)>>();
        foreach (var r in Csv.Read(Path.Combine(dataDir, "price_list.csv")))
        {
            var key = (r["client_id"], r["type_id"]);
            if (!prices.TryGetValue(key, out var list)) prices[key] = list = new();
            list.Add((DateTime.ParseExact(r["effective_from"], "yyyy-MM-dd", CultureInfo.InvariantCulture),
                      double.Parse(r["price_gbp"], CultureInfo.InvariantCulture)));
        }
        foreach (var l in prices.Values) l.Sort((a, b) => a.Item1.CompareTo(b.Item1));

        return new WeeklyData
        {
            Weeks = weeks, TypeIds = typeIds, TypeNames = typeNames, Count = count, Gross = gross,
            OperatingDays = opDays, ClientMix = mix, ClientNames = names, Prices = prices,
        };
    }
}

/// <summary>A tiny CSV reader (handles quoted fields). Enough for these files; no NuGet needed.</summary>
public static class Csv
{
    public static (string[] Header, List<string> Lines) ReadLines(string path)
    {
        var all = File.ReadLines(path).ToList();
        return (SplitLine(all[0]), all.Skip(1).Where(l => l.Length > 0).ToList());
    }

    public static List<Dictionary<string, string>> Read(string path)
    {
        var (header, lines) = ReadLines(path);
        var rows = new List<Dictionary<string, string>>(lines.Count);
        foreach (var l in lines)
        {
            var f = SplitLine(l);
            var d = new Dictionary<string, string>(header.Length);
            for (int i = 0; i < header.Length && i < f.Length; i++) d[header[i]] = f[i];
            rows.Add(d);
        }
        return rows;
    }

    public static string[] SplitLine(string line)
    {
        if (line.IndexOf('"') < 0) return line.Split(',');
        var fields = new List<string>();
        var sb = new StringBuilder();
        bool quoted = false;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (quoted)
            {
                if (c == '"' && i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                else if (c == '"') quoted = false;
                else sb.Append(c);
            }
            else if (c == '"') quoted = true;
            else if (c == ',') { fields.Add(sb.ToString()); sb.Clear(); }
            else sb.Append(c);
        }
        fields.Add(sb.ToString());
        return fields.ToArray();
    }

    public static string Num(double v) => double.IsNaN(v) ? "" : v.ToString("R", CultureInfo.InvariantCulture);

    /// <summary>Wraps text in quotes when it contains a comma, quote or line break, so it stays in one column.</summary>
    public static string Quote(string s) =>
        s.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0 ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
}

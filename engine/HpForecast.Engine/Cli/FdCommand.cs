using System.Globalization;
using System.Text;
using HpForecast.Data;
using HpForecast.Engine;

namespace HpForecast.Cli;

/// <summary>
/// Fractal dimension against lag (the Python script, in C#) for the charts and for checking against the original.
///
///   dotnet run -c Release -- fd --data ../../data --out ../../outputs/fd_curves.csv [--origin 2026-10-26] [--target all]
///   dotnet run -c Release -- fd --in numbers.txt --out fd.csv          (one number per line)
///
/// Writes one row per series and lag:  series,origin,lag,fd,elbow_lag,max_lag,fd_mid,window,fallback
/// For --data the series is the weekly count per operating day, cleaned exactly as the forecast cleans it.
/// </summary>
public static class FdCommand
{
    public static int Run(Options o)
    {
        string outPath = o.Get("out", "outputs/fd_curves.csv");
        var cfg = o.Engine();
        var sb = new StringBuilder("series,origin,lag,fd,elbow_lag,max_lag,fd_mid,window,fallback\n");

        if (o.Has("in"))
        {
            var values = File.ReadAllLines(o.Get("in", "")).Where(l => l.Trim().Length > 0)
                             .Select(l => double.Parse(l, CultureInfo.InvariantCulture)).ToArray();
            Append(sb, "input", "", FractalDimension.Compute(values, new List<(int, int)> { (0, values.Length) }, cfg));
        }
        else
        {
            var data = CancellationData.Load(o.Get("data", "data"));
            int origin = data.Weeks.Length - 1;
            while (double.IsNaN(data.CountOf(WeeklyData.All)[origin])) origin--;
            if (o.Has("origin"))
            {
                var want = DateTime.Parse(o.Get("origin", ""), CultureInfo.InvariantCulture);
                origin = Array.FindIndex(data.Weeks, w => w == want);
                if (origin < 0) throw new InvalidDataException($"No week starting {want:yyyy-MM-dd} in the data.");
            }
            bool perOpDay = !o.Has("raw-counts");
            foreach (var s in data.SeriesNames)
            {
                var c = data.CountOf(s);
                var y = new double[origin + 1];
                for (int w = 0; w <= origin; w++) y[w] = perOpDay ? c[w] / data.OperatingDaysOn(data.Weeks[w]) : c[w];
                var cleaned = DataCleaner.Clean(data.Weeks[..(origin + 1)], y, cfg);
                var fd = FractalDimension.Compute(cleaned.Values, cleaned.Runs(), cfg);
                Append(sb, s, data.Weeks[origin].ToString("yyyy-MM-dd"), fd);
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, sb.ToString());
        Console.WriteLine($"Fractal dimension curves -> {outPath}");
        return 0;
    }

    private static void Append(StringBuilder sb, string series, string origin, FdResult fd)
    {
        for (int i = 0; i < fd.Lags.Length; i++)
            sb.AppendLine(string.Join(",", new[]
            {
                series, origin, ((int)fd.Lags[i]).ToString(), Csv.Num(fd.Fd[i]), fd.ElbowLag.ToString(), fd.MaxLag.ToString(),
                Csv.Num(fd.FdMid), fd.Window.ToString(), fd.UsedFallback ? "1" : "0",
            }));
    }
}

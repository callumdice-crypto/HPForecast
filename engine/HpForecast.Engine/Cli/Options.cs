using HpForecast.Engine;

namespace HpForecast.Cli;

/// <summary>Reads "--name value" and "--flag" arguments, and builds an EngineConfig from them.</summary>
public sealed class Options
{
    private readonly Dictionary<string, string> _v = new(StringComparer.OrdinalIgnoreCase);

    public Options(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--")) continue;
            string key = args[i][2..];
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--")) _v[key] = args[++i];
            else _v[key] = "true";
        }
    }

    public string Get(string key, string fallback) => _v.TryGetValue(key, out var s) ? s : fallback;
    public int GetInt(string key, int fallback) => _v.TryGetValue(key, out var s) ? int.Parse(s) : fallback;
    public double GetDouble(string key, double fallback) =>
        _v.TryGetValue(key, out var s) ? double.Parse(s, System.Globalization.CultureInfo.InvariantCulture) : fallback;
    public bool Has(string key) => _v.ContainsKey(key);

    public EngineConfig Engine()
    {
        var cfg = new EngineConfig();
        if (Get("distance", "dtw").Equals("euclidean", StringComparison.OrdinalIgnoreCase))
            cfg = cfg with { Distance = DistanceKind.Euclidean };
        if (Has("no-rotate")) cfg = cfg with { Rotate = false };
        if (Has("no-lb")) cfg = cfg with { UseLbKeogh = false };
        if (Has("fixed-window")) cfg = cfg with { FixedWindow = GetInt("fixed-window", 8) };
        string cum = Get("cumulative", "none");
        if (cum.Equals("low", StringComparison.OrdinalIgnoreCase)) cfg = cfg with { Cumulative = CumulativeKind.Low };
        else if (cum.Equals("high", StringComparison.OrdinalIgnoreCase)) cfg = cfg with { Cumulative = CumulativeKind.High };
        string predict = Get("predict", "vector");
        if (predict.StartsWith("next", StringComparison.OrdinalIgnoreCase)) cfg = cfg with { Target = PredictionTarget.NextValueZ };
        else if (predict.Contains("literal", StringComparison.OrdinalIgnoreCase)) cfg = cfg with { Target = PredictionTarget.VectorLastCoordinate };
        return cfg;
    }
}

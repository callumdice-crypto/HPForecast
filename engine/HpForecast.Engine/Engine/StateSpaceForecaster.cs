namespace HpForecast.Engine;

/// <summary>Everything about one forecast: the number, and the working behind it (for charts and teaching).</summary>
public sealed class ForecastResult
{
    public double Forecast { get; init; }              // the forecast value, in the same units as the input series
    public double ZPred { get; init; }                 // predicted z-score of the next value
    public int Window { get; init; }                   // N
    public int K { get; init; }                        // neighbours used
    public int NPatterns { get; init; }                // earlier patterns available
    public bool UsedFallback { get; init; }            // true if the weighted-average fallback was used
    public string FallbackReason { get; init; } = "";
    public double EffectiveNeighbours { get; init; }   // (sum w)^2 / sum w^2: how many neighbours really carry the weight
    public int LbSkipped { get; init; }                // candidates skipped by LB_Keogh
    public double WindowMean { get; init; }
    public double WindowSd { get; init; }
    public FdResult Fd { get; init; } = null!;
    public CleanedSeries Cleaned { get; init; } = null!;
    public DateTime LastDate { get; init; }
    public double[] NewestZ { get; init; } = Array.Empty<double>();
    public int[] NeighbourEndPos { get; init; } = Array.Empty<int>();
    public int[] NeighbourPattern { get; init; } = Array.Empty<int>();   // index into the pattern list
    public double[] NeighbourDist { get; init; } = Array.Empty<double>();
    public double[] NeighbourWeight { get; init; } = Array.Empty<double>();
    public double[] NeighbourOutcome { get; init; } = Array.Empty<double>();
    // For the state-space picture: the first two SVD coordinates of every earlier pattern, and of the newest one
    public int[] PatternEndPos { get; init; } = Array.Empty<int>();
    public double[][] PatternCoords { get; init; } = Array.Empty<double[]>();
    public double[] PatternOutcome { get; init; } = Array.Empty<double>();
    public double[] NewestCoords { get; init; } = Array.Empty<double>();
}

/// <summary>One row of the state space: a window of N values, z-scored on its own.</summary>
public sealed record StateRow(double[] Z, double Mean, double Sd, int End, bool HasNext);

/// <summary>
/// The forecasting engine: a state-space nearest-neighbour model (a "lazy" machine-learning method).
/// It is the first prediction of the Vector code (CreateCumulativeSvdEmbedding -> FindKNearestNeighbours ->
/// CalculateDeltaPredictionNoSVD), run on weekly cancellation counts, with the fractal-dimension script
/// choosing the window. Everything is C#.
///
///  0. Clean the history (DataCleaner).
///  1. Choose the window size N from the fractal-dimension-vs-lag curve (FractalDimension).
///  2. State space: every point gives a "pattern" = the last N values ending on it. Each pattern is z-scored
///     on its own with the POPULATION standard deviation (a flat window becomes zeros, sd stored as 1).
///     Each earlier pattern's outcome is the NEXT value on the same z-score scale. The newest pattern has no
///     outcome: it is the one we forecast.
///  3. SVD rotation: stack ALL patterns (including the newest), run an SVD, rotate them into the SVD directions
///     (Vector: U x diag(S), which equals the patterns times V).
///  4. Neighbours: compare the newest pattern with every earlier one by DTW (cost |a-b|, band 70%) with the
///     LB_Keogh test (envelope 15%) to skip candidates. k = (int) max(N, 2.5% of the number of rows).
///  5. Prediction: Gaussian weights exp(-d^2 / sigma^2), sigma = half the median neighbour distance (all weights 1
///     if sigma is below 1e-9). Weighted pseudo-inverse map, no intercept, applied to the newest pattern; if the
///     answer is not finite (or implausible) fall back to the weighted average of the neighbours' outcomes.
///     De-normalise:  forecast = z * (newest window sd) + (newest window mean).
///
/// Only values up to the forecast point are ever used, so there is no look-ahead.
///
/// TWO WEEKS AT A TIME: ForecastAhead(steps) forecasts week 1, then adds that forecast to the history as a new
/// point (the new END POINT of the newest pattern), and forecasts week 2 from there. The window size N is
/// recalculated at every step, as the spec says.
/// </summary>
public sealed class StateSpaceForecaster
{
    private readonly EngineConfig _cfg;
    public StateSpaceForecaster(EngineConfig? cfg = null) => _cfg = cfg ?? new EngineConfig();

    /// <summary>Vector's embedding of one window: optional running min/max, then mean, POPULATION sd, z-scores.</summary>
    public static (double[] Z, double Mean, double Sd) MakeRow(double[] window, CumulativeKind kind = CumulativeKind.None)
    {
        var w = (double[])window.Clone();
        if (kind == CumulativeKind.Low)
        {
            double cur = double.PositiveInfinity;
            for (int j = 0; j < w.Length; j++) { cur = Math.Min(cur, w[j]); w[j] = cur; }
        }
        else if (kind == CumulativeKind.High)
        {
            double cur = double.NegativeInfinity;
            for (int j = 0; j < w.Length; j++) { cur = Math.Max(cur, w[j]); w[j] = cur; }
        }
        double mean = w.Average();
        double sumSq = 0;
        foreach (var x in w) sumSq += (x - mean) * (x - mean);
        double sd = Math.Sqrt(sumSq / w.Length);
        var z = new double[w.Length];
        if (sd >= 1e-10)
            for (int j = 0; j < w.Length; j++) z[j] = (w[j] - mean) / sd;
        return (z, mean, sd < 1e-10 ? 1.0 : sd);            // Vector stores sd = 1 for a flat window
    }

    public static (double[] Z, double Mean, double Sd) ZScore(double[] w) => MakeRow(w, CumulativeKind.None);

    /// <summary>Step 2. Windows never cross a gap. The last row of the latest run (HasNext = false) is the newest pattern.</summary>
    public static List<StateRow> BuildStateSpace(double[] values, List<(int Start, int End)> runs, int n, CumulativeKind kind = CumulativeKind.None)
    {
        var rows = new List<StateRow>();
        foreach (var (s, e) in runs)
            for (int i = s + n - 1; i < e; i++)
            {
                var w = new double[n];
                Array.Copy(values, i - n + 1, w, 0, n);
                var (z, mean, sd) = MakeRow(w, kind);
                rows.Add(new StateRow(z, mean, sd, i, i + 1 < e));
            }
        return rows;
    }

    public ForecastResult ForecastNext(DateTime[] dates, double[] values)
    {
        var cfg = _cfg;
        var cleaned = DataCleaner.Clean(dates, values, cfg);                                   // step 0
        var runs = cleaned.Runs();
        var fd = cfg.FixedWindow.HasValue ? FractalDimension.Fixed(cfg)                       // step 1
                                          : FractalDimension.Compute(cleaned.Values, runs, cfg);
        int n = fd.Window;

        var rows = BuildStateSpace(cleaned.Values, runs, n, cfg.Cumulative);                  // step 2
        var newestRow = rows[^1];
        if (newestRow.HasNext) throw new InvalidOperationException("The newest pattern must be the last point of the latest run.");
        var patIdx = Enumerable.Range(0, rows.Count).Where(i => rows[i].HasNext).ToArray();   // rows that have a known outcome
        if (patIdx.Length < Math.Max(n + 1, 10))
            throw new DataRefusedException($"Only {patIdx.Length} earlier patterns for window {n}: not enough to forecast.");

        double[][] rot;                                                                        // step 3
        if (cfg.Rotate)
        {
            var stack = new double[rows.Count, n];                                             // ALL windows, newest included
            for (int i = 0; i < rows.Count; i++) for (int j = 0; j < n; j++) stack[i, j] = rows[i].Z[j];
            var (_, v) = LinearAlgebra.Svd(stack);
            rot = LinearAlgebra.RotateRows(rows.Select(r => r.Z).ToArray(), v);
        }
        else rot = rows.Select(r => r.Z).ToArray();
        var cand = patIdx.Select(i => rot[i]).ToArray();
        var newestRot = rot[^1];

        int k = (int)Math.Max(n, rows.Count * cfg.KMinFrac);                                   // step 4  (Vector: (int)Math.Max(dim, totalRows * 0.025))
        k = Math.Min(k, cand.Length);
        (int[] hit, double[] dist, int skipped) = cfg.Distance == DistanceKind.Dtw
            ? Dtw.KnnDtw(newestRot, cand, k, cfg.BandFrac, cfg.EnvelopeFrac, cfg.UseLbKeogh)
            : Dtw.KnnEuclidean(newestRot, cand, k);
        if (hit.Length < n) throw new DataRefusedException($"Only {hit.Length} neighbours found for window {n}.");

        // step 5 - weights, weighted map, fallback, de-normalise
        double sigma = cfg.SigmaFrac * Median(dist);
        var w = new double[dist.Length];
        for (int i = 0; i < w.Length; i++) w[i] = sigma < 1e-9 ? 1.0 : Math.Exp(-(dist[i] * dist[i]) / (sigma * sigma));
        double wsum = w.Sum();
        if (!(wsum > 0) || double.IsNaN(wsum)) { Array.Fill(w, 1.0); wsum = w.Length; }
        double eff = wsum * wsum / w.Sum(x => x * x);

        // What each neighbour "became" (the last number of its next state):
        //  VectorRotatedBack     the last number of its next window in the ORIGINAL week positions (= the newest week).
        //                        This is Vector's "map the whole state, read the last coordinate" with the rotation undone.
        //  VectorLastCoordinate  the last coordinate of its next state in SVD space, as Vector is written (always ~0).
        //  NextValueZ            the next value as a z-score of the neighbour's OWN window (crypto-edge port).
        var target = cfg.Target;
        var y = new double[hit.Length];
        for (int i = 0; i < hit.Length; i++)
        {
            int r = patIdx[hit[i]];
            y[i] = target switch
            {
                PredictionTarget.VectorLastCoordinate => rot[r + 1][n - 1],
                PredictionTarget.VectorRotatedBack => rows[r + 1].Z[n - 1],
                _ => (cleaned.Values[rows[r].End + 1] - rows[r].Mean) / rows[r].Sd,
            };
        }
        double fallbackValue = 0;
        for (int i = 0; i < y.Length; i++) fallbackValue += w[i] / wsum * y[i];

        bool usedFallback = false;
        string reason = "";
        double z;
        try
        {
            var xw = new double[hit.Length, n];            // Vector multiplies the columns of P and F by W (so a point counts w^2 in the fit)
            var yw = new double[hit.Length];
            for (int i = 0; i < hit.Length; i++)
            {
                for (int j = 0; j < n; j++) xw[i, j] = cand[hit[i]][j] * w[i];
                yw[i] = y[i] * w[i];
            }
            var beta = LinearAlgebra.PseudoInverseSolve(xw, yw);
            z = 0;
            for (int j = 0; j < n; j++) z += newestRot[j] * beta[j];
            if (!double.IsFinite(z)) throw new ArithmeticException("non-finite prediction");
            if (z < y.Min() - cfg.PlausibilityMargin || z > y.Max() + cfg.PlausibilityMargin)          // spec step 5: an implausible answer falls back to the weighted average
            {
                usedFallback = true;
                reason = "implausible (outside neighbours' outcome range +/- margin)";
                z = fallbackValue;
            }
        }
        catch (Exception ex)
        {
            usedFallback = true;
            reason = "fit failed: " + ex.Message;
            z = fallbackValue;
        }

        double forecast = Math.Max(0.0, z * newestRow.Sd + newestRow.Mean);                    // counts cannot be negative
        return new ForecastResult
        {
            Forecast = forecast, ZPred = z, Window = n, K = hit.Length, NPatterns = patIdx.Length,
            UsedFallback = usedFallback, FallbackReason = reason, EffectiveNeighbours = eff, LbSkipped = skipped,
            WindowMean = newestRow.Mean, WindowSd = newestRow.Sd, Fd = fd, Cleaned = cleaned, LastDate = cleaned.Dates[^1],
            NewestZ = newestRow.Z,
            NeighbourEndPos = hit.Select(i => rows[patIdx[i]].End).ToArray(), NeighbourPattern = hit,
            NeighbourDist = dist, NeighbourWeight = w.Select(x => x / wsum).ToArray(), NeighbourOutcome = y,
            PatternEndPos = patIdx.Select(i => rows[i].End).ToArray(),
            PatternCoords = cand.Select(c => new[] { c[0], c.Length > 1 ? c[1] : 0.0 }).ToArray(),
            PatternOutcome = patIdx.Select(i => (cleaned.Values[rows[i].End + 1] - rows[i].Mean) / rows[i].Sd).ToArray(),
            NewestCoords = new[] { newestRot[0], newestRot.Length > 1 ? newestRot[1] : 0.0 },
        };
    }

    /// <summary>
    /// Forecast `steps` weeks ahead (2 by default for "two weeks at a time"). After each forecast, the forecast is
    /// added to the history as if it had happened - it becomes the END POINT of the newest pattern - and the whole
    /// engine (including the window size) is run again.
    /// </summary>
    public List<ForecastResult> ForecastAhead(DateTime[] dates, double[] values, int steps = 2)
    {
        var results = new List<ForecastResult>(steps);
        var d = dates.ToList();
        var v = values.ToList();
        for (int s = 0; s < steps; s++)
        {
            var res = ForecastNext(d.ToArray(), v.ToArray());
            results.Add(res);
            d.Add(res.LastDate.AddDays(7));          // the extra point, one week after the last real/cleaned point
            v.Add(res.Forecast);
        }
        return results;
    }

    public static double Median(double[] x)
    {
        var s = x.OrderBy(t => t).ToArray();
        int m = s.Length;
        return m % 2 == 1 ? s[m / 2] : 0.5 * (s[m / 2 - 1] + s[m / 2]);
    }
}

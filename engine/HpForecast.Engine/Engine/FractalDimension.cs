namespace HpForecast.Engine;

public sealed record FdResult(
    double[] Lags, double[] Fd, int ElbowLag, int MaxLag, double FdElbow, double FdMax, double FdMid,
    int Window, bool UsedFallback);

/// <summary>
/// Step 1 - fractal dimension vs lag, and the window size N. This is the user's Python script
/// (calculate_fractal_dimension_vs_lag in "python fd vs lag") written in C#, so the whole engine is C#.
/// tests/reference/fd_reference.py keeps the Python original and the tests check the two agree.
///
/// For each lag L = 1..30: embed the series (each point = L consecutive values, delay 1, never across a gap);
/// use 10 radii between 0.1 and 2.0 x the series' standard deviation (so it is unit-free); count how many
/// points sit within each radius of a typical point (the average over all points, each counting itself);
/// the slope of log(count) against log(radius) is the fractal dimension at that lag.
///
/// Window size: find the elbow (the sharpest bend, the most negative second difference), take the midpoint
/// between the FD at the elbow and the maximum FD, and take the lag where the curve first crosses that
/// midpoint. N is clamped to 2..30. It is recalculated for every forecast.
/// </summary>
public static class FractalDimension
{
    public static double[][] EmbedRuns(double[] values, List<(int Start, int End)> runs, int lag)
    {
        var pts = new List<double[]>();
        foreach (var (s, e) in runs)
        {
            for (int i = s; i + lag < e; i++)        // len - lag points per run, exactly as the Python script's range(len(data) - lag * tau)
            {
                var p = new double[lag];
                Array.Copy(values, i, p, 0, lag);
                pts.Add(p);
            }
        }
        return pts.ToArray();
    }

    public static double FdForLag(double[][] pts, double[] radii)
    {
        int n = pts.Length;
        if (n < 2) return double.NaN;
        int lag = pts[0].Length;
        var d = new double[n * (n - 1) / 2];
        int idx = 0;
        for (int i = 0; i < n - 1; i++)
        {
            var pi = pts[i];
            for (int j = i + 1; j < n; j++)
            {
                var pj = pts[j];
                double s = 0;
                for (int k = 0; k < lag; k++) { double t = pi[k] - pj[k]; s += t * t; }
                d[idx++] = Math.Sqrt(s);
            }
        }
        Array.Sort(d);

        var x = new double[radii.Length];
        var y = new double[radii.Length];
        for (int r = 0; r < radii.Length; r++)
        {
            int within = UpperBound(d, radii[r]);                 // pairs at or inside this radius
            double count = 1.0 + 2.0 * within / n;                // +1 = the point itself; each pair counts from both ends
            x[r] = Math.Log(radii[r]);
            y[r] = Math.Log(count);
        }
        return Slope(x, y);
    }

    private static int UpperBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    private static double Slope(double[] x, double[] y)
    {
        double mx = x.Average(), my = y.Average(), sxy = 0, sxx = 0;
        for (int i = 0; i < x.Length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
        return sxx == 0 ? double.NaN : sxy / sxx;
    }

    public static FdResult Compute(double[] values, List<(int Start, int End)> runs, EngineConfig cfg)
    {
        var lags = Enumerable.Range(1, cfg.MaxLag).Select(l => (double)l).ToArray();
        double mean = values.Average();
        double sd = Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / values.Length);     // population sd
        if (sd <= 1e-12) return Fallback(lags, Enumerable.Repeat(double.NaN, lags.Length).ToArray(), cfg);

        var radii = new double[cfg.NRadii];
        for (int i = 0; i < cfg.NRadii; i++)
            radii[i] = (cfg.RadiusLo + (cfg.RadiusHi - cfg.RadiusLo) * i / (cfg.NRadii - 1)) * sd;

        var fd = new double[cfg.MaxLag];
        for (int l = 1; l <= cfg.MaxLag; l++) fd[l - 1] = FdForLag(EmbedRuns(values, runs, l), radii);
        return WindowFromCurve(lags, fd, cfg);
    }

    /// <summary>numpy-style gradient with unit spacing.</summary>
    public static double[] Gradient(double[] y)
    {
        int n = y.Length;
        var g = new double[n];
        g[0] = y[1] - y[0];
        g[n - 1] = y[n - 1] - y[n - 2];
        for (int i = 1; i < n - 1; i++) g[i] = (y[i + 1] - y[i - 1]) / 2.0;
        return g;
    }

    public static FdResult WindowFromCurve(double[] lags, double[] fd, EngineConfig cfg)
    {
        var ok = Enumerable.Range(0, fd.Length).Where(i => double.IsFinite(fd[i])).ToArray();
        if (ok.Length < 3) return Fallback(lags, fd, cfg);

        // Bridge any isolated NaNs for the maths (straight line between neighbours, ends held flat)
        var f = new double[fd.Length];
        for (int i = 0; i < fd.Length; i++)
        {
            if (double.IsFinite(fd[i])) { f[i] = fd[i]; continue; }
            int left = ok.Where(o => o < i).DefaultIfEmpty(-1).Max();
            int right = ok.Where(o => o > i).DefaultIfEmpty(-1).Min();
            if (left < 0) f[i] = fd[right];
            else if (right < 0) f[i] = fd[left];
            else f[i] = fd[left] + (fd[right] - fd[left]) * (lags[i] - lags[left]) / (lags[right] - lags[left]);
        }

        var second = Gradient(Gradient(f));
        int elbowI = 0;
        for (int i = 1; i < second.Length; i++) if (second[i] < second[elbowI]) elbowI = i;     // most negative bend
        double fdElbow = f[elbowI];
        int maxI = 0;
        for (int i = 1; i < f.Length; i++) if (f[i] > f[maxI]) maxI = i;
        double fdMax = f[maxI];
        double fdMid = 0.5 * (fdElbow + fdMax);

        int? crossLag = null;
        for (int i = 0; i < lags.Length - 1; i++)
        {
            double y1 = f[i], y2 = f[i + 1];
            if ((y1 <= fdMid && fdMid <= y2) || (y2 <= fdMid && fdMid <= y1))
            {
                // exactly the Python script: line through the two points, solve for the midpoint, round() = half to even
                double slope = (y2 - y1) / (lags[i + 1] - lags[i]);
                if (slope == 0) { crossLag = (int)lags[i]; break; }
                double intercept = y1 - slope * lags[i];
                double xStar = (fdMid - intercept) / slope;
                crossLag = (int)Math.Round(xStar, MidpointRounding.ToEven);
                break;
            }
        }
        if (crossLag is null)
        {
            int best = 0;
            for (int i = 1; i < f.Length; i++) if (Math.Abs(f[i] - fdMid) < Math.Abs(f[best] - fdMid)) best = i;
            crossLag = (int)lags[best];
        }
        int window = Math.Clamp(crossLag.Value, cfg.MinWindow, cfg.MaxWindow);
        return new FdResult(lags, fd, (int)lags[elbowI], (int)lags[maxI], fdElbow, fdMax, fdMid, window, false);
    }

    public static FdResult Fallback(double[] lags, double[] fd, EngineConfig cfg)
    {
        int n = Math.Clamp(cfg.FallbackWindow, cfg.MinWindow, cfg.MaxWindow);
        return new FdResult(lags, fd, n, n, double.NaN, double.NaN, double.NaN, n, true);
    }

    public static FdResult Fixed(EngineConfig cfg)
    {
        int n = Math.Clamp(cfg.FixedWindow!.Value, cfg.MinWindow, cfg.MaxWindow);
        var lags = Enumerable.Range(1, cfg.MaxLag).Select(l => (double)l).ToArray();
        return new FdResult(lags, Enumerable.Repeat(double.NaN, lags.Length).ToArray(), n, n, double.NaN, double.NaN, double.NaN, n, false);
    }
}

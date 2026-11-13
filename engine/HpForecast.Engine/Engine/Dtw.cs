namespace HpForecast.Engine;

/// <summary>
/// Step 4 helpers, ported line for line from Vector's NearestNeighborFinder:
///   DtwDistance()                dynamic time warping, Sakoe-Chiba band = ceil(70% of the vector length), cost |a - b|
///   ComputeEnvelope()            min / max of the target vector within ceil(15% of its length)
///   Calculate_LB_Keogh_Squared() quick bound used to skip candidates once k neighbours are held
///   FindKNearestNeighbours()     linear scan, oldest first, keep the k best, final order = distance then NEWEST first
///
/// QUIRKS KEPT ON PURPOSE so the same vectors give the same neighbours as the Vector code:
///  * The DTW total is a sum of absolute differences, but Vector calls it "squared" and returns its square root.
///  * The LB_Keogh bound is a sum of squares compared with that absolute-difference total. With a 15% envelope it is
///    not a guaranteed lower bound, so it can occasionally skip a candidate that full DTW would have kept.
///  * The two DTW rows are swapped without being cleared, so a cell just outside the band can still hold a value from
///    two rows earlier. (Only matters when the window is long enough for the band to bite, about 10 weeks or more.)
/// </summary>
public static class Dtw
{
    public static int BandWindow(int n, double bandFrac) => (int)Math.Ceiling(n * bandFrac);
    public static int EnvelopeRadius(int n, double envelopeFrac) => (int)Math.Ceiling(n * envelopeFrac);

    /// <summary>Vector's DtwDistance: the cumulative |a-b| cost along the best banded path (NOT yet square-rooted).</summary>
    public static double DtwTotal(double[] seriesA, double[] seriesB, double bandFrac = 0.70)
    {
        int n = seriesA.Length, m = seriesB.Length;
        if (n == 0 || m == 0) return double.PositiveInfinity;

        int window = BandWindow(n, bandFrac);
        var prevRow = new double[m + 1];
        var currentRow = new double[m + 1];
        for (int j = 0; j <= m; j++) { prevRow[j] = double.PositiveInfinity; currentRow[j] = double.PositiveInfinity; }
        prevRow[0] = 0;

        for (int i = 1; i <= n; i++)
        {
            currentRow[0] = double.PositiveInfinity;
            int start = Math.Max(1, i - window);
            int end = Math.Min(m, i + window);
            for (int j = start; j <= end; j++)
            {
                double cost = Math.Abs(seriesA[i - 1] - seriesB[j - 1]);
                double minPrevCost = Math.Min(prevRow[j], Math.Min(prevRow[j - 1], currentRow[j - 1]));
                currentRow[j] = cost + minPrevCost;
            }
            (prevRow, currentRow) = (currentRow, prevRow);
        }
        return prevRow[m];
    }

    /// <summary>The distance Vector hands on to the weights: the square root of the total, rounded to 15 places.</summary>
    public static double Distance(double[] a, double[] b, double bandFrac = 0.70) =>
        Math.Round(Math.Sqrt(DtwTotal(a, b, bandFrac)), 15);

    /// <summary>Vector's ComputeEnvelope: (lower, upper) of the series within +/- radius positions.</summary>
    public static (double[] Lower, double[] Upper) Envelope(double[] series, int radius)
    {
        int n = series.Length;
        var lower = new double[n];
        var upper = new double[n];
        for (int i = 0; i < n; i++)
        {
            int a = Math.Max(0, i - radius), b = Math.Min(n - 1, i + radius);
            double mn = series[a], mx = series[a];
            for (int j = a + 1; j <= b; j++)
            {
                if (series[j] < mn) mn = series[j];
                if (series[j] > mx) mx = series[j];
            }
            lower[i] = mn; upper[i] = mx;
        }
        return (lower, upper);
    }

    /// <summary>Vector's Calculate_LB_Keogh_Squared.</summary>
    public static double LbKeoghSquared(double[] candidate, double[] upper, double[] lower)
    {
        double sum = 0;
        for (int i = 0; i < candidate.Length; i++)
        {
            double p = candidate[i];
            if (p > upper[i]) { double d = p - upper[i]; sum += d * d; }
            else if (p < lower[i]) { double d = p - lower[i]; sum += d * d; }
        }
        return sum;
    }

    /// <summary>
    /// Vector's FindKNearestNeighbours. Candidates are scanned in the order given (oldest first). Returns the
    /// positions in <paramref name="candidates"/> and their distances, nearest first (ties: the later candidate first),
    /// plus how many candidates the lower bound skipped.
    /// </summary>
    public static (int[] Index, double[] Distance, int Skipped) KnnDtw(
        double[] target, IReadOnlyList<double[]> candidates, int k,
        double bandFrac = 0.70, double envelopeFrac = 0.15, bool useLowerBound = true)
    {
        if (candidates.Count == 0 || k <= 0) return (Array.Empty<int>(), Array.Empty<double>(), 0);

        var (lower, upper) = Envelope(target, EnvelopeRadius(target.Length, envelopeFrac));
        var top = new List<(int index, double distance)>(k);
        double worst = double.PositiveInfinity;
        int skipped = 0;

        for (int idx = 0; idx < candidates.Count; idx++)
        {
            var vec = candidates[idx];
            if (useLowerBound && top.Count == k && LbKeoghSquared(vec, upper, lower) >= worst) { skipped++; continue; }

            double d = DtwTotal(target, vec, bandFrac);
            if (top.Count < k)
            {
                top.Add((idx, d));
                top = top.OrderBy(t => t.distance).ToList();
                worst = top[^1].distance;
            }
            else if (d < worst)
            {
                top.RemoveAt(k - 1);
                top.Add((idx, d));
                top = top.OrderBy(t => t.distance).ToList();
                worst = top[^1].distance;
            }
        }

        var final = top.OrderBy(t => t.distance).ThenByDescending(t => t.index).ToArray();
        return (final.Select(t => t.index).ToArray(),
                final.Select(t => Math.Round(Math.Sqrt(t.distance), 15)).ToArray(), skipped);
    }

    /// <summary>Plain straight-line (Euclidean) nearest neighbours - an ablation, not part of Vector's first prediction.</summary>
    public static (int[] Index, double[] Distance, int Skipped) KnnEuclidean(double[] query, IReadOnlyList<double[]> candidates, int k)
    {
        k = Math.Min(k, candidates.Count);
        var all = new (double D, int I)[candidates.Count];
        for (int i = 0; i < candidates.Count; i++)
        {
            double s = 0;
            for (int j = 0; j < query.Length; j++) { double d = query[j] - candidates[i][j]; s += d * d; }
            all[i] = (Math.Sqrt(s), i);
        }
        var top = all.OrderBy(t => t.D).ThenByDescending(t => t.I).Take(k).ToArray();
        return (top.Select(t => t.I).ToArray(), top.Select(t => t.D).ToArray(), 0);
    }
}

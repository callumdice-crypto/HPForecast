namespace HpForecast.Engine;

/// <summary>Raised when the data breaks a rule that makes forecasting unsafe.</summary>
public sealed class DataRefusedException : Exception
{
    public DataRefusedException(string message) : base(message) { }
}

/// <summary>The cleaned history: kept points, which unbroken run each belongs to, and what was filled.</summary>
public sealed class CleanedSeries
{
    public double[] Values { get; init; } = Array.Empty<double>();
    public DateTime[] Dates { get; init; } = Array.Empty<DateTime>();
    public int[] RunId { get; init; } = Array.Empty<int>();
    public bool[] Filled { get; init; } = Array.Empty<bool>();
    public int NFilled { get; init; }
    public int NGapWeeks { get; init; }
    public int NInvalid { get; init; }

    /// <summary>(start, end) of each unbroken run; end is exclusive.</summary>
    public List<(int Start, int End)> Runs()
    {
        var runs = new List<(int, int)>();
        int start = 0;
        for (int i = 1; i <= Values.Length; i++)
        {
            if (i == Values.Length || RunId[i] != RunId[start])
            {
                runs.Add((start, i));
                start = i;
            }
        }
        return runs;
    }

    public int LatestRunLength { get { var r = Runs(); var last = r[^1]; return last.End - last.Start; } }
}

/// <summary>
/// Step 0 - Cleaning (rules from the engine spec, adapted from stock candles to weekly counts).
///  * Values must be valid: finite and not negative (a count of zero is fine). Invalid values become holes.
///  * Times must be strictly increasing and sit on a regular weekly grid, or the data is REFUSED.
///  * Holes up to MaxFillGap weeks are filled by a straight line. Longer holes stay as gaps and split the
///    history into unbroken runs. No pattern and no curve is ever allowed to cross a gap.
///  * At least MinPoints unbroken points are needed at the end of the history.
///  * Only the latest MaxPoints points are used.
///  * No differencing, no logs, no smoothing. Scaling happens later, window by window.
/// </summary>
public static class DataCleaner
{
    public static CleanedSeries Clean(DateTime[] dates, double[] values, EngineConfig cfg)
    {
        if (dates.Length != values.Length) throw new DataRefusedException("Dates and values must be the same length.");
        if (dates.Length == 0) throw new DataRefusedException("The series is empty.");
        for (int i = 1; i < dates.Length; i++)
        {
            if (dates[i] <= dates[i - 1])
                throw new DataRefusedException("Times must be strictly increasing (no duplicates, no going backwards).");
            if ((dates[i] - dates[i - 1]).TotalDays % 7 != 0)
                throw new DataRefusedException("Times must sit on a regular weekly grid.");
        }

        // Put on the full weekly grid so missing weeks show up as NaN holes
        int total = (dates[^1] - dates[0]).Days / 7 + 1;
        var grid = new double[total];
        Array.Fill(grid, double.NaN);
        for (int i = 0; i < dates.Length; i++) grid[(dates[i] - dates[0]).Days / 7] = values[i];

        int nInvalid = 0;
        for (int i = 0; i < total; i++)
        {
            double v = grid[i];
            if (double.IsInfinity(v) || (!double.IsNaN(v) && v < 0)) { grid[i] = double.NaN; nInvalid++; }
        }

        // Drop leading / trailing holes
        int first = Array.FindIndex(grid, v => !double.IsNaN(v));
        int last = Array.FindLastIndex(grid, v => !double.IsNaN(v));
        if (first < 0) throw new DataRefusedException("There are no valid values.");
        var vals = grid[first..(last + 1)];
        var gridDates = Enumerable.Range(first, last - first + 1).Select(i => dates[0].AddDays(7 * i)).ToArray();

        // Fill short holes, keep long ones as gaps
        int n = vals.Length;
        var filled = new bool[n];
        var keep = new bool[n];
        Array.Fill(keep, true);
        for (int i = 0; i < n;)
        {
            if (!double.IsNaN(vals[i])) { i++; continue; }
            int j = i;
            while (j < n && double.IsNaN(vals[j])) j++;
            int hole = j - i;
            if (hole <= cfg.MaxFillGap)
            {
                double left = vals[i - 1], right = vals[j];
                for (int h = 0; h < hole; h++)
                {
                    vals[i + h] = left + (right - left) * (h + 1) / (hole + 1);
                    filled[i + h] = true;
                }
            }
            else
            {
                for (int h = i; h < j; h++) keep[h] = false;
            }
            i = j;
        }

        // Unbroken runs: consecutive kept weeks share a run id
        var keptPos = Enumerable.Range(0, n).Where(i => keep[i]).ToArray();
        var runId = new int[keptPos.Length];
        for (int i = 1; i < keptPos.Length; i++)
            runId[i] = runId[i - 1] + (keptPos[i] - keptPos[i - 1] > 1 ? 1 : 0);
        var outVals = keptPos.Select(i => vals[i]).ToArray();
        var outDates = keptPos.Select(i => gridDates[i]).ToArray();
        var outFilled = keptPos.Select(i => filled[i]).ToArray();
        int gapWeeks = n - keptPos.Length;

        // Latest MaxPoints only
        if (outVals.Length > cfg.MaxPoints)
        {
            int cut = outVals.Length - cfg.MaxPoints;
            outVals = outVals[cut..]; outDates = outDates[cut..]; outFilled = outFilled[cut..];
            runId = runId[cut..];
            int baseId = runId[0];
            for (int i = 0; i < runId.Length; i++) runId[i] -= baseId;
        }

        var cleaned = new CleanedSeries
        {
            Values = outVals, Dates = outDates, RunId = runId, Filled = outFilled,
            NFilled = outFilled.Count(f => f), NGapWeeks = gapWeeks, NInvalid = nInvalid,
        };
        if (cleaned.LatestRunLength < cfg.MinPoints)
            throw new DataRefusedException(
                $"Only {cleaned.LatestRunLength} unbroken points before the forecast point; at least {cfg.MinPoints} are needed.");
        return cleaned;
    }
}

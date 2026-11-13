using HpForecast.Engine;

namespace HpForecast.Cli;

/// <summary>Built-in checks. Run:  dotnet run -c Release -- selftest   (no data files needed)</summary>
public static class SelfTest
{
    private static int _pass, _fail;

    private static void Check(string name, Func<bool> test)
    {
        bool ok;
        string note = "";
        try { ok = test(); }
        catch (Exception ex) { ok = false; note = " (" + ex.GetType().Name + ": " + ex.Message + ")"; }
        Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name}{note}");
        if (ok) _pass++; else _fail++;
    }

    private static DateTime[] Weeks(int n, string start = "2020-01-06") =>
        Enumerable.Range(0, n).Select(i => DateTime.Parse(start).AddDays(7 * i)).ToArray();

    private static double[] SeasonalSeries(int n, int seed = 1, double noise = 0.05)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(i =>
            50 + 0.05 * i + 8 * Math.Sin(2 * Math.PI * i / 52.0) + 3 * Math.Sin(2 * Math.PI * i / 13.0)
            + noise * 50 * (rng.NextDouble() - 0.5) * 2).ToArray();
    }

    public static int Run()
    {
        Console.WriteLine("Linear algebra");
        Check("SVD: V is orthonormal and singular values match the matrix size", () =>
        {
            var rng = new Random(3);
            var a = new double[40, 6];
            for (int i = 0; i < 40; i++) for (int j = 0; j < 6; j++) a[i, j] = rng.NextDouble() - 0.5;
            var (s, v) = LinearAlgebra.Svd(a);
            double fro = 0; foreach (var x in a) fro += x * x;
            double sumS2 = s.Sum(x => x * x);
            bool orth = true;
            for (int p = 0; p < 6; p++) for (int q = 0; q < 6; q++)
            {
                double dot = 0; for (int i = 0; i < 6; i++) dot += v[i, p] * v[i, q];
                if (Math.Abs(dot - (p == q ? 1 : 0)) > 1e-9) orth = false;
            }
            bool desc = Enumerable.Range(1, 5).All(i => s[i] <= s[i - 1] + 1e-12);
            return orth && desc && Math.Abs(fro - sumS2) < 1e-9;
        });
        Check("SVD: known matrix has singular values 3 and 2", () =>
        {
            var (s, _) = LinearAlgebra.Svd(new double[,] { { 3, 0 }, { 0, 2 }, { 0, 0 } });
            return Math.Abs(s[0] - 3) < 1e-12 && Math.Abs(s[1] - 2) < 1e-12;
        });
        Check("SVD: each column of V has its largest entry positive (sign rule)", () =>
        {
            var rng = new Random(9);
            var a = new double[30, 5];
            for (int i = 0; i < 30; i++) for (int j = 0; j < 5; j++) a[i, j] = rng.NextDouble() - 0.3;
            var (_, v) = LinearAlgebra.Svd(a);
            for (int j = 0; j < 5; j++)
            {
                int best = 0; for (int i = 1; i < 5; i++) if (Math.Abs(v[i, j]) > Math.Abs(v[best, j])) best = i;
                if (v[best, j] <= 0) return false;
            }
            return true;
        });
        Check("Pseudo-inverse solve recovers an exact linear relationship", () =>
        {
            var rng = new Random(5);
            var truth = new[] { 1.5, -2.0, 0.5, 3.0 };
            var a = new double[25, 4]; var y = new double[25];
            for (int i = 0; i < 25; i++) { for (int j = 0; j < 4; j++) { a[i, j] = rng.NextDouble(); y[i] += a[i, j] * truth[j]; } }
            var beta = LinearAlgebra.PseudoInverseSolve(a, y);
            return beta.Zip(truth, (b, t) => Math.Abs(b - t)).Max() < 1e-9;
        });
        Check("Pseudo-inverse solve copes with a repeated column (no blow-up)", () =>
        {
            var rng = new Random(6);
            var a = new double[20, 3]; var y = new double[20];
            for (int i = 0; i < 20; i++) { a[i, 0] = rng.NextDouble(); a[i, 1] = a[i, 0]; a[i, 2] = rng.NextDouble(); y[i] = 2 * a[i, 0] + a[i, 2]; }
            var beta = LinearAlgebra.PseudoInverseSolve(a, y);
            double err = 0; for (int i = 0; i < 20; i++) err = Math.Max(err, Math.Abs(a[i, 0] * beta[0] + a[i, 1] * beta[1] + a[i, 2] * beta[2] - y[i]));
            return err < 1e-9 && beta.All(double.IsFinite);
        });

        Console.WriteLine("DTW and neighbours (ported from Vector's NearestNeighborFinder)");
        Check("DTW of a series with itself is 0, and it is symmetric", () =>
        {
            var a = new[] { 1.0, 2, 3, 2, 1 }; var b = new[] { 1.0, 1, 2, 3, 2 };
            return Dtw.DtwTotal(a, a) == 0 && Math.Abs(Dtw.DtwTotal(a, b) - Dtw.DtwTotal(b, a)) < 1e-12;
        });
        Check("DTW can slide in time (distance 0) where the straight-line distance cannot", () =>
        {
            var a = new[] { 0.0, 0, 1 }; var b = new[] { 0.0, 1, 1 };
            return Dtw.DtwTotal(a, b) == 0 && Math.Abs(a[1] - b[1]) > 0.5;
        });
        Check("Hand-worked DTW: [1,2,3] against [2,2,4] costs 2 (absolute differences), distance sqrt(2)", () =>
        {
            var a = new[] { 1.0, 2, 3 }; var b = new[] { 2.0, 2, 4 };
            return Math.Abs(Dtw.DtwTotal(a, b) - 2.0) < 1e-12 && Math.Abs(Dtw.Distance(a, b) - Math.Sqrt(2.0)) < 1e-15;
        });
        Check("A band of 0 makes DTW the plain sum of absolute differences", () =>
        {
            var a = new[] { 1.0, 4, 2, 8 }; var b = new[] { 2.0, 3, 5, 7 };
            double l1 = a.Zip(b, (x, y) => Math.Abs(x - y)).Sum();
            return Math.Abs(Dtw.DtwTotal(a, b, 0.0) - l1) < 1e-12;
        });
        Check("Band and envelope sizes follow Vector: ceil(70%) and ceil(15%) of the vector length", () =>
            Dtw.BandWindow(6, 0.70) == 5 && Dtw.BandWindow(10, 0.70) == 7 && Dtw.BandWindow(2, 0.70) == 2
            && Dtw.EnvelopeRadius(10, 0.15) == 2 && Dtw.EnvelopeRadius(6, 0.15) == 1 && Dtw.EnvelopeRadius(2, 0.15) == 1);
        Check("Envelope and LB_Keogh: hand-worked example (bound = 4)", () =>
        {
            var (lo, up) = Dtw.Envelope(new[] { 1.0, 5, 2, 8 }, 1);
            double lb = Dtw.LbKeoghSquared(new[] { 6.0, 0, 9, 1 }, up, lo);
            return lo.SequenceEqual(new[] { 1.0, 1, 2, 2 }) && up.SequenceEqual(new[] { 5.0, 5, 8, 8 }) && Math.Abs(lb - 4.0) < 1e-12;
        });
        Check("Neighbours: nearest first; equal distances put the LATER candidate first", () =>
        {
            var c = new[] { new[] { 1.0, 2, 3 }, new[] { 1.0, 2, 3 }, new[] { 9.0, 9, 9 } };
            var (idx, d, _) = Dtw.KnnDtw(new[] { 1.0, 2, 3 }, c, 3);
            return idx.SequenceEqual(new[] { 1, 0, 2 }) && d[0] == 0 && d[1] == 0 && d[2] > 0;
        });
        Check("Neighbours with the bound switched off match a brute-force sort of every candidate", () =>
        {
            var rng = new Random(12);
            var cands = Enumerable.Range(0, 200).Select(_ => Enumerable.Range(0, 8).Select(__ => rng.NextDouble() * 2 - 1).ToArray()).ToArray();
            var q = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
            var (idx, dist, _) = Dtw.KnnDtw(q, cands, 10, 0.70, 0.15, false);
            var brute = Enumerable.Range(0, cands.Length).Select(i => (i, d: Dtw.DtwTotal(q, cands[i])))
                                  .OrderBy(t => t.d).ThenByDescending(t => t.i).Take(10).ToArray();
            return idx.SequenceEqual(brute.Select(t => t.i)) && dist.Zip(brute, (x, t) => Math.Abs(x - Math.Round(Math.Sqrt(t.d), 15))).Max() < 1e-15;
        });
        Check("The lower bound skips far-away candidates only once k neighbours are held", () =>
        {
            var rng = new Random(13);
            var q = new[] { 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5 };
            var near = Enumerable.Range(0, 10).Select(_ => q.Select(x => x + 0.05 * (rng.NextDouble() - 0.5)).ToArray());
            var far = Enumerable.Range(0, 100).Select(_ => q.Select(x => x + 10 + rng.NextDouble()).ToArray());
            var cands = near.Concat(far).ToArray();
            var with = Dtw.KnnDtw(q, cands, 10, 0.70, 0.15, true);
            var without = Dtw.KnnDtw(q, cands, 10, 0.70, 0.15, false);
            var few = Dtw.KnnDtw(q, cands.Take(10).ToArray(), 10, 0.70, 0.15, true);       // only k candidates: nothing can be skipped
            return with.Skipped == 100 && without.Skipped == 0 && few.Skipped == 0 && with.Index.OrderBy(i => i).SequenceEqual(Enumerable.Range(0, 10));
        });

        Console.WriteLine("Cleaning rules");
        Check("Refuses times that go backwards", () => Throws(() =>
            DataCleaner.Clean(new[] { DateTime.Parse("2020-01-13"), DateTime.Parse("2020-01-06") }, new[] { 1.0, 2.0 }, new EngineConfig())));
        Check("Refuses duplicate times", () => Throws(() =>
            DataCleaner.Clean(new[] { DateTime.Parse("2020-01-06"), DateTime.Parse("2020-01-06") }, new[] { 1.0, 2.0 }, new EngineConfig())));
        Check("Refuses times that are not on a weekly grid", () => Throws(() =>
            DataCleaner.Clean(new[] { DateTime.Parse("2020-01-06"), DateTime.Parse("2020-01-10") }, new[] { 1.0, 2.0 }, new EngineConfig())));
        Check("Refuses fewer than 120 unbroken points", () => Throws(() =>
            DataCleaner.Clean(Weeks(100), Enumerable.Repeat(5.0, 100).ToArray(), new EngineConfig())));
        Check("A short hole (2 weeks) is filled by a straight line", () =>
        {
            var v = Enumerable.Range(0, 130).Select(i => (double)i).ToArray(); v[50] = double.NaN; v[51] = double.NaN;
            var c = DataCleaner.Clean(Weeks(130), v, new EngineConfig());
            return c.NFilled == 2 && Math.Abs(c.Values[50] - 50) < 1e-9 && Math.Abs(c.Values[51] - 51) < 1e-9 && c.Runs().Count == 1;
        });
        Check("A long hole (4 weeks) is NOT filled and splits the history into two runs", () =>
        {
            var v = Enumerable.Range(0, 200).Select(i => 10.0 + i % 7).ToArray();
            for (int i = 40; i < 44; i++) v[i] = double.NaN;
            var c = DataCleaner.Clean(Weeks(200), v, new EngineConfig());
            var runs = c.Runs();
            return runs.Count == 2 && c.NGapWeeks == 4 && c.Values.Length == 196 && runs[0].End == 40;
        });
        Check("Negative values are treated as holes (zero is fine)", () =>
        {
            var v = Enumerable.Repeat(3.0, 130).ToArray(); v[70] = -4; v[90] = 0;
            var c = DataCleaner.Clean(Weeks(130), v, new EngineConfig());
            return c.NInvalid == 1 && c.NFilled == 1 && c.Values[90] == 0;
        });
        Check("Only the latest MaxPoints points are kept", () =>
        {
            var c = DataCleaner.Clean(Weeks(300), SeasonalSeries(300), new EngineConfig { MaxPoints = 150 });
            return c.Values.Length == 150 && c.Dates[^1] == Weeks(300)[^1];
        });

        Console.WriteLine("State space");
        Check("z-score: a flat window becomes zeros with sd stored as 1", () =>
        {
            var (z, m, sd) = StateSpaceForecaster.ZScore(new[] { 4.0, 4, 4, 4 });
            return z.All(x => x == 0) && m == 4 && sd == 1.0;
        });
        Check("z-score: result has mean 0 and (population) sd 1", () =>
        {
            var (z, _, _) = StateSpaceForecaster.ZScore(new[] { 1.0, 5, 2, 8, 3 });
            double m = z.Average(); double sd = Math.Sqrt(z.Sum(x => (x - m) * (x - m)) / z.Length);
            return Math.Abs(m) < 1e-12 && Math.Abs(sd - 1) < 1e-12;
        });
        Check("Cumulative embedding as in Vector: Low = running minimum, High = running maximum, then z-scored", () =>
        {
            var (zl, ml, sl) = StateSpaceForecaster.MakeRow(new[] { 5.0, 3, 4, 1 }, CumulativeKind.Low);      // -> 5,3,3,1
            var (zh, mh, _) = StateSpaceForecaster.MakeRow(new[] { 1.0, 3, 2, 5 }, CumulativeKind.High);      // -> 1,3,3,5
            return Math.Abs(ml - 3.0) < 1e-12 && Math.Abs(sl - Math.Sqrt(2.0)) < 1e-12 && Math.Abs(zl[0] - Math.Sqrt(2.0)) < 1e-12
                && zl[1] == 0 && Math.Abs(mh - 3.0) < 1e-12 && Math.Abs(zh[3] - Math.Sqrt(2.0)) < 1e-12;
        });
        Check("Patterns never cross a gap, and only the newest pattern has no outcome", () =>
        {
            var v = Enumerable.Range(0, 260).Select(i => 10.0 + Math.Sin(i)).ToArray();
            for (int i = 80; i < 85; i++) v[i] = double.NaN;
            var c = DataCleaner.Clean(Weeks(260), v, new EngineConfig());
            var runs = c.Runs();
            var rows = StateSpaceForecaster.BuildStateSpace(c.Values, runs, 6);
            bool inOneRun = rows.All(r => runs.Any(run => r.End - 5 >= run.Start && r.End < run.End));
            int expected = runs.Sum(r => Math.Max(0, (r.End - r.Start) - 6 + 1));
            return inOneRun && rows.Count == expected && rows.Count(r => !r.HasNext) == runs.Count && !rows[^1].HasNext;
        });

        Console.WriteLine("Engine");
        var dates = Weeks(300); var vals = SeasonalSeries(300);
        Check("Forecast is finite and close to the truth on a clean seasonal series", () =>
        {
            var f = new StateSpaceForecaster().ForecastNext(dates[..250], vals[..250]);
            double truth = 50 + 0.05 * 250 + 8 * Math.Sin(2 * Math.PI * 250 / 52.0) + 3 * Math.Sin(2 * Math.PI * 250 / 13.0);
            return double.IsFinite(f.Forecast) && Math.Abs(f.Forecast - truth) / truth < 0.20 && f.Window is >= 2 and <= 30;
        });
        Check("Same input gives the same forecast (deterministic)", () =>
        {
            var a = new StateSpaceForecaster().ForecastNext(dates[..250], vals[..250]);
            var b = new StateSpaceForecaster().ForecastNext(dates[..250], vals[..250]);
            return a.Forecast == b.Forecast && a.Window == b.Window;
        });
        Check("No look-ahead: changing values AFTER the forecast point changes nothing", () =>
        {
            var changed = (double[])vals.Clone();
            for (int i = 250; i < 300; i++) changed[i] = 9999;
            var a = new StateSpaceForecaster().ForecastNext(dates[..250], vals[..250]);
            var b = new StateSpaceForecaster().ForecastNext(dates[..250], changed[..250]);
            return a.Forecast == b.Forecast;
        });
        Check("Euclidean neighbours: rotating by the SVD leaves the next-value forecast unchanged (rotation preserves distances)", () =>
        {
            var on = new StateSpaceForecaster(new EngineConfig { Distance = DistanceKind.Euclidean, Rotate = true, Target = PredictionTarget.NextValueZ }).ForecastNext(dates[..250], vals[..250]);
            var off = new StateSpaceForecaster(new EngineConfig { Distance = DistanceKind.Euclidean, Rotate = false, Target = PredictionTarget.NextValueZ }).ForecastNext(dates[..250], vals[..250]);
            return Math.Abs(on.Forecast - off.Forecast) < 1e-6;
        });
        Check("Flat history does not crash (window of zeros, sd treated as 1)", () =>
        {
            var f = new StateSpaceForecaster().ForecastNext(Weeks(150), Enumerable.Repeat(7.0, 150).ToArray());
            return Math.Abs(f.Forecast - 7.0) < 1e-6;
        });
        Check("Two weeks at a time: week 2 uses the week-1 forecast as the new end point", () =>
        {
            var r = new StateSpaceForecaster().ForecastAhead(dates[..250], vals[..250], 2);
            return r.Count == 2
                && r[1].Cleaned.Values.Length == r[0].Cleaned.Values.Length + 1
                && r[1].Cleaned.Values[^1] == r[0].Forecast
                && r[1].LastDate == r[0].LastDate.AddDays(7);
        });
        Check("The engine refuses data with too few points", () => Throws(() =>
            new StateSpaceForecaster().ForecastNext(Weeks(80), SeasonalSeries(80))));

        Check("k follows Vector: (int) max(N, 2.5% of all rows), never more than the patterns available", () =>
        {
            var f = new StateSpaceForecaster().ForecastNext(dates[..250], vals[..250]);
            return f.K == Math.Min((int)Math.Max(f.Window, (f.NPatterns + 1) * 0.025), f.NPatterns);
        });
        Check("A fixed window is used when one is asked for (and the fractal-dimension step is skipped)", () =>
        {
            var f = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8 }).ForecastNext(dates[..250], vals[..250]);
            return f.Window == 8 && double.IsNaN(f.Fd.FdMid);
        });
        Check("Default readout is Vector's with the rotation undone; it is not the degenerate 'always 0' readout", () =>
        {
            var back = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8 }).ForecastNext(dates[..250], vals[..250]);
            var lit = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8, Target = PredictionTarget.VectorLastCoordinate }).ForecastNext(dates[..250], vals[..250]);
            return new EngineConfig().Target == PredictionTarget.VectorRotatedBack
                && double.IsFinite(back.Forecast) && Math.Abs(back.ZPred) > 1e-6
                && Math.Abs(lit.ZPred) < 1e-9                       // Vector as written: the last rotated coordinate of mean-zero windows is 0
                && Math.Abs(lit.Forecast - lit.WindowMean) < 1e-6;  // ... so the forecast is just the window average
        });
        Check("With no rotation there is nothing to undo: rotated-back and literal readouts agree", () =>
        {
            var a = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8, Rotate = false }).ForecastNext(dates[..250], vals[..250]);
            var b = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8, Rotate = false, Target = PredictionTarget.VectorLastCoordinate }).ForecastNext(dates[..250], vals[..250]);
            return Math.Abs(a.Forecast - b.Forecast) < 1e-9;
        });
        Check("Next-value readout (crypto-edge style) runs and differs from Vector's", () =>
        {
            var a = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8 }).ForecastNext(dates[..250], vals[..250]);
            var b = new StateSpaceForecaster(new EngineConfig { FixedWindow = 8, Target = PredictionTarget.NextValueZ }).ForecastNext(dates[..250], vals[..250]);
            return double.IsFinite(b.Forecast) && Math.Abs(a.Forecast - b.Forecast) > 1e-9;
        });
        Check("Pseudo-inverse uses Math.NET's cut-off: a column below max(rows,cols) x largest x 2^-53 is ignored", () =>
        {
            // A column that is (almost) a copy of another one: a plain least-squares fit would blow up, Math.NET's rule drops it.
            var a = new double[6, 2];
            var y = new double[6];
            for (int i = 0; i < 6; i++) { a[i, 0] = i + 1.0; a[i, 1] = (i + 1.0) * 1e-18; y[i] = 2.0 * (i + 1.0); }
            var beta = LinearAlgebra.PseudoInverseSolve(a, y);
            return Math.Abs(beta[0] - 2.0) < 1e-9 && Math.Abs(beta[1]) < 1e-6;
        });
        Check("Running-minimum windows (Vector's Low branch) can be switched on", () =>
        {
            var f = new StateSpaceForecaster(new EngineConfig { FixedWindow = 6, Cumulative = CumulativeKind.Low }).ForecastNext(dates[..250], vals[..250]);
            return double.IsFinite(f.Forecast);
        });

        Console.WriteLine("Fractal dimension (the Python script, in C#)");
        Check("A straight ramp has fractal dimension close to 1 at lag 1", () =>
        {
            var ramp = Enumerable.Range(0, 500).Select(i => (double)i).ToArray();
            var fd = FractalDimension.Compute(ramp, new List<(int, int)> { (0, 500) }, new EngineConfig());
            return fd.Fd[0] > 0.85 && fd.Fd[0] < 1.05 && fd.Lags.Length == 30;
        });
        Check("The window comes from where the curve crosses the midpoint of its elbow and its peak, kept within 2..30", () =>
        {
            var fd = FractalDimension.Compute(vals[..250], new List<(int, int)> { (0, 250) }, new EngineConfig());
            return fd.Window is >= 2 and <= 30 && !fd.UsedFallback && double.IsFinite(fd.FdMid)
                && Math.Abs(fd.FdMid - 0.5 * (fd.Fd[fd.ElbowLag - 1] + fd.Fd[fd.MaxLag - 1])) < 1e-12;
        });
        Check("Curve helper: a hand-made curve gives the expected window", () =>
        {
            var lags = Enumerable.Range(1, 30).Select(l => (double)l).ToArray();
            var fd = lags.Select(l => l <= 10 ? l * 0.2 : 2.0).ToArray();      // rises 0.2 a lag to 2.0 at lag 10, then flat
            var r = FractalDimension.WindowFromCurve(lags, fd, new EngineConfig());
            return r.MaxLag == 10 && r.Window is >= 2 and <= 10;
        });

        Console.WriteLine($"\n{_pass} passed, {_fail} failed.");
        return _fail == 0 ? 0 : 1;
    }

    private static bool Throws(Action a)
    {
        try { a(); return false; } catch (DataRefusedException) { return true; }
    }
}

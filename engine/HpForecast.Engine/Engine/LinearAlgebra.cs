namespace HpForecast.Engine;

/// <summary>
/// Small linear-algebra helpers written from scratch, so the project needs no NuGet packages
/// (your original code used MathNet.Numerics for the SVD and pseudo-inverse).
/// </summary>
public static class LinearAlgebra
{
    /// <summary>
    /// Singular value decomposition by the one-sided Jacobi method, for an m x n matrix with m >= n.
    /// Returns the singular values S (largest first) and the n x n matrix V of right singular vectors.
    /// Rotating the rows of A into the SVD directions is then simply A * V (which equals U * S).
    ///
    /// SIGN RULE: a singular vector and its negative are equally valid, but DTW on rotated
    /// coordinates would give different answers for each choice. We fix it: in every column of V the
    /// entry with the largest absolute value is made positive. (The Python reference does the same.)
    /// </summary>
    public static (double[] S, double[,] V) Svd(double[,] a)
    {
        int m = a.GetLength(0), n = a.GetLength(1);
        if (m < n) throw new ArgumentException("Svd needs at least as many rows as columns.");

        var u = (double[,])a.Clone();            // its columns are rotated until they are all at right angles
        var v = new double[n, n];
        for (int i = 0; i < n; i++) v[i, i] = 1.0;

        for (int sweep = 0; sweep < 60; sweep++)
        {
            double worst = 0.0;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    double alpha = 0, beta = 0, gamma = 0;
                    for (int i = 0; i < m; i++)
                    {
                        double up = u[i, p], uq = u[i, q];
                        alpha += up * up; beta += uq * uq; gamma += up * uq;
                    }
                    double denom = Math.Sqrt(alpha * beta);
                    if (denom == 0.0 || gamma == 0.0) continue;
                    double rel = Math.Abs(gamma) / denom;
                    if (rel > worst) worst = rel;
                    if (rel < 1e-15) continue;

                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t = (zeta >= 0 ? 1.0 : -1.0) / (Math.Abs(zeta) + Math.Sqrt(1.0 + zeta * zeta));
                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = c * t;
                    for (int i = 0; i < m; i++)
                    {
                        double up = u[i, p], uq = u[i, q];
                        u[i, p] = c * up - s * uq;
                        u[i, q] = s * up + c * uq;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double vp = v[i, p], vq = v[i, q];
                        v[i, p] = c * vp - s * vq;
                        v[i, q] = s * vp + c * vq;
                    }
                }
            }
            if (worst < 1e-13) break;
        }

        var sv = new double[n];
        for (int j = 0; j < n; j++)
        {
            double ss = 0;
            for (int i = 0; i < m; i++) ss += u[i, j] * u[i, j];
            sv[j] = Math.Sqrt(ss);
        }

        // Largest singular value first
        var order = Enumerable.Range(0, n).OrderByDescending(j => sv[j]).ToArray();
        var sSorted = new double[n];
        var vSorted = new double[n, n];
        for (int newJ = 0; newJ < n; newJ++)
        {
            int oldJ = order[newJ];
            sSorted[newJ] = sv[oldJ];
            for (int i = 0; i < n; i++) vSorted[i, newJ] = v[i, oldJ];
        }

        // Sign rule (see above)
        for (int j = 0; j < n; j++)
        {
            int best = 0;
            for (int i = 1; i < n; i++)
                if (Math.Abs(vSorted[i, j]) > Math.Abs(vSorted[best, j])) best = i;
            if (vSorted[best, j] < 0)
                for (int i = 0; i < n; i++) vSorted[i, j] = -vSorted[i, j];
        }
        return (sSorted, vSorted);
    }

    /// <summary>A * V for an m x n matrix A and n x n matrix V: rotates every row into the SVD directions.</summary>
    public static double[][] RotateRows(double[][] rows, double[,] v)
    {
        int n = v.GetLength(0);
        var result = new double[rows.Length][];
        for (int r = 0; r < rows.Length; r++)
        {
            var x = rows[r];
            var y = new double[n];
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < n; i++) s += x[i] * v[i, j];
                y[j] = s;
            }
            result[r] = y;
        }
        return result;
    }

    /// <summary>
    /// Least-squares solution of A * beta = y using the pseudo-inverse (small singular values are ignored, by Math.NET's own cut-off rule).
    /// This is the "weighted straight-line map" step of the engine, after the rows were weighted.
    /// A must have at least as many rows as columns.
    /// </summary>
    public static double[] PseudoInverseSolve(double[,] a, double[] y)
    {
        int m = a.GetLength(0), n = a.GetLength(1);
        var (s, v) = Svd(a);
        // Math.NET's PseudoInverse() rule: tolerance = max(rows, cols) x largest singular value x 2^-53;
        // a singular value below it is treated as zero (its inverse is set to 0).
        double tol = Math.Max(m, n) * s[0] * Math.Pow(2, -53);

        // beta = V * diag(1/s^2) * V^T * A^T * y     (for the singular values above the tolerance)
        var aty = new double[n];
        for (int j = 0; j < n; j++)
        {
            double acc = 0;
            for (int i = 0; i < m; i++) acc += a[i, j] * y[i];
            aty[j] = acc;
        }
        var w = new double[n];
        for (int k = 0; k < n; k++)
        {
            if (s[k] < tol) { w[k] = 0; continue; }
            double acc = 0;
            for (int j = 0; j < n; j++) acc += v[j, k] * aty[j];
            w[k] = acc / (s[k] * s[k]);
        }
        var beta = new double[n];
        for (int j = 0; j < n; j++)
        {
            double acc = 0;
            for (int k = 0; k < n; k++) acc += v[j, k] * w[k];
            beta[j] = acc;
        }
        return beta;
    }
}

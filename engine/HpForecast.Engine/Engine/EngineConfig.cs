namespace HpForecast.Engine;

public enum DistanceKind { Dtw, Euclidean }

/// <summary>
/// Vector's embedding can turn each window into a running minimum ("Low") or running maximum ("High") before it is
/// z-scored (CreateCumulativeSvdEmbedding). Share prices need that; cancellation counts do not, so the default is None.
/// </summary>
public enum CumulativeKind { None, Low, High }

/// <summary>
/// What the neighbours are asked to predict (the readout of step 5).
///  VectorRotatedBack     (DEFAULT) Vector's prediction, with one fix. Vector maps the whole current state to the whole
///                        next state, in SVD space, then reads the LAST coordinate as the newest week. After the SVD
///                        rotation the last coordinate is the weakest direction (always 0 for z-scored windows), not the
///                        newest week. Here the predicted state is rotated BACK to the original week positions first, so
///                        the last number really is the newest week. (Equivalent: each neighbour's outcome is the last
///                        number of its next z-scored window.)
///  VectorLastCoordinate  Vector exactly as written: read the last coordinate in the rotated space. Always 0 for
///                        z-scored windows, so the forecast becomes the average of the window. Kept to show this.
///  NextValueZ            the next value of the series on the z-score scale of the pattern's own window (how the
///                        crypto-edge port does it).
/// </summary>
public enum PredictionTarget { VectorRotatedBack, VectorLastCoordinate, NextValueZ }

/// <summary>
/// Settings for the state-space nearest-neighbour engine. The defaults ARE the Vector code / the engine spec.
/// </summary>
public sealed record EngineConfig
{
    // Step 0 - cleaning
    public int MinPoints { get; init; } = 120;        // unbroken points needed before the forecast point
    public int MaxPoints { get; init; } = 2500;       // only the latest N points are used
    public int MaxFillGap { get; init; } = 2;         // holes up to this long are filled; longer ones stay gaps

    // Step 1 - fractal dimension vs lag (the Python script, in C#)
    public int MaxLag { get; init; } = 30;
    public int NRadii { get; init; } = 10;
    public double RadiusLo { get; init; } = 0.1;      // x the series' standard deviation
    public double RadiusHi { get; init; } = 2.0;
    public int MinWindow { get; init; } = 2;
    public int MaxWindow { get; init; } = 30;
    public int FallbackWindow { get; init; } = 8;     // only if the FD curve is unusable
    public int? FixedWindow { get; init; } = null;    // ablation: skip the FD step and force this window

    // Step 2 - embedding
    public CumulativeKind Cumulative { get; init; } = CumulativeKind.None;

    // Step 3 - SVD rotation
    public bool Rotate { get; init; } = true;

    // Step 4 - neighbours (Vector: FindKNearestNeighbours)
    public DistanceKind Distance { get; init; } = DistanceKind.Dtw;
    public double BandFrac { get; init; } = 0.70;     // DTW band = ceil(70% of the window length)
    public double EnvelopeFrac { get; init; } = 0.15; // LB_Keogh envelope radius = ceil(15% of the window length)
    public bool UseLbKeogh { get; init; } = true;
    public double KMinFrac { get; init; } = 0.025;    // k = (int) max(N, 2.5% of the number of rows)

    // Step 5 - prediction (Vector: CalculateDeltaPredictionNoSVD)
    public double SigmaFrac { get; init; } = 0.5;     // sigma = half the median neighbour distance
    public double PlausibilityMargin { get; init; } = 1.0;
    public PredictionTarget Target { get; init; } = PredictionTarget.VectorRotatedBack;
}

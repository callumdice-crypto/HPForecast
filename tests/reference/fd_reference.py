"""Reference copy of calculate_fractal_dimension_vs_lag(), taken verbatim from the original Python script
("python fd vs lag.txt").  Used ONLY by the tests, to check that the C# version (engine/HpForecast.Engine/Engine/FractalDimension.cs)
gives the same numbers.  The engine itself does not use Python for this any more."""
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree as KDTree


def calculate_fractal_dimension_vs_lag(data, max_embedding_dimension=20, tau=1, adjustment_factor=0):
    """
    Calculate fractal dimension for lags 1..max_embedding_dimension using correlation dimension approach.
    Then find:
      - opt_lag: the "elbow" or flattening (via 2nd derivative) + adjustment_factor
      - lag_at_max_fd: max FD
      - lag_at_avg_fd_intersection: lag where FD curve crosses average(FD@opt_lag, FD@max)
    Returns:
      (lags, fractal_dimensions, opt_lag, lag_at_max_fd, lag_at_avg_fd_intersection,
       fd_optlag, fd_max, fd_average)
    """
    data = np.array(data)
    lags = list(range(1, max_embedding_dimension + 1))
    fractal_dimensions = []

    for lag in lags:
        embedded = []
        for i in range(len(data) - lag * tau):
            point = [data[i + j * tau] for j in range(lag)]
            embedded.append(point)
        embedded = np.array(embedded)

        if len(embedded) < 2:
            fractal_dimensions.append(np.nan)
            continue

        tree = KDTree(embedded)
        epsilon_vals = np.linspace(0.1, 2.0, 10)
        counts = []
        for eps in epsilon_vals:
            count = tree.query_ball_point(embedded, r=eps, return_length=True).mean()
            counts.append(count)

        if any(c == 0 for c in counts):
            fractal_dimensions.append(np.nan)
            continue

        log_eps = np.log(epsilon_vals)
        log_counts = np.log(counts)
        slope, intercept, _, _, _ = stats.linregress(log_eps, log_counts)
        fractal_dimensions.append(slope)

    if len(fractal_dimensions) < 2:
        return (lags, fractal_dimensions, None, None, None, None, None, None)

    diffs = np.gradient(fractal_dimensions)
    second_diffs = np.gradient(diffs)
    epsilon_index = np.argmin(second_diffs) + adjustment_factor
    epsilon_index = max(0, min(len(fractal_dimensions) - 1, epsilon_index))

    opt_lag = lags[epsilon_index]
    fd_optlag = fractal_dimensions[epsilon_index]

    max_fd_value = np.nanmax(fractal_dimensions)
    max_fd_index = np.nanargmax(fractal_dimensions)
    lag_at_max_fd = lags[max_fd_index]

    avg_fd_value = (fd_optlag + max_fd_value) / 2

    lag_at_avg_fd_intersection = None
    for i in range(len(fractal_dimensions) - 1):
        y1 = fractal_dimensions[i]
        y2 = fractal_dimensions[i+1]
        if (y1 <= avg_fd_value <= y2) or (y2 <= avg_fd_value <= y1):
            slope = (y2 - y1) / (lags[i+1] - lags[i])
            intercept = y1 - slope * lags[i]
            x_star = (avg_fd_value - intercept) / slope
            lag_at_avg_fd_intersection = round(x_star)
            break

    if lag_at_avg_fd_intersection is None:
        arr = np.array(fractal_dimensions)
        idx_near = np.nanargmin(np.abs(arr - avg_fd_value))
        lag_at_avg_fd_intersection = lags[idx_near]

    return (
        lags,
        fractal_dimensions,
        opt_lag,
        lag_at_max_fd,
        lag_at_avg_fd_intersection,
        fd_optlag,
        max_fd_value,
        avg_fd_value
    )

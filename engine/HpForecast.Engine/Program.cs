using HpForecast.Cli;

// Cancellation forecasting - the C# prediction engine.
//   dotnet run -c Release -- selftest
//   dotnet run -c Release -- forecast --data ../../data --out ../../outputs
//   dotnet run -c Release -- backtest --data ../../data --out ../../outputs/csharp_backtest.csv
if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
{
    Console.WriteLine(@"Cancellation forecast (state-space nearest-neighbour engine)

  selftest   run the built-in checks (no data needed)
  forecast   forecast the next 2 weeks for every appointment type, and the missed income by client price
  backtest   walk-forward test over the past, writes a CSV of forecast vs actual
  fd         fractal dimension against lag (the Python script, in C#), for the charts

Common options:  --data <folder with the CSV files>   --out <file or folder>
backtest/forecast options:  --raw-counts   --horizon 2
ablation options:  --distance euclidean   --no-rotate   --no-lb   --fixed-window 13   --label ""name""
Vector options:    --predict vector (default: rotated back) | vector-literal (last rotated coordinate, always 0) | next-value   --cumulative low|high");
    return 0;
}

var options = new Options(args.Skip(1).ToArray());
try
{
    return args[0].ToLowerInvariant() switch
    {
        "selftest" => SelfTest.Run(),
        "forecast" => ForecastCommand.Run(options),
        "backtest" => BacktestCommand.Run(options),
        "fd" => FdCommand.Run(options),
        _ => Fail($"Unknown command '{args[0]}'. Run with --help."),
    };
}
catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException or InvalidDataException)
{
    return Fail(ex.Message + "\nHave you run  python -m hpforecast data  first?");
}

static int Fail(string msg) { Console.Error.WriteLine(msg); return 1; }

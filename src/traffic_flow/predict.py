"""
Forecast traffic flow from recent readings.

    python -m traffic_flow.predict --csv data/raw/test.csv
    python -m traffic_flow.predict --time "2016-03-04 08:00" --readings 98 105 110 121 118 125 130 128 135 131 140 138
    python -m traffic_flow.predict --backtest

The forecaster is fitted on first use (about a second) and cached in artifacts/.
"""
import argparse
import json

from traffic_flow.forecast import DEFAULT_PATH, Forecaster


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", help="PeMS-style CSV; its last 12 rows are the recent readings.")
    parser.add_argument("--readings", nargs="+", type=float,
                        help="The 12 most recent 5-minute readings, oldest first.")
    parser.add_argument("--time", help="Time of the last reading, e.g. '2016-03-04 08:00'.")
    parser.add_argument("--backtest", action="store_true",
                        help="Report accuracy on --test-csv instead of forecasting.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--forecaster", default=DEFAULT_PATH)
    parser.add_argument("--refit", action="store_true", help="Fit again even if a saved one exists.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args(argv)

    if args.refit:
        f = Forecaster.fit(args.train_csv)
        f.save(args.forecaster)
    else:
        f = Forecaster.load_or_fit(args.forecaster, args.train_csv)

    if args.backtest:
        table = f.backtest(args.test_csv)
        print(table.to_json(orient="records") if args.json else table.round(3).to_string(index=False))
        return 0

    try:
        if args.csv:
            result = f.predict_from_csv(args.csv)
        elif args.readings and args.time:
            result = f.predict(args.readings, args.time)
        else:
            parser.error("give --csv, or both --readings and --time, or --backtest")
    except ValueError as err:
        parser.exit(2, "error: %s\n" % err)

    if args.json:
        print(json.dumps(result))
    else:
        print("%-10s %-18s %s" % ("ahead", "time", "forecast (veh / 5 min)"))
        for r in result:
            print("%-10s %-18s %.1f" % ("%d min" % r["horizon_minutes"], r["time"], r["flow"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

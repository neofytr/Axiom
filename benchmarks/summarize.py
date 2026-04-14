"""summarize.py — merge axiom + tf bench outputs into one comparison table.

reads two files (axiom output, tf output), each containing RESULT lines:
    RESULT <suite> <case> <metric> <value>

produces a markdown table joined on (suite, case, metric), with a "ratio"
column (axiom / tf for throughput metrics, tf / axiom for latency/error),
highlights cases where Axiom loses more than 5%.

usage: python3 summarize.py axiom.txt tf.txt [--suite=gemm]
"""

import sys, re, collections, argparse


def parse_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"RESULT\s+(\S+)\s+(\S+)\s+(\S+)\s+([\deE+\-.nan]+)", line)
            if not m: continue
            suite, case, metric, value = m.group(1), m.group(2), m.group(3), m.group(4)
            try:
                v = float(value)
            except ValueError:
                continue
            results[(suite, case, metric)] = v
    return results


# metrics where larger is better (Axiom wins when ratio > 1)
HIGHER_IS_BETTER = {"gflops", "gbs", "tokens_per_sec"}
# metrics where smaller is better
LOWER_IS_BETTER = {"lat_ms", "sec", "ms_per_step", "max_err", "mean_err"}


def fmt_value(v, metric):
    if metric in ("gflops", "gbs", "tokens_per_sec"):
        return f"{v:.1f}"
    if metric == "lat_ms":
        if v < 1.0:   return f"{v:.3f}"
        if v < 100:   return f"{v:.2f}"
        return f"{v:.1f}"
    if metric == "sec":
        return f"{v:.2f}"
    if metric == "ms_per_step":
        return f"{v:.1f}"
    if "err" in metric:
        return f"{v:.2e}"
    return f"{v:.3g}"


def ratio_str(ax, tf, metric):
    if tf == 0: return "-"
    if metric in HIGHER_IS_BETTER:
        r = ax / tf
        better = r > 1.0
        pct = (r - 1.0) * 100.0
    else:
        if ax == 0: return "-"
        r = tf / ax
        better = r > 1.0
        pct = (r - 1.0) * 100.0
    mark = "**Axiom**" if better else "TF"
    return f"{mark} {pct:+.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("axiom")
    p.add_argument("tf")
    p.add_argument("--suite", default=None,
                   help="only show cases from this suite")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="percent gap below which we don't flag")
    p.add_argument("--markdown", action="store_true", default=True)
    args = p.parse_args()

    ax = parse_results(args.axiom)
    tf = parse_results(args.tf)

    all_keys = sorted(set(ax.keys()) | set(tf.keys()))

    suites = collections.defaultdict(list)
    for k in all_keys:
        suite, case, metric = k
        if args.suite and suite != args.suite: continue
        if metric in ("mean_err",): continue  # dedup with max_err
        a = ax.get(k); t = tf.get(k)
        suites[suite].append((case, metric, a, t))

    regressions = []

    for suite, rows in suites.items():
        # group rows by case so we can print one row per case
        by_case = collections.defaultdict(dict)
        for (case, metric, a, t) in rows:
            by_case[case][metric] = (a, t)

        print(f"\n## {suite}\n")
        # choose header columns based on which metrics appear
        metrics_seen = set()
        for v in by_case.values(): metrics_seen.update(v.keys())
        # preferred column order
        order = ["lat_ms", "gflops", "gbs", "tokens_per_sec", "ms_per_step",
                 "sec", "max_err"]
        cols = [m for m in order if m in metrics_seen]

        header = ["case"]
        for m in cols:
            header += [f"ax_{m}", f"tf_{m}", f"{m}_cmp"]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")

        for case in sorted(by_case.keys()):
            row = [case]
            for m in cols:
                a, t = by_case[case].get(m, (None, None))
                if a is None and t is None:
                    row += ["-", "-", "-"]
                    continue
                as_ = fmt_value(a, m) if a is not None else "-"
                ts_ = fmt_value(t, m) if t is not None else "-"
                if a is not None and t is not None:
                    cmp_ = ratio_str(a, t, m)
                    # check regression
                    if m in HIGHER_IS_BETTER and a < t * (1 - args.threshold/100):
                        regressions.append((suite, case, m, a, t))
                    elif m in LOWER_IS_BETTER and a > t * (1 + args.threshold/100) and m != "max_err":
                        regressions.append((suite, case, m, a, t))
                else:
                    cmp_ = "-"
                row += [as_, ts_, cmp_]
            print("| " + " | ".join(row) + " |")

    if regressions:
        print("\n## Regressions (>{}% gap)\n".format(args.threshold))
        print("| suite | case | metric | axiom | tf |")
        print("|---|---|---|---|---|")
        for (s, c, m, a, t) in regressions:
            print(f"| {s} | {c} | {m} | {fmt_value(a, m)} | {fmt_value(t, m)} |")
    else:
        print("\n_No regressions >{}% detected._\n".format(args.threshold))


if __name__ == "__main__":
    main()

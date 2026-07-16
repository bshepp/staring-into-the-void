"""Pretty-print a downloaded null_sweep JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    d = json.loads(Path(path).read_text())
    cfg = d["config"]
    r = d["results"]
    print(f"timestamp: {d['timestamp']}")
    print(f"config: N_NULL={cfg['n_null_realizations']} N_PER_NULL={cfg['n_sources_per_null']} "
          f"N_RR={cfg['n_rr']} seed={cfg['seed']}")
    print(f"null_mean_H1: {r['null_mean_h1']:.6f}")
    print(f"null_std_H1:  {r['null_std_h1']:.6f}")
    print(f"rr_loaded:    {r['rr_loaded']}")
    print()
    print(f"{'factor':>6} {'H1':>10} {'z':>8} {'p_value':>10} {'detected':>10}")
    mu, sd = r["null_mean_h1"], r["null_std_h1"]
    for a in r["attenuation"]:
        z = (a["total_persistence_H1"] - mu) / sd
        print(f"{a['factor']:>6.2f} {a['total_persistence_H1']:>10.3f} "
              f"{z:>8.2f} {a['p_value']:>10.4g} {str(a['detected']):>10}")


if __name__ == "__main__":
    main(sys.argv[1])

"""Proportions-only entity / pair co-occurrence summary.

Reports the *proportions* quoted in the revised Part 2 of the rebuttal
(Reviewer 2, comment 2, item 3), for two conditions:

  released  -- the prompts as released;
  noleak    -- the control condition of item 4, i.e. the two potentially
               outcome-adjacent prompt fields removed
               ("Known Drug Response Profiles" for cell lines,
                "Known Drug-Drug Interaction Patterns" for drugs).

Only plain proportions are printed here -- no enrichment ratios and no
p-values.  The alias lexicon, sentence splitting and matching logic are
imported from pair_content_audit.py, so the two scripts cannot drift apart.

Source data:
  json/drug_features_v3_clean.json            json/cell_line_features_v3_clean.json
  json/drug_features_v3_clean_noleak.json     json/cell_line_features_v3_clean_noleak.json
  data/split/all_items.npy

Run:
  /opt/miniforge3/envs/lsq/bin/python cooccurrence_proportions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pair_content_audit as pca  # noqa: E402


def _pct(num, den):
    return f"{100.0 * num / den:.2f}%" if den else "n/a"


def main():
    header = (f"{'condition':10s} | {'drug side':^58s} | {'cell side':^44s}")
    print(header)
    print(f"{'':10s} | {'1 direction':>18s} {'either direction':>18s} "
          f"{'pair-cell inst.':>19s} | {'named':>13s} {'named+response':>15s} "
          f"{'covers cell':>13s}")
    print("-" * len(header))

    for name, (df, cf) in pca.VARIANTS.items():
        if not os.path.exists(os.path.join(pca.JSON, df)):
            print(f"  skip {name}: {df} not found")
            continue
        r = pca.audit_variant(name, df, cf)
        b1, b3 = r["B1"], r["B3"]
        cov = r["B5"]["coverage"]

        one = f"{b1['pairs_named']} ({_pct(b1['pairs_named'], b1['distinct_pairs'])})"
        either = (f"{b1['pairs_named_either_dir']} "
                  f"({_pct(b1['pairs_named_either_dir'], b1['distinct_pairs'])})")
        inst = (f"{b1['instances_named_either_dir']} ({_pct(b1['instances_named_either_dir'], b1['instances'])})")
        cell_named = (f"{b3['named']['benchmark']['hits']} "
                      f"({_pct(b3['named']['benchmark']['hits'], b3['benchmark_combos'])})")
        cell_near = (f"{b3['named_near_response']['benchmark']['hits']} "
                     f"({_pct(b3['named_near_response']['benchmark']['hits'], b3['benchmark_combos'])})")
        cell_cov = f"{cov['cells']}/{cov['cells_total']} cells, {cov['drugs']} drugs"

        print(f"{name:10s} | {one:>18s} {either:>18s} {inst:>19s} | "
              f"{cell_named:>13s} {cell_near:>15s} {cell_cov:>13s}")

    print()
    print("Denominators: drug side over the distinct benchmark drug pairs "
          "(14,958) and pair-cell instances (330,765);")
    print("cell side over the benchmark (cell line, drug) combinations (11,132).")


if __name__ == "__main__":
    main()

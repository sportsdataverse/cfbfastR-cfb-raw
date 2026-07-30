"""Keep only games whose final is still NOT current.

Grepping the scrape logs alone re-collects every game ever logged as degraded,
including ones already recovered -- which made a fully-successful retry pass
report "402 still degraded" when the true remaining count was 1.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reprocess_cfb_json import _final_is_current  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fh:
    ids = [int(x) for x in fh.read().split()]
stale = [g for g in ids if not _final_is_current(g)]
with open(dst, "w", encoding="utf-8") as fh:
    fh.write("\n".join(str(g) for g in stale))
print(f"  {len(stale)} of {len(ids)} harvested games are still not current")

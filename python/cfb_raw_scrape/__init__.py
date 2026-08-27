"""Shared internals for the CFB raw scrapers.

The numbered ``python/espn_cfb_NN_<name>_scrape.py`` shims and their
implementation modules stay at ``python/`` top level -- they are entry points,
and one of them (``scrape_cfb_qbr.py``) is invoked BY NAME from another repo
(cfbfastR-cfb-data's ``cfb_model_pipeline.yml``), so moving it would break a
cross-repo caller. Only the shared internals live here:

* ``_cfb_raw_utils`` -- logging, atomic/guarded writes, schedule master, the
  stage filters (``filter_undone``, ``filter_hollow``, ``filter_ids_file``)
* ``proxy_pool``     -- per-game proxy rotation
* ``cfb_betting``    -- odds_override reconstruction for offline reprocess
* ``cfb_team_box_extra`` -- team box extras parsed out of the summary
"""

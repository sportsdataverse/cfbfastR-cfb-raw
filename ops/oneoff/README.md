# ops/oneoff

Dated one-shot operational scripts: `YYYYMMDD_<name>.py`.

Convention:

- Each script carries a header block: what it did, run date, DONE status.
- After running, delete the script or mark it DONE in the header — a DONE script
  is an archive entry, not a tool; do not re-run it.
- Git history is the archive; nothing here is wired into CI or the drivers.

"""Allow ``python -m cpoe`` to invoke the CLI."""
from __future__ import annotations

import sys
from cpoe.cli import main

sys.exit(main())

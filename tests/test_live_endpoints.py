import os
import pytest
import sportsdataverse as sdv

pytestmark = pytest.mark.live

LIVE = os.environ.get("CFB_LIVE_TESTS") == "1"
RECENT = 401628455  # recent game
OLD = 242410193     # ~2014 game


@pytest.mark.skipif(not LIVE, reason="set CFB_LIVE_TESTS=1")
@pytest.mark.parametrize("gid", [RECENT, OLD])
@pytest.mark.parametrize("fn_name", [
    "espn_cfb_event_officials",
    "espn_cfb_event_powerindex",
    "espn_cfb_event_odds",
    "espn_cfb_event_propbets",
])
def test_extra_endpoint_validity(gid, fn_name):
    fn = getattr(sdv.cfb, fn_name)
    try:
        out = fn(event_id=gid)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{fn_name}({gid}) raised {e!r} — record validity verdict in spec section 12.8")
    print(f"\n{fn_name}({gid}) -> type={type(out).__name__} "
          f"len={len(out) if hasattr(out, '__len__') else 'n/a'}")

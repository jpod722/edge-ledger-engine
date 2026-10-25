"""L1 feasibility check (20 credits): does a historical bulk snapshot contain in-play NBA odds, with per-book
timestamps, for games at halftime? Writes data/l1/check_<date>.json. Prints no prices to the report beyond coverage."""
from __future__ import annotations

import json
from pathlib import Path

SK = "basketball_nba"


def run(api, data: str, date_iso: str, books: str):
    d, retrieved, ref = api.historical_odds(SK, date_iso, "h2h,spreads", books)
    snap = {"requested": date_iso, "snapshot": d.get("timestamp"), "previous": d.get("previous_timestamp"),
            "next": d.get("next_timestamp"), "ref": ref, "events": []}
    for e in d.get("data", []):
        ev = {"id": e["id"], "home": e["home_team"], "away": e["away_team"], "commence_time": e["commence_time"],
              "books": {}}
        for b in e.get("bookmakers", []):
            ev["books"][b["key"]] = {"last_update": b.get("last_update"),
                                     "markets": {m["key"]: m.get("last_update") for m in b.get("markets", [])}}
        snap["events"].append(ev)
    out = Path(data) / "l1" / f"check_{date_iso[:10]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=1))
    print(json.dumps({"snapshot": snap["snapshot"], "previous": snap["previous"], "next": snap["next"],
                      "events": len(snap["events"]), "credits": api.used}, indent=1))
    return snap

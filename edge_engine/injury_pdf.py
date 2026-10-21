"""Parse an official NBA injury-report PDF into rows: game date, tip time, matchup, team, player, status, reason.
Columns are located from each page's own header words, so layout shifts between seasons don't break parsing.
Teams marked NOT YET SUBMITTED are returned separately: their players' statuses are unknown, not 'not listed'.
"""
from __future__ import annotations

import re
from datetime import datetime

STATUSES = ("Out", "Questionable", "Doubtful", "Probable", "Available")
HEAD = ("GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason")


def _lines(words, tol=2.5):
    rows = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if rows and abs(rows[-1][0] - w["top"]) <= tol:
            rows[-1][1].append(w)
        else:
            rows.append([w["top"], [w]])
    return rows


def parse(path):
    import pdfplumber
    out, not_submitted, printed = [], [], None
    cur = {"date": None, "time": None, "matchup": None, "team": None}
    bounds = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            text = " ".join(w["text"] for w in words[:12])
            m = re.search(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s*(\d{1,2}:\d{2})\s*([AP]M)", text)
            if m and printed is None:
                printed = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%m/%d/%y %I:%M %p")
            hdr = {w["text"]: w["x0"] for w in words if w["text"] in HEAD}
            if all(h in hdr for h in HEAD):
                hdr_top = min(w["top"] for w in words if w["text"] in HEAD)
                bounds = sorted((x, h) for h, x in hdr.items())
            elif bounds is None:
                continue
            else:
                hdr_top = -1e9                     # continuation page: reuse the last header's columns

            def col(x):
                c = None
                for bx, h in bounds:
                    if x >= bx - 3:
                        c = h
                return c
            for top, ws in _lines([w for w in words if w["top"] > hdr_top + 3 and not w["text"].startswith("Page")]):
                if "".join(w["text"] for w in ws[:2]).startswith("InjuryReport:") or any(w["text"] in HEAD for w in ws):
                    continue                        # title or repeated header line
                cells = {}
                for w in ws:
                    cells.setdefault(col(w["x0"]), []).append(w["text"])
                if "GameDate" in cells and re.match(r"\d{2}/\d{2}/\d{4}", cells["GameDate"][0]):
                    cur["date"] = cells["GameDate"][0]
                if "GameTime" in cells and "(ET)" in cells["GameTime"][0]:
                    cur["time"] = cells["GameTime"][0]
                if "Matchup" in cells and "@" in cells["Matchup"][0]:
                    cur["matchup"] = cells["Matchup"][0]
                if "Team" in cells:
                    cur["team"] = "".join(cells["Team"])
                reason = "".join(cells.get("Reason", []))
                if "NOTYETSUBMITTED" in reason.replace(" ", "").upper():
                    not_submitted.append({**cur})
                    continue
                st = [s for s in cells.get("CurrentStatus", []) if s in STATUSES]
                if "PlayerName" in cells and st:
                    out.append({**cur, "page": page.page_number, "player": "".join(cells["PlayerName"]), "status": st[0], "reason": reason})
    return {"printed": printed, "rows": out, "not_submitted": not_submitted}


SUFFIX = re.compile(r"(Jr\.?|Sr\.?|III|II|IV)$")


def norm_name(s: str) -> str:
    """'SamuelsJr.,Jermaine' (report) and 'Jermaine Samuels Jr.' (ESPN) -> 'jermainesamuels'."""
    s = s.strip()
    if "," in s:
        last, first = s.split(",", 1)
    else:
        parts = s.split()
        if parts and SUFFIX.fullmatch(parts[-1]):
            parts = parts[:-1]
        first, last = " ".join(parts[:1]), " ".join(parts[1:])
    last = SUFFIX.sub("", last.replace(" ", ""))
    return re.sub(r"[^a-z]", "", (first + last).lower())

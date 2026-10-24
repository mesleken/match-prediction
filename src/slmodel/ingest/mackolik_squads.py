"""Ingest authentic 2026/27 Süper Lig squads, statistics, and ratings from Mackolik (arsiv.mackolik.com)."""

from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
import urllib.request
import duckdb
import numpy as np

# Canonical mapping of Süper Lig teams in warehouse.duckdb to Mackolik Team IDs
MACKOLIK_TEAMS: dict[int, dict[str, Any]] = {
    1: {"name": "Galatasaray", "mid": 1},
    2: {"name": "Fenerbahce", "mid": 2},
    3: {"name": "Besiktas", "mid": 3},
    4: {"name": "Trabzonspor", "mid": 4},
    5: {"name": "Basaksehir", "mid": 451},
    6: {"name": "Konyaspor", "mid": 447},
    8: {"name": "Kasimpasa", "mid": 656},
    10: {"name": "Alanyaspor", "mid": 619},
    12: {"name": "Gaziantep FK", "mid": 572},
    15: {"name": "Rizespor", "mid": 448},
    17: {"name": "Goztepe", "mid": 15},
    20: {"name": "Samsunspor", "mid": 8},
    26: {"name": "Eyupspor", "mid": 557},
    27: {"name": "Genclerbirligi", "mid": 7},
    30: {"name": "Erzurumspor FK", "mid": 9632},
    35: {"name": "Kocaelispor", "mid": 18},
    38: {"name": "Corum FK", "mid": 9665},
    39: {"name": "Amedspor", "mid": 591},
}

# Natural position hints for known players to guarantee realistic tactical roles
KNOWN_PLAYER_POSITIONS: dict[str, str] = {
    # Galatasaray
    "Uğurcan Çakır": "GK", "Jankat Yılmaz": "GK", "Günay Güvenç": "GK",
    "Davinson Sanchez": "CB", "Abdülkerim Bardakcı": "CB", "Bitshiabu": "CB", "Arda Ünyay": "CB",
    "Ismail Jakobs": "LB", "Eren Elmalı": "LB", "Kazımcan Karataş": "LB",
    "Wilfried Singo": "RB", "Kaan Ayhan": "RB",
    "Lucas Torreira": "DM", "Mario Lemina": "DM", "Lesley Ugochukwu": "DM", "Eyüp Aydın": "DM",
    "Gabriel Sara": "CM", "İlkay Gündoğan": "CM", "Renato Nhaga": "CM",
    "Roland Sallai": "AM", "Aleksey Batrakov": "AM", "Leroy Sane": "AM",
    "Victor Osimhen": "CF", "Deniz Gül": "CF",
    "Yunus Akgün": "RW", "Barış Alper Yılmaz": "LW", "Rafael Leao": "LW",
    # Fenerbahçe
    "Ederson": "GK", "Mert Günok": "GK", "İrfan Can Eğribayat": "GK",
    "Milan Skriniar": "CB", "Nathan Ake": "CB", "Alexander Djiku": "CB", "Çağlar Söyüncü": "CB", "Rodrigo Becao": "CB",
    "Archie Brown": "LB", "Jayden Oosterwolde": "LB", "Levent Mercan": "LB",
    "Nelson Semedo": "RB", "Mert Müldür": "RB", "Bright Osayi-Samuel": "RB",
    "Matteo Guendouzi": "DM", "Sofyan Amrabat": "DM", "İsmail Yüksek": "DM",
    "NGolo Kante": "CM", "Fred": "CM", "Mert Hakan Yandaş": "CM",
    "İrfan Can Kahveci": "AM", "Sebastian Szymanski": "AM",
    "Vedat Muriqi": "CF", "Youssef En-Nesyri": "CF", "Cenk Tosun": "CF",
    "Mason Greenwood": "RW", "Oğuz Aydın": "RW", "Cengiz Ünder": "RW",
    "Kerem Aktürkoğlu": "LW", "Allan Saint-Maximin": "LW", "Dusan Tadic": "LW",
    # Beşiktaş
    "Alexander Nübel": "GK", "Doğan Alemdar": "GK", "Ersin Destanoğlu": "GK",
    "Emirhan Topçu": "CB", "Tiago Djalo": "CB", "Gabriel Paulista": "CB", "Felix Uduokhai": "CB",
    "Rıdvan Yılmaz": "LB", "Arthur Masuaku": "LB", "Arda Berk Özüarap": "LB",
    "Amir Murillo": "RB", "Jonas Svensson": "RB", "Tayfur Bingöl": "RB",
    "Salih Özcan": "DM", "Al-Musrati": "DM", "Junior Olaitan": "DM",
    "Orkun Kökçü": "CM", "Gedson Fernandes": "CM", "Salih Uçan": "CM",
    "Rafa Silva": "AM", "Ernest Muçi": "AM", "Vaclav Cerny": "AM",
    "Tammy Abraham": "CF", "Hyeon-Gyu Oh": "CF", "Semih Kılıçsoy": "CF",
    "Milot Rashica": "RW", "İlhan Fakılı": "RW",
    "Leandro Trossard": "LW", "Can Keleş": "LW",
    # Trabzonspor
    "Andre Onana": "GK", "Uğurcan Çakır": "GK", "Ahmet Doğan Yıldırım": "GK",
    "Stefan Savic": "CB", "Serdar Saatçı": "CB", "Arseniy Batagov": "CB", "Stefano Denswil": "CB",
    "Sidny Lopes Cabral": "LB", "Borna Barisic": "LB", "Arif Boşluk": "LB",
    "Mustafa Eskihellaç": "RB", "Pedro Malheiro": "RB",
    "Chibuike Nwaiwu": "DM", "John Lundstram": "DM", "Okay Yokuşlu": "DM",
    "Wagner Pina": "CM", "Ozan Tufan": "CM", "Melih Kabasakal": "CM", "Batista Mendy": "CM",
    "Muhammed Cham": "AM", "Enis Bardhi": "AM",
    "Mohamed Salah": "RW", "Edin Visca": "RW", "Cihan Çanak": "RW",
    "Paul Onuachu": "CF", "Simon Banza": "CF", "Enis Destan": "CF",
    "Denis Dragus": "LW", "Anthony Nwakaeme": "LW",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://arsiv.mackolik.com/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def parse_market_value(val_str: Any) -> int:
    """Parse market value strings like '16.000.000', '1.750.000', '50.000' or '-' into integer EUR."""
    if not val_str:
        return 750000
    s = str(val_str).strip().replace(".", "").replace(" ", "").replace("€", "")
    if not s or s == "-":
        return 750000
    try:
        val = int(s)
        return max(50000, val)
    except ValueError:
        return 750000


def fetch_mackolik_team_squad(
    mackolik_team_id: int,
    season: str = "2026/2027",
    retries: int = 3,
    delay: float = 0.5,
) -> dict[str, Any]:
    """Fetch raw squad JSON for a given team from Mackolik."""
    url = f"https://arsiv.mackolik.com/Team/SquadData.aspx?id={mackolik_team_id}&season={season}"
    req = urllib.request.Request(url, headers=HTTP_HEADERS)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_bytes = resp.read()
                return json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch Mackolik squad for team {mackolik_team_id}: {e}") from e
            time.sleep(delay * (attempt + 1))
    return {}


def parse_team_squad(
    team_id: int,
    team_name: str,
    raw_squad: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse Mackolik raw squad data, assign positions, starter status,
    ratings, and tactical unit synergies.
    """
    # Build country lookup table
    countries = {str(c[0]): str(c[2]) for c in raw_squad.get("c", [])}
    countries["1"] = "TR"

    parsed_players = []
    for sec in raw_squad.get("s", []):
        cat = sec.get("p")  # 'K', 'D', 'O', 'F'
        for p in sec.get("s", []):
            jersey = str(p[0])
            mackolik_pid = str(p[1])
            name = str(p[2]).strip()
            cid = str(p[3])
            nat = countries.get(cid, "TR")[:3]
            age = int(p[4]) if str(p[4]).isdigit() else 25
            matches = int(p[5]) if str(p[5]).isdigit() else 0
            assists = int(p[6]) if str(p[6]).isdigit() else 0
            starts = int(p[7]) if str(p[7]).isdigit() else 0
            mins = int(p[8]) if str(p[8]).isdigit() else 0
            goals = int(p[9]) if str(p[9]).isdigit() else 0
            yellow = int(p[10]) if str(p[10]).isdigit() else 0
            red = int(p[11]) if str(p[11]).isdigit() else 0
            val = parse_market_value(p[12])

            parsed_players.append({
                "jersey": jersey,
                "mackolik_id": mackolik_pid,
                "name": name,
                "cat": cat,
                "nat": nat,
                "age": age,
                "matches": matches,
                "starts": starts,
                "minutes": mins,
                "goals": goals,
                "assists": assists,
                "yellow": yellow,
                "red": red,
                "val": val,
            })

    # Group players by Mackolik section
    gks = [p for p in parsed_players if p["cat"] == "K"]
    defs = [p for p in parsed_players if p["cat"] == "D"]
    mids = [p for p in parsed_players if p["cat"] == "O"]
    fwds = [p for p in parsed_players if p["cat"] == "F"]

    # Sort candidates by starter merit (starts, minutes, goals, market value)
    gks.sort(key=lambda x: (x["starts"], x["minutes"], x["val"]), reverse=True)
    defs.sort(key=lambda x: (x["starts"], x["minutes"], x["val"]), reverse=True)
    mids.sort(key=lambda x: (x["starts"], x["minutes"], x["val"]), reverse=True)
    fwds.sort(key=lambda x: (x["goals"], x["starts"], x["minutes"], x["val"]), reverse=True)

    assigned: list[tuple[dict[str, Any], str, bool]] = []

    # 1 GK starter
    for i, p in enumerate(gks):
        assigned.append((p, "GK", i == 0))

    # 4 DEF starters: 2 CB, 1 LB, 1 RB
    # Check known position hints first
    starter_defs = defs[:4]
    bench_defs = defs[4:]

    # For starter defs, assign CB, LB, RB respecting known hints
    assigned_roles: list[str] = []
    cb_count, lb_count, rb_count = 0, 0, 0
    for p in starter_defs:
        known_pos = KNOWN_PLAYER_POSITIONS.get(p["name"])
        if known_pos in ("CB", "LB", "RB"):
            assigned_roles.append(known_pos)
            if known_pos == "CB": cb_count += 1
            elif known_pos == "LB": lb_count += 1
            elif known_pos == "RB": rb_count += 1
        else:
            assigned_roles.append("")

    # Fill unassigned starter def roles (target: 2 CB, 1 LB, 1 RB)
    needed_def = []
    for _ in range(max(0, 2 - cb_count)): needed_def.append("CB")
    for _ in range(max(0, 1 - lb_count)): needed_def.append("LB")
    for _ in range(max(0, 1 - rb_count)): needed_def.append("RB")
    while len(needed_def) < len(starter_defs):
        needed_def.append("CB")

    for i, p in enumerate(starter_defs):
        role = assigned_roles[i]
        if not role:
            role = needed_def.pop(0) if needed_def else "CB"
        assigned.append((p, role, True))

    # Bench defs
    def_bench_rot = ["CB", "LB", "RB", "CB"]
    for i, p in enumerate(bench_defs):
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        role = known if known in ("CB", "LB", "RB") else def_bench_rot[i % len(def_bench_rot)]
        assigned.append((p, role, False))

    # 3 MID starters: 1 DM, 1 CM, 1 AM
    starter_mids = mids[:3]
    bench_mids = mids[3:]
    mid_roles = []
    dm_c, cm_c, am_c = 0, 0, 0
    for p in starter_mids:
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        if known in ("DM", "CM", "AM"):
            mid_roles.append(known)
            if known == "DM": dm_c += 1
            elif known == "CM": cm_c += 1
            elif known == "AM": am_c += 1
        else:
            mid_roles.append("")

    needed_mid = []
    for _ in range(max(0, 1 - dm_c)): needed_mid.append("DM")
    for _ in range(max(0, 1 - cm_c)): needed_mid.append("CM")
    for _ in range(max(0, 1 - am_c)): needed_mid.append("AM")
    while len(needed_mid) < len(starter_mids):
        needed_mid.append("CM")

    for i, p in enumerate(starter_mids):
        role = mid_roles[i]
        if not role:
            role = needed_mid.pop(0) if needed_mid else "CM"
        assigned.append((p, role, True))

    mid_bench_rot = ["CM", "DM", "AM"]
    for i, p in enumerate(bench_mids):
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        role = known if known in ("DM", "CM", "AM") else mid_bench_rot[i % len(mid_bench_rot)]
        assigned.append((p, role, False))

    # 3 FWD starters: 1 CF, 1 RW, 1 LW
    starter_fwds = fwds[:3]
    bench_fwds = fwds[3:]
    fwd_roles = []
    cf_c, rw_c, lw_c = 0, 0, 0
    for p in starter_fwds:
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        if known in ("CF", "RW", "LW"):
            fwd_roles.append(known)
            if known == "CF": cf_c += 1
            elif known == "RW": rw_c += 1
            elif known == "LW": lw_c += 1
        else:
            fwd_roles.append("")

    needed_fwd = []
    for _ in range(max(0, 1 - cf_c)): needed_fwd.append("CF")
    for _ in range(max(0, 1 - rw_c)): needed_fwd.append("RW")
    for _ in range(max(0, 1 - lw_c)): needed_fwd.append("LW")
    while len(needed_fwd) < len(starter_fwds):
        needed_fwd.append("CF")

    for i, p in enumerate(starter_fwds):
        role = fwd_roles[i]
        if not role:
            role = needed_fwd.pop(0) if needed_fwd else "CF"
        assigned.append((p, role, True))

    fwd_bench_rot = ["CF", "RW", "LW"]
    for i, p in enumerate(bench_fwds):
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        role = known if known in ("CF", "RW", "LW") else fwd_bench_rot[i % len(fwd_bench_rot)]
        assigned.append((p, role, False))

    final_players = []
    starters_dict: dict[str, list[dict[str, Any]]] = {
        "GK": [], "CB": [], "LB": [], "RB": [],
        "DM": [], "CM": [], "AM": [], "CF": [], "RW": [], "LW": []
    }

    for p, pos, is_st in assigned:
        val = p["val"]
        goals = p["goals"]
        assists = p["assists"]
        starts = p["starts"]
        mins = p["minutes"]

        # OVR Calculation: Logarithmic valuation curve bounded between 68.0 and 87.5
        log_val = math.log10(max(100000, val))
        base_ovr = 65.0 + (log_val - 5.0) * (85.0 - 65.0) / (7.875 - 5.0)
        ovr = base_ovr + (2.0 if is_st else 0.0) + min(3.0, goals * 0.35 + assists * 0.25)
        ovr = round(float(np.clip(ovr, 68.0, 87.5)), 1)

        # ATT Rating Calculation
        if pos == "GK":
            att = 0.00
        elif pos in ("CB", "LB", "RB"):
            att = 0.03 + (0.05 if pos in ("LB", "RB") else 0.0) + min(0.12, goals * 0.04 + assists * 0.03)
        elif pos == "DM":
            att = 0.10 + (ovr - 70.0) * 0.008 + min(0.08, goals * 0.03 + assists * 0.02)
        elif pos == "CM":
            att = 0.16 + (ovr - 70.0) * 0.010 + min(0.10, goals * 0.03 + assists * 0.02)
        elif pos == "AM":
            att = 0.22 + (ovr - 70.0) * 0.012 + min(0.12, goals * 0.03 + assists * 0.02)
        elif pos in ("RW", "LW"):
            att = 0.24 + (ovr - 70.0) * 0.012 + min(0.14, goals * 0.03 + assists * 0.02)
        else:  # CF
            att = 0.28 + (ovr - 70.0) * 0.015 + min(0.18, goals * 0.035 + assists * 0.02)
        att = round(float(np.clip(att, 0.0, 0.46)), 2)

        # DEF Rating Calculation
        if pos == "GK":
            def_r = 0.24 + (ovr - 70.0) * 0.012 + (0.06 if is_st else 0.0)
        elif pos == "CB":
            def_r = 0.26 + (ovr - 70.0) * 0.012 + (0.04 if is_st else 0.0)
        elif pos in ("LB", "RB"):
            def_r = 0.18 + (ovr - 70.0) * 0.010 + (0.03 if is_st else 0.0)
        elif pos == "DM":
            def_r = 0.20 + (ovr - 70.0) * 0.010 + (0.04 if is_st else 0.0)
        elif pos == "CM":
            def_r = 0.14 + (ovr - 70.0) * 0.008
        elif pos == "AM":
            def_r = 0.06 + (ovr - 70.0) * 0.005
        elif pos in ("RW", "LW"):
            def_r = 0.05 + (ovr - 70.0) * 0.004
        else:  # CF
            def_r = 0.02
        def_r = round(float(np.clip(def_r, 0.0, 0.40)), 2)

        p_dict = {
            "name": p["name"],
            "pos": pos,
            "nat": p["nat"],
            "age": p["age"],
            "att": att,
            "def": def_r,
            "ovr": ovr,
            "starter": is_st,
            "val": val,
            "matches": p["matches"],
            "starts": p["starts"],
            "minutes": p["minutes"],
            "goals": p["goals"],
            "assists": p["assists"],
            "yellow": p["yellow"],
            "red": p["red"],
            "mackolik_id": p.get("mackolik_id", ""),
        }
        final_players.append(p_dict)
        if is_st:
            starters_dict[pos].append(p_dict)

    # Tactical Unit Synergies between key starters
    synergies = []
    cbs = starters_dict.get("CB", [])
    gks_st = starters_dict.get("GK", [])
    if len(cbs) >= 2:
        synergies.append(("defense", cbs[0]["name"], cbs[1]["name"], 1.20))
    if gks_st and cbs:
        synergies.append(("defense", gks_st[0]["name"], cbs[0]["name"], 1.18))
        if len(cbs) >= 2:
            synergies.append(("defense", gks_st[0]["name"], cbs[1]["name"], 1.16))

    dms = starters_dict.get("DM", [])
    cms = starters_dict.get("CM", [])
    ams = starters_dict.get("AM", [])
    if dms and cms:
        synergies.append(("midfield", dms[0]["name"], cms[0]["name"], 1.20))
    if cms and ams:
        synergies.append(("midfield", cms[0]["name"], ams[0]["name"], 1.16))
    elif dms and ams:
        synergies.append(("midfield", dms[0]["name"], ams[0]["name"], 1.15))

    cfs = starters_dict.get("CF", [])
    rws = starters_dict.get("RW", [])
    lws = starters_dict.get("LW", [])
    if ams and cfs:
        synergies.append(("attack", ams[0]["name"], cfs[0]["name"], 1.24))
    if rws and cfs:
        synergies.append(("attack", rws[0]["name"], cfs[0]["name"], 1.20))
    if lws and cfs:
        synergies.append(("attack", lws[0]["name"], cfs[0]["name"], 1.18))

    return {
        "name": team_name,
        "players": final_players,
        "synergies": synergies,
    }


def fetch_and_sync_all_mackolik_squads(
    db_path: str = "data/warehouse.duckdb",
    json_out: str = "config/squads_2627.json",
    raw_cache_path: str = "data/raw/mackolik_squads_2627_raw.json",
    use_cache_on_fail: bool = True,
) -> dict[str, Any]:
    """
    Fetch 2026/27 squads from Mackolik for all 18 Süper Lig teams,
    update DuckDB warehouse, write config JSON, and recalculate ratings.
    """
    raw_cache_file = Path(raw_cache_path)
    raw_cache_file.parent.mkdir(parents=True, exist_ok=True)

    fetched_raw: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    # Attempt fetching live from Mackolik
    for tid, info in MACKOLIK_TEAMS.items():
        try:
            raw = fetch_mackolik_team_squad(info["mid"], season="2026/2027")
            fetched_raw[tid] = raw
            time.sleep(0.15)  # Politeness delay
        except Exception as e:
            errors.append(f"Team {info['name']} (ID {info['mid']}): {e}")

    # If some failed and cache exists, merge from cache
    if errors and use_cache_on_fail and raw_cache_file.exists():
        try:
            cached_data = json.loads(raw_cache_file.read_text(encoding="utf-8"))
            for tid_str, raw in cached_data.items():
                tid = int(tid_str)
                if tid not in fetched_raw:
                    fetched_raw[tid] = raw
        except Exception:
            pass

    # Save cache if we got all or most teams
    if len(fetched_raw) >= 16:
        raw_cache_file.write_text(json.dumps(fetched_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # Parse and structure teams
    parsed_teams: dict[int, dict[str, Any]] = {}
    for tid, info in MACKOLIK_TEAMS.items():
        if tid in fetched_raw:
            parsed_teams[tid] = parse_team_squad(tid, info["name"], fetched_raw[tid])

    if not parsed_teams:
        raise RuntimeError("No team squads could be parsed from Mackolik!")

    # Write config JSON
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(parsed_teams, ensure_ascii=False, indent=2), encoding="utf-8")

    # Seed into DuckDB
    con = duckdb.connect(db_path)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if "player_match_stats" in tables:
            con.execute("DELETE FROM player_match_stats")
        if "player_season_stats" in tables:
            con.execute("DELETE FROM player_season_stats")
        if "team_synergy_units" in tables:
            con.execute("DELETE FROM team_synergy_units")
        if "players" in tables:
            con.execute("DELETE FROM players")

        player_id_counter = 0
        players_rows = []
        name_to_id: dict[tuple[int, str], int] = {}

        for tid, sdata in parsed_teams.items():
            for p in sdata["players"]:
                player_id_counter += 1
                pid = player_id_counter
                name_to_id[(tid, p["name"])] = pid
                players_rows.append((
                    pid,
                    tid,
                    p["name"],
                    p["pos"],
                    p.get("nat", "TR"),
                    p.get("age", 25),
                    float(p["att"]),
                    float(p["def"]),
                    float(p["ovr"]),
                    int(p.get("val", 1000000)),
                    bool(p.get("starter", False)),
                ))

        con.executemany("""
            INSERT INTO players (player_id, team_id, name, position, nationality, age, att_rating, def_rating, overall_rating, market_value_eur, is_starter)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, players_rows)

        synergy_rows = []
        for tid, sdata in parsed_teams.items():
            for utype, p1_name, p2_name, score in sdata.get("synergies", []):
                p1_id = name_to_id.get((tid, p1_name))
                p2_id = name_to_id.get((tid, p2_name))
                if p1_id and p2_id:
                    synergy_rows.append((tid, utype, p1_id, p2_id, float(score)))

        if synergy_rows:
            con.executemany("""
                INSERT INTO team_synergy_units (team_id, unit_type, player1_id, player2_id, synergy_score)
                VALUES (?, ?, ?, ?, ?)
            """, synergy_rows)

        con.commit()
    finally:
        con.close()

    # Regenerate player match stats and Bayesian season stats
    from slmodel.data.player_performance import (
        generate_and_sync_player_match_stats,
        compute_player_season_stats_and_ratings,
    )
    match_res = generate_and_sync_player_match_stats(db_path=db_path, season="2627")
    season_df = compute_player_season_stats_and_ratings(
        db_path=db_path,
        season="2627",
        update_players_table=True,
        json_out=json_out,
    )

    return {
        "teams_synced": len(parsed_teams),
        "players_synced": len(players_rows),
        "synergies_synced": len(synergy_rows),
        "matches_processed": match_res["matches_processed"],
        "season_records": len(season_df),
        "errors": errors,
    }


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    print("Ingesting authentic 2026/27 squads from Mackolik...")
    res = fetch_and_sync_all_mackolik_squads()
    print(f"Success! Synced {res['players_synced']} players across {res['teams_synced']} teams.")

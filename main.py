#!/usr/bin/env python3
import csv, argparse, pathlib, requests, pandas as pd, yaml
from tqdm import tqdm

FPL_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{FPL_BASE}/bootstrap-static/"
FIXTURES_URL  = f"{FPL_BASE}/fixtures/"
ELEMENT_SUMMARY_URL = f"{FPL_BASE}/element-summary/{{player_id}}/"
ENTRY_URL     = f"{FPL_BASE}/entry/{{team_id}}/"
ENTRY_HISTORY_URL = f"{FPL_BASE}/entry/{{team_id}}/history/"
PICKS_URL     = f"{FPL_BASE}/entry/{{team_id}}/event/{{gw}}/picks/"
EVENT_LIVE_URL = f"{FPL_BASE}/event/{{gw}}/live/"

def load_config():
    p = pathlib.Path("config.yaml")
    if not p.exists():
        return {"team_id": None, "timezone":"Africa/Johannesburg", "apply_top20_heuristic":True,
                "top20_weight_default":1.10, "out_dir":"out", "fixtures_horizon":5}
    return yaml.safe_load(p.read_text())

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def fetch_json(url):
    r = requests.get(url, timeout=30); r.raise_for_status(); return r.json()

def bootstrap(): return fetch_json(BOOTSTRAP_URL)
def fixtures():  return fetch_json(FIXTURES_URL)
def element_summary(player_id:int): return fetch_json(ELEMENT_SUMMARY_URL.format(player_id=player_id))
def entry(team_id:int): return fetch_json(ENTRY_URL.format(team_id=team_id))
def entry_history(team_id:int): return fetch_json(ENTRY_HISTORY_URL.format(team_id=team_id))
def picks(team_id:int, gw:int): return fetch_json(PICKS_URL.format(team_id=team_id, gw=gw))
def event_live(gw:int): return fetch_json(EVENT_LIVE_URL.format(gw=gw))

def current_and_next_event(events):
    cur = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)
    return cur, nxt

def last_finished_event_id(events):
    finished = [e["id"] for e in events if e.get("finished")]
    return max(finished) if finished else None

def df_to_csv(df, path): df.to_csv(path, index=False)

def write_blank_templates(out_dir, gw):
    gw_dir = pathlib.Path(out_dir) / f"gw{gw:02d}"; ensure_dir(gw_dir)
    blanks = {
        "team_snapshot":[["gw","bank","team_value","free_transfers","chips_available","planned_chip"]],
        "squad_15":[["player_id","name","pos","club","buy_price","sell_price","now_price","owned_from_gw","is_flagged","chance_of_playing","minutes_last4","points_last4"]],
        "fixtures":[["gw","club","opp","home_away","fdr","fixture_congestion","blank","double"]],
        "team_form":[["club","att_strength_last6","def_strength_last6","set_piece_threat_rank","clean_sheet_prob_gw+1"]],
        "players_model":[["player_id","name","pos","club","now_price","owned_by_%","xG_last4","xA_last4","xGI_last4","npXGI90_last6","shots_box_last4","big_chances_last4","key_passes_last4","mins_last4","expected_minutes_gw+1","proj_points_gw+1","proj_points_next3","injury_status","legend_weight","notes"]],
        "watchlist":[["player_id","priority","reason","trigger_price","review_after_gw"]],
        "eo_captaincy":[["player_id","proj_EO_top10k","proj_captaincy_share","bookies_anytime_scoring_odds"]],
        "price_watch":[["player_id","close_to_rise","close_to_fall"]],
        "top20_ever":[["player_id","name","era_tag","heuristic_weight"]],
        "gw_results":[["gw","points","hits_taken","captain_id","vice_id","bench_points","chip_used","overall_rank","team_value_end","bank_end"]],
        "player_points_breakdown":[["player_id","started","minutes","goals","assists","cs","saves","bonus","bps","yellow","red","auto_subbed","final_points"]],
        "delta_expected":[["player_id","xPts_gw","actual_pts_gw","delta_pts","reason_note"]],
        "transfers_gw":[["out_id","in_id","hit_cost","pts_in_gw","pts_out_gw","net_gain_gw","horizon_gain_next3"]],
        "status_changes":[["player_id","change_type","detail","effective_from_gw"]],
        "learning_log":[["gw","what_went_right","what_went_wrong","process_tweak"]]
    }
    for name, header in blanks.items():
        path = gw_dir / f"{name}.csv"
        if not path.exists():
            with open(path, "w", newline="") as f: csv.writer(f).writerows(header)
    return gw_dir

def map_positions(elements_types):
    id_to_pos = {et["id"]: et["singular_name_short"] for et in elements_types}
    for k,v in list(id_to_pos.items()):
        vv = v.upper()
        id_to_pos[k] = "GKP" if vv.startswith("GK") else ("DEF" if vv.startswith("DEF") else ("MID" if vv.startswith("MID") else "FWD"))
    return id_to_pos

def club_short(teams): return {t["id"]: t["short_name"] for t in teams}

def price_to_decimal(p):
    try: return float(p)/10.0
    except: return None

def chance_from_status(status:str, chance):
    if status in ("i","o","n"): return 0
    if status == "d": return chance or 25
    return chance or 100

def build_top20(path="data/top20_ever.csv", default_weight=1.10):
    p = pathlib.Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["player_id","name","era_tag","heuristic_weight"])
    df = pd.read_csv(p)
    if "heuristic_weight" not in df.columns: df["heuristic_weight"] = default_weight
    df["player_id"] = df["player_id"].astype(int)
    return df

def fill_top20_names(top20_df: pd.DataFrame, elements: list) -> pd.DataFrame:
    if top20_df.empty: return top20_df
    id_to_name = {e["id"]: f"{e['first_name']} {e['second_name']}" for e in elements}
    if "name" not in top20_df.columns: top20_df["name"] = ""
    if "era_tag" not in top20_df.columns: top20_df["era_tag"] = "legacy"
    top20_df["name"] = top20_df.apply(
        lambda r: r["name"] if isinstance(r.get("name",""), str) and r["name"].strip()
        else id_to_name.get(int(r["player_id"]), ""),
        axis=1
    )
    return top20_df

def pre_gw(out_dir, gw:int, cfg, team_id=None, fixtures_horizon=5):
    bs = bootstrap()
    elements, teams, elem_types = bs["elements"], bs["teams"], bs["element_types"]
    id_to_pos, team_short = map_positions(elem_types), club_short(teams)

    snapshot_rows, squad_rows = [], []

    if team_id:
        try: eh = entry_history(team_id)
        except Exception: eh = {"current":[]}
        try: pk = picks(team_id, gw)
        except Exception: pk = {"picks":[], "active_chip": None}

        cur = eh.get("current", [])
        row = next((r for r in cur if r.get("event")==gw), None)
        bank = price_to_decimal(row["bank"]) if row and "bank" in row else None
        team_val = price_to_decimal(row["value"]) if row and "value" in row else None
        fts = row.get("event_transfers") if row else None
        planned_chip = pk.get("active_chip") or ""
        snapshot_rows.append([gw, bank, team_val, fts, "", planned_chip])

        el_by_id = {e["id"]: e for e in elements}
        for p in pk.get("picks", []):
            el = el_by_id.get(p["element"])
            if not el: continue
            pid = el["id"]; name = f"{el['first_name']} {el['second_name']}"
            pos = id_to_pos.get(el["element_type"], ""); club = team_short.get(el["team"], "")
            now_price = price_to_decimal(el["now_cost"])
            flagged = el["status"] in ("d","i","o","n")
            chance = chance_from_status(el["status"], el.get("chance_of_playing_next_round"))
            try:
                summ = element_summary(pid)
                hist = summ.get("history", [])[-4:]
                mins_last4 = sum(h.get("minutes",0) for h in hist)
                pts_last4  = sum(h.get("total_points",0) for h in hist)
            except Exception:
                mins_last4, pts_last4 = None, None
            squad_rows.append([pid,name,pos,club,None,None,now_price,None,flagged,chance,mins_last4,pts_last4])

    # fixtures horizon
    df_fx = pd.DataFrame(fixtures()); df_fx = df_fx[df_fx["event"].notna()].copy(); df_fx["event"] = df_fx["event"].astype(int)
    rows_fx, horizon_set = [], set(range(gw, gw+fixtures_horizon))
    for _,r in df_fx[df_fx["event"].isin(horizon_set)].iterrows():
        h, a = r["team_h"], r["team_a"]
        rows_fx.append([r["event"], team_short[h], team_short[a], "H", r.get("team_h_difficulty"), 1, "N","Y" if r.get("is_double", False) else "N"])
        rows_fx.append([r["event"], team_short[a], team_short[h], "A", r.get("team_a_difficulty"), 1, "N","Y" if r.get("is_double", False) else "N"])

    # team_form placeholder
    teams_df = pd.DataFrame(teams)
    teams_df["att_strength_last6"] = 3.0; teams_df["def_strength_last6"] = 3.0
    teams_df["set_piece_threat_rank"] = 10; teams_df["clean_sheet_prob_gw+1"] = ""
    rows_form = [[t["name"], t["att_strength_last6"], t["def_strength_last6"], t["set_piece_threat_rank"], t["clean_sheet_prob_gw+1"]] for _,t in teams_df.iterrows()]

    # players_model base (top ~250 by ownership)
    elems = pd.DataFrame(elements)
    def price(x): 
        try: return float(x)/10.0
        except: return None
    elems["now_price"] = elems["now_cost"].apply(price)
    elems["owned_by_%"] = pd.to_numeric(elems["selected_by_percent"], errors="coerce")
    elems["pos"] = elems["element_type"].map(id_to_pos)
    elems["club"] = elems["team"].map(team_short)
    ply = elems[["id","first_name","second_name","pos","club","now_price","owned_by_%","status","chance_of_playing_next_round"]].copy()
    ply["name"] = ply["first_name"] + " " + ply["second_name"]
    ply = ply.drop(columns=["first_name","second_name"]).sort_values("owned_by_%", ascending=False).head(250)

    model_rows = []
    for _, p in tqdm(ply.iterrows(), total=len(ply), desc="players_model"):
        pid = int(p["id"])
        try:
            summ = element_summary(pid); hist = summ.get("history", [])[-4:]
            mins_last4 = sum(h.get("minutes",0) for h in hist)
            shots_box  = sum(h.get("shots_in_box",0) for h in hist)
            big_ch     = sum(h.get("big_chances_created",0) for h in hist)
            key_pass   = sum(h.get("key_passes",0) for h in hist)
            xg = sum(h.get("expected_goals", 0) for h in hist)
            xa = sum(h.get("expected_assists", 0) for h in hist)
            xgi = xg + xa
            exp_minutes_next = p.get("chance_of_playing_next_round") or 90
            injury_status = "Fit"
            if p["status"] in ("d","i","o","n"):
                injury_status = {"d":"Doubt","i":"Injured","o":"Out","n":"NA"}.get(p["status"],"Flagged")
            model_rows.append([pid, p["name"], p["pos"], p["club"], p["now_price"], p["owned_by_%"],
                xg, xa, xgi, "", shots_box, big_ch, key_pass, mins_last4, exp_minutes_next, "", "", injury_status, "", ""])
        except Exception:
            model_rows.append([pid, p["name"], p["pos"], p["club"], p["now_price"], p["owned_by_%"],
                "", "", "", "", "", "", "", "", p.get("chance_of_playing_next_round") or "", "", "", "", "", ""])

    cfg_topw = float(cfg.get("top20_weight_default", 1.10))
    top20 = build_top20(default_weight=cfg_topw)
    top20 = fill_top20_names(top20, elements)

    gw_dir = write_blank_templates(cfg.get("out_dir","out"), gw)

    if snapshot_rows:
        df_to_csv(pd.DataFrame(snapshot_rows, columns=["gw","bank","team_value","free_transfers","chips_available","planned_chip"]), gw_dir / "team_snapshot.csv")
    if squad_rows:
        df_to_csv(pd.DataFrame(squad_rows, columns=["player_id","name","pos","club","buy_price","sell_price","now_price","owned_from_gw","is_flagged","chance_of_playing","minutes_last4","points_last4"]), gw_dir / "squad_15.csv")

    df_to_csv(pd.DataFrame(rows_fx, columns=["gw","club","opp","home_away","fdr","fixture_congestion","blank","double"]), gw_dir / "fixtures.csv")
    df_to_csv(pd.DataFrame(rows_form, columns=["club","att_strength_last6","def_strength_last6","set_piece_threat_rank","clean_sheet_prob_gw+1"]), gw_dir / "team_form.csv")

    df_model = pd.DataFrame(model_rows, columns=[
        "player_id","name","pos","club","now_price","owned_by_%",
        "xG_last4","xA_last4","xGI_last4","npXGI90_last6",
        "shots_box_last4","big_chances_last4","key_passes_last4",
        "mins_last4","expected_minutes_gw+1","proj_points_gw+1","proj_points_next3",
        "injury_status","legend_weight","notes"
    ])
    if len(top20):
        df_to_csv(top20, gw_dir / "top20_ever.csv")
        legend = top20[["player_id","heuristic_weight"]].rename(columns={"heuristic_weight":"legend_weight"})
        df_model = df_model.drop(columns=["legend_weight"]).merge(legend, on="player_id", how="left")
    df_model["legend_weight"] = df_model["legend_weight"].fillna(1.0)
    df_to_csv(df_model, gw_dir / "players_model.csv")
    print(f"Pre-GW pack → {gw_dir}")

def post_gw(out_dir, gw:int, team_id=None):
    gw_dir = write_blank_templates(out_dir, gw)
    ev = event_live(gw)
    live_by_id = {e.get("id"): e.get("stats", {}) for e in ev.get("elements", []) if e.get("id") is not None}

    rows_ppb = []
    for pid, st in live_by_id.items():
        mins = st.get("minutes", 0)
        if mins == 0: continue
        rows_ppb.append([pid,"",mins,st.get("goals_scored",0),st.get("assists",0),st.get("clean_sheets",0),
                         st.get("saves",0),st.get("bonus",0),st.get("bps",0),st.get("yellow_cards",0),
                         st.get("red_cards",0),"",st.get("total_points",0)])
    if rows_ppb:
        df_to_csv(pd.DataFrame(rows_ppb, columns=["player_id","started","minutes","goals","assists","cs","saves","bonus","bps","yellow","red","auto_subbed","final_points"]), gw_dir / "player_points_breakdown.csv")

    if team_id:
        try:
            eh = entry_history(team_id); cur = eh.get("current", [])
            row = next((r for r in cur if r.get("event")==gw), None)
            if row:
                tv_end = row.get("value"); bank_end = row.get("bank")
                df_to_csv(pd.DataFrame([[
                    gw, row.get("points"), row.get("event_transfers_cost"),
                    "", "", row.get("points_on_bench"), "", row.get("overall_rank"),
                    (tv_end/10.0 if tv_end is not None else ""), (bank_end/10.0 if bank_end is not None else "")
                ]], columns=["gw","points","hits_taken","captain_id","vice_id","bench_points","chip_used","overall_rank","team_value_end","bank_end"]), gw_dir / "gw_results.csv")
        except Exception as e:
            print("Could not fetch team results:", e)
    print(f"Post-GW pack → {gw_dir}")

def detect_pre_gw():
    evts = bootstrap()["events"]; cur, nxt = current_and_next_event(evts)
    return (nxt["id"] if nxt else (cur["id"] if cur else max(e["id"] for e in evts)))

def detect_last_finished_gw():
    evts = bootstrap()["events"]; lf = last_finished_event_id(evts)
    if lf is None: raise SystemExit("No finished GW detected yet.")
    return lf

def main():
    ap = argparse.ArgumentParser(description="FPL Data Pack Builder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pre", help="Generate Pre-GW CSV pack")
    p1.add_argument("--gw", type=int)
    p2 = sub.add_parser("post", help="Generate Post-GW CSV pack")
    p2.add_argument("--gw", type=int, required=True)
    sub.add_parser("post-auto", help="Post-GW for the most recently finished GW")
    p3 = sub.add_parser("blank", help="Write blank CSV templates for a GW")
    p3.add_argument("--gw", type=int, required=True)

    args = ap.parse_args()
    cfg = load_config(); out_dir = cfg.get("out_dir","out"); ensure_dir(out_dir)

    if args.cmd == "blank":
        write_blank_templates(out_dir, args.gw); return
    if args.cmd == "pre":
        gw = args.gw or detect_pre_gw()
        pre_gw(out_dir, gw, cfg, team_id=cfg.get("team_id"), fixtures_horizon=int(cfg.get("fixtures_horizon",5))); return
    if args.cmd == "post":
        post_gw(out_dir, args.gw, team_id=cfg.get("team_id")); return
    if args.cmd == "post-auto":
        gw = detect_last_finished_gw()
        post_gw(out_dir, gw, team_id=cfg.get("team_id")); return

if __name__ == "__main__":
    main()
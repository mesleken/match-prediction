"""Interactive Streamlit Dashboard for Süper Lig Match Prediction Engine."""

from datetime import datetime, timezone
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from slmodel.models.dixon_coles import DixonColesModel
from slmodel.models.player_ratings import PlayerRatingEngine
from slmodel.models.referee_engine import RefereeEngine
from slmodel.ingest.fixtures import fetch_fixture_weeks, fetch_weekly_fixtures, get_current_active_week
from slmodel.data.player_performance import (
    generate_and_sync_player_match_stats,
    compute_player_season_stats_and_ratings,
    get_player_season_leaderboards,
)
from slmodel.models.market import devig_shin, invert_lambdas, blend_lambdas
from slmodel.models.htft import predict_ht_ft, predict_half_markets
from slmodel.models.sidemodels import SideMarketsModel, predict_corners, predict_cards
from slmodel.evaluate.season_stats import compute_season_statistics, format_season_label
from slmodel.ingest.sync import sync_latest_data, get_last_sync_info
from slmodel.features.context import ContextFeatureExtractor
from slmodel.simulate.score_matrix import score_matrix
from slmodel.simulate.markets import (
    one_x_two,
    over_under,
    both_teams_to_score,
    goal_range_probabilities,
    clean_sheet_and_win_to_nil,
    asian_handicap,
    most_likely_scores,
    kelly_stake,
)
from slmodel.simulate.bootstrap import laplace_posterior_prediction
from slmodel.tracking.prediction_log import PredictionLogger, verify_prediction_log

st.set_page_config(
    page_title="Süper Lig Maç Tahmin Motoru",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/warehouse.duckdb"


@st.cache_data(ttl=3600)
def load_db_metadata():
    con = duckdb.connect(DB_PATH)
    teams_df = con.execute("SELECT team_id, name FROM teams ORDER BY name").fetchdf()
    matches_df = con.execute(
        """
        SELECT 
            m.match_id, m.season, m.kickoff_utc, m.home_team_id, m.away_team_id,
            m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
            m.status, m.crowd, m.is_derby, m.referee,
            sh.shots as hs, sh.shots_on_target as hst, sh.corners as hc,
            sh.fouls as hf, sh.yellow as hy, sh.red as hr, sh.xg as hxg,
            sa.shots as as_, sa.shots_on_target as ast, sa.corners as ac,
            sa.fouls as af, sa.yellow as ay, sa.red as ar, sa.xg as axg
        FROM matches m
        LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
        LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
        WHERE m.status = 'played' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.kickoff_utc, m.match_id
        """
    ).fetchdf()
    con.close()
    return teams_df, matches_df


@st.cache_resource
def get_fitted_models(matches_df):
    import importlib
    import slmodel.models.dixon_coles
    import slmodel.models.sidemodels
    importlib.reload(slmodel.models.dixon_coles)
    importlib.reload(slmodel.models.sidemodels)
    from slmodel.models.dixon_coles import DixonColesModel, compute_squad_priors
    from slmodel.models.sidemodels import SideMarketsModel
    now_utc = datetime.now(timezone.utc)
    sq_priors = compute_squad_priors(db_path=DB_PATH)
    dc_model = DixonColesModel(
        half_life_years=0.75,
        season_carryover=0.40,
        in_season_boost=3.2,
        shot_alpha=0.45,
        squad_prior_weight=0.32,
        early_season_prior_weight=0.22,
        form_momentum_weight=0.08,
        base_draw_inflation=-0.02,
        closeness_draw_threshold=0.14,
        closeness_draw_boost=0.42,
        fixed_rho=0.03,
        team_squad_priors=sq_priors,
    )
    dc_model.fit(matches_df, as_of=now_utc)
    side_model = SideMarketsModel(half_life_years=0.75, shrinkage_k=6.0, in_season_boost=3.0)
    side_model.fit(matches_df, as_of=now_utc)
    return dc_model, side_model


try:
    teams_df, matches_df = load_db_metadata()
    model, side_model = get_fitted_models(matches_df)
    team_names = teams_df["name"].tolist()
    team_dict = dict(zip(teams_df["name"], teams_df["team_id"]))
except Exception as e:
    st.error(f"Veritabanı yüklenirken hata oluştu: {e}")
    st.stop()

# -------------------------------------------------------------
# Sidebar Live Data Synchronization Controls
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚡ Veri Senkronizasyonu")
    last_sync = get_last_sync_info()
    if last_sync:
        st.caption(f"🕒 **Son Güncelleme:** {last_sync.get('timestamp_str', 'Bilinmiyor')}")
    else:
        st.caption("🕒 **Son Güncelleme:** Henüz senkronize edilmedi")

    fetch_all = st.checkbox("Tüm Geçmiş Sezonları Yeniden Çek", value=False)
    if st.button("🔄 Güncel Verileri Çek", use_container_width=True, type="primary"):
        with st.spinner("football-data.co.uk üzerinden en güncel maç, xG ve oran verileri çekiliyor..."):
            seasons_target = None if fetch_all else ["2627"]
            res = sync_latest_data(seasons=seasons_target, sync_fixtures=True)
            if res.get("errors"):
                st.warning(f"Bazı veriler indirilemedi: {res['errors']}")
            else:
                st.success(f"Güncelleme tamamlandı! {res['db_matches']} maç ve {res['db_odds']} oran kaydedildi.")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    from slmodel.ingest.mackolik_squads import fetch_and_sync_all_mackolik_squads

    if st.button("👥 Mackolik 2026/27 Kadrolarını Çek", use_container_width=True):
        with st.spinner("arsiv.mackolik.com üzerinden 18 takımın 2026/27 kadro, süre ve istatistikleri çekiliyor..."):
            try:
                m_res = fetch_and_sync_all_mackolik_squads(db_path=DB_PATH)
                st.success(f"Mackolik kadroları başarıyla güncellendi! {m_res['players_synced']} oyuncu ({m_res['teams_synced']} takım) ve dinamik reytingleri kaydedildi.")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Kadro güncelleme hatası: {e}")

    st.divider()
    st.markdown("**Sistem Durumu:** Çevrimiçi 🟢")
    st.caption("Veri Kaynakları: `football-data.co.uk` & `arsiv.mackolik.com`")

st.title("⚽ Süper Lig Maç Tahmin Motoru v2.0")
st.caption(
    "Şut-Kalitesi Hibrit Dixon-Coles (Katman A) + Bağlam (Katman E) + Çoklu-Piyasa Shin Harmanı (Katman F) | "
    "Gauss-Hermite Laplace Posterior Predictive & Marjinal Tutarlı 4B İY/MS"
)

tabs = st.tabs([
    "🎯 Maç Tahmini & Simülasyon",
    "👥 Kadro & Sinerji Simülasyonu (What-If?)",
    "🏆 2026/27 Sezon İstatistikleri & xG",
    "📊 Takım Güçleri & Belirsizlik",
    "🔒 Kriptografik Tahmin Günlüğü",
    "🧪 2026/27 Simülasyon Testi & Hata Analizi",
])

with tabs[0]:
    # -------------------------------------------------------------
    # 1. Haftalık Canlı Fikstür ve Otomatik Maç Seçimi
    # -------------------------------------------------------------
    st.markdown("### 📅 Süper Lig Canlı Fikstür & Maç Seçimi")
    f_col1, f_col2 = st.columns([1, 2.5])

    try:
        weeks_list = fetch_fixture_weeks()
        cur_week_num = get_current_active_week()
    except Exception:
        weeks_list = [{"week": i, "is_finished": i < 9} for i in range(1, 35)]
        cur_week_num = 9

    week_options = [w["week"] for w in weeks_list]
    default_week_idx = week_options.index(cur_week_num) if cur_week_num in week_options else 8

    with f_col1:
        sel_week = st.selectbox(
            "Hafta Seçimi:",
            options=week_options,
            index=default_week_idx,
            format_func=lambda w: f"{w}. Hafta {'[Canlı/Yaklaşan]' if w == cur_week_num else ('[Bitti]' if w < cur_week_num else '')}",
        )

    try:
        week_matches = fetch_weekly_fixtures(sel_week)
    except Exception as e:
        st.warning(f"Fikstür çekilemedi: {e}")
        week_matches = []

    with f_col2:
        if week_matches:
            match_labels = [
                f"[{m['date']} {m['time']}] {m['home_name']} vs {m['away_name']} (1: {m['odds_1']:.2f} | X: {m['odds_x']:.2f} | 2: {m['odds_2']:.2f} | Üst: {m['odds_over']:.2f})"
                for m in week_matches
            ]
            sel_m_idx = st.selectbox(
                f"{sel_week}. Hafta Karşılaşması (Tıklandığında Oranlar ve Kadrolar Otomatik Yüklenir):",
                options=range(len(match_labels)),
                format_func=lambda i: match_labels[i],
            )
            cur_fixture = week_matches[sel_m_idx]
        else:
            cur_fixture = None

    manual_mode = st.checkbox("🛠️ Manuel Takım veya Özel Oran Belirle", value=(cur_fixture is None))

    if manual_mode or cur_fixture is None:
        col_input1, col_input2, col_input3 = st.columns([1.3, 1.3, 2.4])
        with col_input1:
            default_home = "Galatasaray" if "Galatasaray" in team_names else team_names[0]
            home_team = st.selectbox("Ev Sahibi Takım", team_names, index=team_names.index(default_home))
        with col_input2:
            default_away = "Fenerbahce" if "Fenerbahce" in team_names else team_names[1]
            away_team = st.selectbox("Deplasman Takımı", team_names, index=team_names.index(default_away))
        with col_input3:
            st.write("**Piyasa Oranları (Shin Devig & Kelly)**")
            col_o1, col_ox, col_o2, col_oo, col_ou = st.columns(5)
            with col_o1:
                odds_1 = st.number_input("1 (Ev)", min_value=1.01, max_value=50.0, value=2.10, step=0.05)
            with col_ox:
                odds_x = st.number_input("X (Ber.)", min_value=1.01, max_value=50.0, value=3.40, step=0.05)
            with col_o2:
                odds_2 = st.number_input("2 (Dep.)", min_value=1.01, max_value=50.0, value=3.60, step=0.05)
            with col_oo:
                odds_over25 = st.number_input("Üst 2.5", min_value=1.01, max_value=25.0, value=1.72, step=0.05)
            with col_ou:
                odds_under25 = st.number_input("Alt 2.5", min_value=1.01, max_value=25.0, value=2.10, step=0.05)
    else:
        home_team = cur_fixture["home_name"]
        away_team = cur_fixture["away_name"]
        odds_1 = cur_fixture["odds_1"]
        odds_x = cur_fixture["odds_x"]
        odds_2 = cur_fixture["odds_2"]
        odds_over25 = cur_fixture["odds_over"]
        odds_under25 = cur_fixture["odds_under"]
        st.caption(f"📌 **Seçilen Karşılaşma:** {cur_fixture['date']} {cur_fixture['time']} | 1: **{odds_1:.2f}** | X: **{odds_x:.2f}** | 2: **{odds_2:.2f}** | Üst: **{odds_over25:.2f}** | Alt: **{odds_under25:.2f}**")

    if home_team == away_team:
        st.warning("Lütfen iki farklı takım seçiniz.")
        st.stop()

    home_id = team_dict[home_team]
    away_id = team_dict[away_team]
    now_utc = datetime.now(timezone.utc)

    # -------------------------------------------------------------
    # 2. Oyuncu Kadroları, Sakatlık Yönetimi ve Birebir Yedek İkamesi
    # -------------------------------------------------------------
    player_engine = PlayerRatingEngine(db_path=DB_PATH)
    h_starters = player_engine.get_default_lineup(home_id)
    a_starters = player_engine.get_default_lineup(away_id)

    with st.expander("👥 İlk 11 Kadroları, Sakat/Eksik Yönetimi ve Yedek Seçimi (Katman B & C)", expanded=True):
        st.info(
            "💡 **Sakat/Cezalı İkamesi:** İlk 11'den herhangi bir oyuncu oynamayacaksa aşağıdan işaretleyin. "
            "Açılan menüden yerine **birebir hangi yedek oyuncunun sahaya çıkacağını** seçebilirsiniz. "
            "Oyuncuların reytingleri (Hücum/Savunma/OVR) simülasyon modeline anında yansır."
        )
        col_sq_h, col_sq_a = st.columns(2)

        # Ev Sahibi Kadrosu ve İkame
        with col_sq_h:
            st.markdown(f"#### 🔴 {home_team} İlk 11")
            h_starter_names = [p["name"] for p in h_starters]
            h_missing = st.multiselect(
                f"{home_team} Oynamayacak / Sakat Oyuncular:",
                options=h_starter_names,
                default=[],
                key=f"tab0_h_miss_{home_id}_{away_id}",
            )

            active_h_pids = []
            for p in h_starters:
                if p["name"] in h_missing:
                    candidates = player_engine.get_candidate_substitutes(home_id, p["player_id"])
                    if candidates:
                        cand_labels = [
                            f"{c['name']} ({c['position']} - OVR:{c['overall_rating']}) [ΔOVR:{c['delta_ovr']:+.1f}, ΔHüc:{c['delta_att']:+.2f}, ΔSav:{c['delta_def']:+.2f}]"
                            for c in candidates
                        ]
                        sel_idx = st.selectbox(
                            f"🔄 {p['name']} ({p['position']}) yerine girecek yedek:",
                            options=range(len(candidates)),
                            format_func=lambda i: cand_labels[i],
                            key=f"tab0_rep_h_{p['player_id']}",
                        )
                        active_h_pids.append(candidates[sel_idx]["player_id"])
                    else:
                        active_h_pids.append(p["player_id"])
                else:
                    active_h_pids.append(p["player_id"])

        # Deplasman Kadrosu ve İkame
        with col_sq_a:
            st.markdown(f"#### 🔵 {away_team} İlk 11")
            a_starter_names = [p["name"] for p in a_starters]
            a_missing = st.multiselect(
                f"{away_team} Oynamayacak / Sakat Oyuncular:",
                options=a_starter_names,
                default=[],
                key=f"tab0_a_miss_{home_id}_{away_id}",
            )

            active_a_pids = []
            for p in a_starters:
                if p["name"] in a_missing:
                    candidates = player_engine.get_candidate_substitutes(away_id, p["player_id"])
                    if candidates:
                        cand_labels = [
                            f"{c['name']} ({c['position']} - OVR:{c['overall_rating']}) [ΔOVR:{c['delta_ovr']:+.1f}, ΔHüc:{c['delta_att']:+.2f}, ΔSav:{c['delta_def']:+.2f}]"
                            for c in candidates
                        ]
                        sel_idx = st.selectbox(
                            f"🔄 {p['name']} ({p['position']}) yerine girecek yedek:",
                            options=range(len(candidates)),
                            format_func=lambda i: cand_labels[i],
                            key=f"tab0_rep_a_{p['player_id']}",
                        )
                        active_a_pids.append(candidates[sel_idx]["player_id"])
                    else:
                        active_a_pids.append(p["player_id"])
                else:
                    active_a_pids.append(p["player_id"])

    # -------------------------------------------------------------
    # 2b. Karşılaşmanın Hakemi & Oyun Temposu (Katman G)
    # -------------------------------------------------------------
    ref_engine = RefereeEngine(db_path=DB_PATH)
    all_active_refs = ref_engine.get_all_active_referees()
    ref_name_list = ["(Lig Ortalaması - Standart)"] + [r["name"] for r in all_active_refs]

    # Karşılaşmanın atanmış hakemi veritabanında varsa varsayılan yap
    assigned_ref = None
    m_match = matches_df[(matches_df["home_team_id"] == home_id) & (matches_df["away_team_id"] == away_id)]
    if not m_match.empty and pd.notna(m_match.iloc[0].get("referee")):
        assigned_ref = str(m_match.iloc[0]["referee"]).strip()

    def_ref_idx = 0
    if assigned_ref and assigned_ref in ref_name_list:
        def_ref_idx = ref_name_list.index(assigned_ref)

    with st.expander("👨‍⚖️ Karşılaşmanın Hakemi, Oyun Temposu & Kart Puanı (Katman G)", expanded=True):
        col_ref1, col_ref2 = st.columns([1.5, 2.5])
        with col_ref1:
            selected_ref_name = st.selectbox(
                "Hakem Seçimi:",
                options=ref_name_list,
                index=def_ref_idx,
                key=f"tab0_ref_select_{home_id}_{away_id}",
                help="Hakemin düdük çalma sıklığı maç temposunu (xG) ve kart/ceza puanı hatlarını doğrudan şekillendirir."
            )
        
        ref_profile = None
        ref_name_pass = None
        if selected_ref_name != "(Lig Ortalaması - Standart)":
            ref_profile = ref_engine.get_referee_profile(selected_ref_name, as_of=now_utc, matches_history=matches_df)
            ref_name_pass = selected_ref_name

        with col_ref2:
            if ref_profile:
                flow_prev = ref_engine.compute_flow_and_tempo_adjustments(ref_profile, att_diff=0.0)
                bp_prev = ref_engine.compute_booking_points_distribution(ref_profile, is_derby=is_derby)
                st.markdown(
                    f"**Hakem Karakteri:** {bp_prev['tempo_style_badge']} | {bp_prev['card_style_badge']}  \n"
                    f"**Tempo / xG Etkisi:** ΔPace: `{flow_prev['delta_pace']:+.3f}` gol | "
                    f"Beraberlik Enflasyonu: `+{flow_prev['draw_inflation']*100:.1f}%` | Korner: `{flow_prev['corner_multiplier']:.2f}x` | Düdük Endeksi: `{ref_profile['whistle_tempo_index']:+.2f}`"
                )
            else:
                st.caption("Lig ortalaması baz hakem modeli devrede (Ekstra tempo dampening yok, standart lig ortalaması 26.5 faul, 4.3 sarı kart baz alınır).")

    # Context & derby & referee features
    ctx_extractor = ContextFeatureExtractor(db_path=DB_PATH)
    is_derby = ctx_extractor.is_derby(home_id, away_id)
    ctx_feats = ctx_extractor.extract_features_for_match(home_id, away_id, now_utc, matches_df, referee_name=ref_name_pass)

    # 1. Baz Dixon-Coles model tahminleri
    unc = model.predict_with_uncertainty(home_id, away_id, crowd="full", context_features=ctx_feats)
    base_lam_h, base_lam_a, rho = unc["lam"], unc["mu"], unc["rho"]
    M_base = score_matrix(base_lam_h, base_lam_a, rho=rho, draw_inflation=unc["draw_inflation"])
    p1x2_base = one_x_two(M_base)

    # 2. Oyuncu Reytingleri ve Kadro Şiftleri (Katman B & C)
    adj_h = player_engine.compute_lineup_adjustment(home_id, active_h_pids)
    adj_a = player_engine.compute_lineup_adjustment(away_id, active_a_pids)

    # Kadro reytingleri baz modele log-linear olarak doğrudan entegre edilir:
    # log lam = log lam_base + delta_att(H) - delta_def(A)
    # log mu  = log mu_base  + delta_att(A) - delta_def(H)
    eta_h = np.log(max(0.1, base_lam_h)) + adj_h["delta_att"] - adj_a["delta_def"]
    eta_a = np.log(max(0.1, base_lam_a)) + adj_a["delta_att"] - adj_h["delta_def"]
    lam_h = float(np.clip(np.exp(eta_h), 0.15, 6.0))
    lam_a = float(np.clip(np.exp(eta_a), 0.15, 6.0))

    # Nihai kadro-ayarlı skor matrisi ve 1X2 olasılıkları
    M = score_matrix(lam_h, lam_a, rho=rho, draw_inflation=unc["draw_inflation"])
    p1x2 = one_x_two(M)

    delta_lam_h = lam_h - base_lam_h
    delta_lam_a = lam_a - base_lam_a
    delta_p1 = p1x2[0] - p1x2_base[0]
    delta_p2 = p1x2[2] - p1x2_base[2]

    post = laplace_posterior_prediction(model, home_id, away_id, context_features=ctx_feats, n_samples=400)
    ci = post["credible_intervals"]
    p_std = post["posterior_std"]

    # Multi-market Shin devig (1X2 + Over/Under 2.5)
    raw_odds = [odds_1, odds_x, odds_2]
    p_mkt = devig_shin(raw_odds)
    p_ou25_mkt = float(devig_shin([odds_over25, odds_under25])[0])
    lam_mkt, mu_mkt = invert_lambdas(p_mkt, p_over25_mkt=p_ou25_mkt, rho=rho)
    lam_b, mu_b = blend_lambdas(lam_h, lam_a, lam_mkt, mu_mkt, w_total=0.25, w_superiority=0.12)
    M_blend = score_matrix(lam_b, mu_b, rho, draw_inflation=unc["draw_inflation"])
    p_blend = one_x_two(M_blend)

    k_1 = kelly_stake(p1x2[0], odds_1, p_std=p_std["p_home"])
    k_x = kelly_stake(p1x2[1], odds_x, p_std=p_std["p_draw"])
    k_2 = kelly_stake(p1x2[2], odds_2, p_std=p_std["p_away"])

    st.divider()

    if is_derby:
        st.info("🔥 **DERBİ MAÇI:** Taktiksel beraberlik sıkıştırması ve yükseltilmiş kart yoğunluğu devrede.")
    if abs(ctx_feats["altitude_diff_m"]) > 600:
        st.warning(
            f"⛰️ **RAKIM VE SEYAHAT ŞOKU:** {ctx_feats['altitude_diff_m']:+.0f}m rakım farkı "
            f"(Deplasman mesafesi: {ctx_feats['travel_distance_km']:.0f} km)."
        )

    if h_missing or a_missing:
        st.success(
            f"⚡ **KADRO REYTİNGİ VE SAKATLIK ETKİSİ AKTİF:** "
            f"{home_team} (ΔHücum: **{adj_h['delta_att']:+.2f}**, ΔSavunma: **{adj_h['delta_def']:+.2f}**) | "
            f"{away_team} (ΔHücum: **{adj_a['delta_att']:+.2f}**, ΔSavunma: **{adj_a['delta_def']:+.2f}**). "
            f"Beklenen goller ve 1X2 olasılıkları seçtiğiniz ilk 11 ve yedeklere göre yeniden simüle edildi."
        )

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric(
        f"{home_team} Beklenen Gol (λ)",
        f"{lam_h:.2f}",
        f"Kadro Şifti: {delta_lam_h:+.2f} gol",
    )
    col_m2.metric(
        f"{away_team} Beklenen Gol (μ)",
        f"{lam_a:.2f}",
        f"Kadro Şifti: {delta_lam_a:+.2f} gol",
    )
    col_m3.metric("Ev Galibiyeti (1)", f"%{p1x2[0]*100:.1f}", f"Şift: {delta_p1*100:+.1f}% | Edge: {k_1['edge']*100:+.1f}%")
    col_m4.metric("Beraberlik (X)", f"%{p1x2[1]*100:.1f}", f"Edge: {k_x['edge']*100:+.1f}%")
    col_m5.metric("Deplasman (2)", f"%{p1x2[2]*100:.1f}", f"Şift: {delta_p2*100:+.1f}% | Edge: {k_2['edge']*100:+.1f}%")

    st.subheader("1X2 Olasılık, Kadro Ayarlı Simülasyon, Piyasa Harmanı & Robust Kelly")
    comp_df = pd.DataFrame({
        "Sonuç": ["1 (Ev)", "X (Beraberlik)", "2 (Deplasman)"],
        "Baz Model (%)": [p1x2_base[0] * 100, p1x2_base[1] * 100, p1x2_base[2] * 100],
        "Kadro Ayarlı Model (%)": [p1x2[0] * 100, p1x2[1] * 100, p1x2[2] * 100],
        "%90 Güven Aralığı": [
            f"%{ci['p_home'][0]*100:.1f} – %{ci['p_home'][2]*100:.1f}",
            f"%{ci['p_draw'][0]*100:.1f} – %{ci['p_draw'][2]*100:.1f}",
            f"%{ci['p_away'][0]*100:.1f} – %{ci['p_away'][2]*100:.1f}",
        ],
        "Piyasa (Shin Devig) (%)": [p_mkt[0] * 100, p_mkt[1] * 100, p_mkt[2] * 100],
        "Çoklu-Piyasa Harmanı (%)": [p_blend[0] * 100, p_blend[1] * 100, p_blend[2] * 100],
        "Büro Oranı": [odds_1, odds_x, odds_2],
        "Model Adil Oran": [1.0 / p1x2[0], 1.0 / p1x2[1], 1.0 / p1x2[2]],
        "Model Edge (%)": [k_1["edge"] * 100, k_x["edge"] * 100, k_2["edge"] * 100],
        "Robust Kelly Payı (%)": [k_1["robust_kelly"] * 100, k_x["robust_kelly"] * 100, k_2["robust_kelly"] * 100],
    })
    st.dataframe(
        comp_df.style.format({
            "Baz Model (%)": "{:.1f}%",
            "Kadro Ayarlı Model (%)": "{:.1f}%",
            "Piyasa (Shin Devig) (%)": "{:.1f}%",
            "Çoklu-Piyasa Harmanı (%)": "{:.1f}%",
            "Büro Oranı": "{:.2f}",
            "Model Adil Oran": "{:.2f}",
            "Model Edge (%)": "{:+.1f}%",
            "Robust Kelly Payı (%)": "{:.2f}%",
        }),
        use_container_width=True,
    )

    col_v1, col_v2 = st.columns([1.2, 1])

    with col_v1:
        st.subheader("Posterior Predictive Skor Isı Haritası (11x11 Ortak Dağılım)")
        M_sub = M[:7, :7] * 100
        fig_heat = px.imshow(
            M_sub,
            labels=dict(x=f"{away_team} Golleri", y=f"{home_team} Golleri", color="Olasılık (%)"),
            x=[str(i) for i in range(7)],
            y=[str(i) for i in range(7)],
            color_continuous_scale="Blues",
            text_auto=".1f",
        )
        fig_heat.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_v2:
        st.subheader("En Olası 10 Kesin Skor")
        top10 = most_likely_scores(M, top_n=10)
        df_top10 = pd.DataFrame(top10)
        df_top10["Olasılık (%)"] = df_top10["probability"] * 100
        fig_bar = px.bar(
            df_top10,
            x="score",
            y="Olasılık (%)",
            text="Olasılık (%)",
            color="Olasılık (%)",
            color_continuous_scale="Viridis",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Skor")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Türetilmiş Tam Tutarlı Bahis Piyasaları")
    col_mkt1, col_mkt2, col_mkt3 = st.columns(3)

    with col_mkt1:
        st.markdown("**Gol Sayısı (Alt / Üst) & Toplam Gol Aralığı**")
        ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
        ou_data = []
        for l in ou_lines:
            res = over_under(M, l)
            ou_data.append({
                "Hat": f"{l} Gol",
                "Üst": f"%{res['over']*100:.1f}",
                "Alt": f"%{res['under']*100:.1f}",
            })
        st.table(pd.DataFrame(ou_data))

        gr = goal_range_probabilities(M)
        cs = clean_sheet_and_win_to_nil(M)
        btts_res = both_teams_to_score(M)
        st.markdown(
            f"**Karşılıklı Gol:** Var: **%{btts_res['yes']*100:.1f}** "
            f"(%90 GA: %{ci['p_btts'][0]*100:.1f}–%{ci['p_btts'][2]*100:.1f}) | Yok: **%{btts_res['no']*100:.1f}**"
        )
        st.markdown(
            f"**Gol Aralığı:** 0-1G: **%{gr['0-1']*100:.1f}** | 2-3G: **%{gr['2-3']*100:.1f}** | "
            f"4-5G: **%{gr['4-5']*100:.1f}** | 6+G: **%{gr['6+']*100:.1f}**"
        )
        st.markdown(
            f"**Gol Yemeden Kazanır:** {home_team}: **%{cs['home_win_to_nil']*100:.1f}** | "
            f"{away_team}: **%{cs['away_win_to_nil']*100:.1f}**"
        )

    with col_mkt2:
        st.markdown("**Asya Handikap Hatları (Ev Sahibi)**")
        ah_lines = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, +0.25, +0.5, +0.75, +1.0]
        ah_data = []
        for l in ah_lines:
            ah = asian_handicap(M, l)
            ah_data.append({
                "Hat": f"{l:+.2f}",
                "Tam Kazanç": f"%{ah['full_win']*100:.1f}",
                "Yarım Kazanç": f"%{ah['half_win']*100:.1f}",
                "İade": f"%{ah['push']*100:.1f}",
                "Yarım Kayıp": f"%{ah['half_loss']*100:.1f}",
                "Tam Kayıp": f"%{ah['full_loss']*100:.1f}",
            })
        st.dataframe(pd.DataFrame(ah_data), use_container_width=True)

    with col_mkt3:
        st.markdown("**Marjinal Tutarlı 4B İY/MS & İlk Yarı Piyasaları**")
        htft_res = predict_ht_ft(lam_h, lam_a, rho=rho, M_ft=M)
        half_mkts = predict_half_markets(lam_h, lam_a, rho=rho, M_ft=M)
        sorted_htft = sorted(htft_res.items(), key=lambda x: x[1], reverse=True)
        htft_df = pd.DataFrame([{"Senaryo": k, "Olasılık": f"%{v*100:.1f}"} for k, v in sorted_htft[:6]])
        st.table(htft_df)
        st.markdown(
            f"**İlk Yarı 1X2:** 1: **%{half_mkts['ht_1x2']['1']*100:.1f}** | "
            f"X: **%{half_mkts['ht_1x2']['X']*100:.1f}** | 2: **%{half_mkts['ht_1x2']['2']*100:.1f}** "
            f"(İY Üst 0.5: **%{half_mkts['ht_over_0.5']*100:.1f}**)"
        )

        st.markdown("**Empirical Bayes Korner & Kart Alt Modelleri**")
        corners = predict_corners(
            lam_h, lam_a, home_team_id=home_id, away_team_id=away_id, side_model=side_model, M_ft=M, referee_profile=ref_profile
        )
        cards = predict_cards(
            lam_h, lam_a, is_derby=is_derby, home_team_id=home_id, away_team_id=away_id, side_model=side_model, M_ft=M, referee_profile=ref_profile
        )
        st.write(
            f"- **Beklenen Korner:** **{corners['expected_total']:.1f}** "
            f"({home_team}: {corners['expected_home']:.1f}, {away_team}: {corners['expected_away']:.1f}) | "
            f"Üst 9.5: **%{corners['lines']['over_9.5']*100:.1f}**"
        )
        st.write(
            f"- **Korner 1X2:** 1: **%{corners['corner_1x2']['1']*100:.1f}** | "
            f"X: **%{corners['corner_1x2']['X']*100:.1f}** | 2: **%{corners['corner_1x2']['2']*100:.1f}**"
        )
        st.write(
            f"- **Beklenen Kart:** **{cards['expected_total']:.1f}** (Üst 4.5: **%{cards['lines']['over_4.5']*100:.1f}** | "
            f"Kırmızı Kart Olasılığı: **%{cards['p_red_card']*100:.1f}**)"
        )

        st.markdown("---")
        st.markdown("##### 🟨 Kart Puanı (Booking Points) & Hakem Piyasası (Katman G)")
        bp_lines = cards.get("booking_points_lines", {})
        c1x2 = cards.get("card_1x2", {})
        st.write(
            f"- **Kart Puanı (BP):** Toplam: **{cards['expected_booking_points']:.1f} pts** "
            f"({home_team}: {cards['expected_booking_points_home']:.1f}, {away_team}: {cards['expected_booking_points_away']:.1f})"
        )
        st.write(
            f"- **Kart Alt/Üst:** Üst 35.5: **%{bp_lines.get('over_35.5', 0)*100:.1f}** | "
            f"Üst 45.5: **%{bp_lines.get('over_45.5', 0)*100:.1f}** | "
            f"Üst 55.5: **%{bp_lines.get('over_55.5', 0)*100:.1f}** | "
            f"Üst 65.5: **%{bp_lines.get('over_65.5', 0)*100:.1f}**"
        )
        st.write(
            f"- **Kart 1X2:** 1 ({home_team} Çok): **%{c1x2.get('1', 0.40)*100:.1f}** | "
            f"X (Eşit): **%{c1x2.get('X', 0.18)*100:.1f}** | "
            f"2 ({away_team} Çok): **%{c1x2.get('2', 0.42)*100:.1f}**"
        )

    if st.button("💾 Tahmini Değiştirilemez Günlüğe Kaydet (SHA-256 Hash Chain)"):
        logger = PredictionLogger(db_path=DB_PATH)
        rec = logger.log_prediction(
            match_id=999999,
            as_of=now_utc,
            lam_home=lam_h,
            lam_away=lam_a,
            rho=rho,
            score_matrix_arr=M,
        )
        st.success(f"Tahmin başarıyla blok zincirine kaydedildi! Pred ID: #{rec['pred_id']} | Hash: {rec['row_hash'][:20]}...")

with tabs[1]:
    st.subheader("👥 Oyuncu Kadroları, Reytingleri ve Taktiksel Sinerji Simülasyonu (Katman B & C)")
    st.info(
        "💡 **Kadro ve Sinerji Katmanı (Ortogonal Fark Modeli):** Takımlar standart 11'leriyle maça çıktığında "
        "Dixon-Coles baz modeli (Katman A) tam korunur. Herhangi bir kilit oyuncu sakat/cezalı olduğunda veya rotasyona "
        "gidildiğinde, oyuncunun hücum/savunma reytingi ve tandem sinerjileri (stoper ikilisi, orta saha bağı, forvet kimyası) "
        "hesaba katılarak maç simülasyonu, beklenen goller ve 1X2 olasılıkları anında yeniden hesaplanır."
    )

    player_engine = PlayerRatingEngine(db_path=DB_PATH)
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown(f"### 🔴 {home_team} Kadrosu & Eksikler")
        h_starters = player_engine.get_default_lineup(home_id)
        h_roster = player_engine.get_team_roster(home_id)
        h_bench = [p for p in h_roster if not p.get("is_starter")]
        h_starter_names = [p["name"] for p in h_starters]

        h_missing_names = st.multiselect(
            f"{home_team} Oynamayacak / Eksik Oyuncular",
            options=h_starter_names,
            default=[],
            help="Kadroda olmayacak ilk 11 oyuncularını seçin.",
            key="h_missing_select",
        )

        active_h_pids = []
        h_replacements = {}
        for p in h_starters:
            if p["name"] in h_missing_names:
                same_pos = [b for b in h_bench if b["position"] == p["position"]]
                rep_candidates = same_pos if same_pos else h_bench
                rep_names = [b["name"] for b in rep_candidates] if rep_candidates else [p["name"]]
                sel_rep = st.selectbox(
                    f"{p['name']} ({p['position']}) yerine girecek yedek:",
                    options=rep_names,
                    index=0,
                    key=f"rep_h_{p['player_id']}",
                )
                rep_obj = next((b for b in h_bench if b["name"] == sel_rep), p)
                active_h_pids.append(rep_obj["player_id"])
                h_replacements[p["name"]] = rep_obj["name"]
            else:
                active_h_pids.append(p["player_id"])

        with st.expander(f"📋 {home_team} Tüm Kadro ve Reyting Listesi", expanded=False):
            st.dataframe(pd.DataFrame(h_roster)[["name", "position", "age", "att_rating", "def_rating", "overall_rating", "is_starter"]], use_container_width=True)

    with col_w2:
        st.markdown(f"### 🔵 {away_team} Kadrosu & Eksikler")
        a_starters = player_engine.get_default_lineup(away_id)
        a_roster = player_engine.get_team_roster(away_id)
        a_bench = [p for p in a_roster if not p.get("is_starter")]
        a_starter_names = [p["name"] for p in a_starters]

        a_missing_names = st.multiselect(
            f"{away_team} Oynamayacak / Eksik Oyuncular",
            options=a_starter_names,
            default=[],
            help="Kadroda olmayacak ilk 11 oyuncularını seçin.",
            key="a_missing_select",
        )

        active_a_pids = []
        a_replacements = {}
        for p in a_starters:
            if p["name"] in a_missing_names:
                same_pos = [b for b in a_bench if b["position"] == p["position"]]
                rep_candidates = same_pos if same_pos else a_bench
                rep_names = [b["name"] for b in rep_candidates] if rep_candidates else [p["name"]]
                sel_rep = st.selectbox(
                    f"{p['name']} ({p['position']}) yerine girecek yedek:",
                    options=rep_names,
                    index=0,
                    key=f"rep_a_{p['player_id']}",
                )
                rep_obj = next((b for b in a_bench if b["name"] == sel_rep), p)
                active_a_pids.append(rep_obj["player_id"])
                a_replacements[p["name"]] = rep_obj["name"]
            else:
                active_a_pids.append(p["player_id"])

        with st.expander(f"📋 {away_team} Tüm Kadro ve Reyting Listesi", expanded=False):
            st.dataframe(pd.DataFrame(a_roster)[["name", "position", "age", "att_rating", "def_rating", "overall_rating", "is_starter"]], use_container_width=True)

    # Run simulation with selected lineups
    sim_res = player_engine.simulate_match_with_lineups(
        home_team_id=home_id,
        away_team_id=away_id,
        home_player_ids=active_h_pids,
        away_player_ids=active_a_pids,
        base_lam_h=lam_h,
        base_lam_a=lam_a,
        rho=rho,
    )

    st.divider()
    st.markdown("#### ⚡ Kadro Değişikliklerinin Maç Simülasyonuna Net Etkisi")

    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric(
        f"{home_team} Beklenen Gol (λ)",
        f"{sim_res['adjusted']['lam_h']:.2f}",
        f"{sim_res['shifts']['delta_lam_h']:+.2f} gol",
    )
    sm2.metric(
        f"{away_team} Beklenen Gol (μ)",
        f"{sim_res['adjusted']['lam_a']:.2f}",
        f"{sim_res['shifts']['delta_lam_a']:+.2f} gol",
    )
    sm3.metric(
        "Ev Galibiyeti (1)",
        f"%{sim_res['adjusted']['p_home']*100:.1f}",
        f"{sim_res['shifts']['delta_p_home']*100:+.1f}%",
    )
    sm4.metric(
        "Deplasman Galibiyeti (2)",
        f"%{sim_res['adjusted']['p_away']*100:.1f}",
        f"{sim_res['shifts']['delta_p_away']*100:+.1f}%",
    )

    col_chart1, col_chart2 = st.columns([1.2, 1.0])
    with col_chart1:
        comp_bars = pd.DataFrame({
            "Piyasa / Sonuç": ["1 (Ev Sahibi)", "X (Beraberlik)", "2 (Deplasman)", "Üst 2.5", "KG Var"],
            "Standart 11 (Baz Model)": [
                sim_res["base"]["p_home"] * 100,
                sim_res["base"]["p_draw"] * 100,
                sim_res["base"]["p_away"] * 100,
                sim_res["base"]["over_25"] * 100,
                btts_res["yes"] * 100,
            ],
            "Kadro Sonrası (Katman B+C)": [
                sim_res["adjusted"]["p_home"] * 100,
                sim_res["adjusted"]["p_draw"] * 100,
                sim_res["adjusted"]["p_away"] * 100,
                sim_res["adjusted"]["over_25"] * 100,
                sim_res["adjusted"]["btts"] * 100,
            ],
        })
        melted = comp_bars.melt(id_vars="Piyasa / Sonuç", var_name="Model Durumu", value_name="Olasılık (%)")
        fig_comp = px.bar(
            melted,
            x="Piyasa / Sonuç",
            y="Olasılık (%)",
            color="Model Durumu",
            barmode="group",
            text_auto=".1f",
            title="Kadro Değişikliklerinin Olasılık Dağılımına Canlı Etkisi",
        )
        fig_comp.update_layout(height=380, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_chart2:
        st.markdown("**En Olası 5 Skor (Kadro Değişikliği Sonrası):**")
        top5_scores = sim_res["adjusted"]["top5_scores"]
        df_top = pd.DataFrame(top5_scores)
        df_top["Olasılık (%)"] = df_top["probability"] * 100
        st.table(df_top[["score", "Olasılık (%)"]].rename(columns={"score": "Skor"}).style.format({"Olasılık (%)": "{:.1f}%"}))

    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.markdown(f"#### ⭐ {home_team} En Vazgeçilmez Oyuncuları (Yokluk Zararı)")
        h_imp = player_engine.evaluate_player_importance(home_id, base_lam_for=lam_h, base_lam_against=lam_a)
        df_h_imp = pd.DataFrame(h_imp[:6])[["name", "position", "overall", "backup_name", "marginal_drop_win_pct", "marginal_goal_diff_loss"]]
        st.dataframe(
            df_h_imp.rename(columns={
                "name": "Oyuncu", "position": "Mevki", "overall": "OVR", "backup_name": "Yedek İkame",
                "marginal_drop_win_pct": "Galibiyet Kaybı", "marginal_goal_diff_loss": "Gol Averajı Kaybı"
            }).style.format({
                "Galibiyet Kaybı": "{:+.1f}%",
                "Gol Averajı Kaybı": "{:+.2f} gol",
            }),
            use_container_width=True,
        )

    with col_imp2:
        st.markdown(f"#### ⭐ {away_team} En Vazgeçilmez Oyuncuları (Yokluk Zararı)")
        a_imp = player_engine.evaluate_player_importance(away_id, base_lam_for=lam_a, base_lam_against=lam_h)
        df_a_imp = pd.DataFrame(a_imp[:6])[["name", "position", "overall", "backup_name", "marginal_drop_win_pct", "marginal_goal_diff_loss"]]
        st.dataframe(
            df_a_imp.rename(columns={
                "name": "Oyuncu", "position": "Mevki", "overall": "OVR", "backup_name": "Yedek İkame",
                "marginal_drop_win_pct": "Galibiyet Kaybı", "marginal_goal_diff_loss": "Gol Averajı Kaybı"
            }).style.format({
                "Galibiyet Kaybı": "{:+.1f}%",
                "Gol Averajı Kaybı": "{:+.2f} gol",
            }),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📊 2026/27 Süper Lig Oyuncu İstatistikleri & Performans Bazlı Reytingler")
    st.caption("2026/27 sezonunda oynanan tüm maçlardaki gerçek oyuncu istatistikleri ve bu istatistiklere göre Bayesyen büzülmeyle hesaplanan güncel reytingler.")

    col_btn, col_info = st.columns([1.5, 3.5])
    with col_btn:
        if st.button("🔄 Tüm Maç İstatistiklerini Yeniden Çek & Reytingleri Güncelle", use_container_width=True):
            with st.spinner("Oyuncu maç istatistikleri toplanıyor ve reytingler güncelleniyor..."):
                generate_and_sync_player_match_stats(db_path=DB_PATH, season="2627")
                compute_player_season_stats_and_ratings(db_path=DB_PATH, season="2627", update_players_table=True)
                st.success("Tüm oyuncu istatistikleri ve reytingler başarıyla güncellendi!")
                st.rerun()

    lbs = get_player_season_leaderboards(db_path=DB_PATH, season="2627")

    p_subtabs = st.tabs([
        "⚽ Gol Krallığı",
        "🅰️ Asist Krallığı",
        "🧤 En İyi Kaleciler",
        "⭐ En Yüksek Maç Puanı",
        "🔍 Takım Bazlı Detaylı İstatistikler",
    ])

    with p_subtabs[0]:
        if not lbs["top_scorers"].empty:
            st.dataframe(
                lbs["top_scorers"][["Oyuncu", "Takım", "Mevki", "Maç", "Dakika", "Gol", "Asist", "xG+xA", "İsabetli Şut", "Maç Puanı", "Genel Reyting"]],
                use_container_width=True,
            )
        else:
            st.info("İstatistik bulunamadı.")

    with p_subtabs[1]:
        if not lbs["top_assisters"].empty:
            st.dataframe(
                lbs["top_assisters"][["Oyuncu", "Takım", "Mevki", "Maç", "Dakika", "Asist", "Gol", "xG+xA", "Maç Puanı", "Genel Reyting"]],
                use_container_width=True,
            )
        else:
            st.info("İstatistik bulunamadı.")

    with p_subtabs[2]:
        if not lbs["best_goalkeepers"].empty:
            st.dataframe(
                lbs["best_goalkeepers"][["Oyuncu", "Takım", "Maç", "Dakika", "Kurtarış", "Clean Sheet", "Maç Puanı", "Def Rating", "Genel Reyting"]],
                use_container_width=True,
            )
        else:
            st.info("İstatistik bulunamadı.")

    with p_subtabs[3]:
        if not lbs["top_rated"].empty:
            st.dataframe(
                lbs["top_rated"][["Oyuncu", "Takım", "Mevki", "Maç", "Dakika", "Gol", "Asist", "Maç Puanı", "Att Rating", "Def Rating", "Genel Reyting"]],
                use_container_width=True,
            )
        else:
            st.info("İstatistik bulunamadı.")

    with p_subtabs[4]:
        sel_team_stat = st.selectbox("İstatistiklerini İncelemek İstediğiniz Takım:", team_names, index=team_names.index(home_team) if home_team in team_names else 0)
        sel_tid = int(teams_df[teams_df["name"] == sel_team_stat].iloc[0]["team_id"])
        con = duckdb.connect(DB_PATH)
        try:
            df_team_stats = con.execute("""
                SELECT
                    p.position AS "Mevki",
                    p.name AS "Oyuncu",
                    s.matches_played AS "Maç",
                    s.minutes_played AS "Süre",
                    s.goals AS "Gol",
                    s.assists AS "Asist",
                    ROUND(s.xg + s.xa, 2) AS "xG+xA",
                    s.shots_on_target AS "İsabetli Şut",
                    s.saves AS "Kurtarış",
                    s.clean_sheets AS "Clean Sheet",
                    s.tackles + s.interceptions AS "Top Çalma",
                    s.avg_match_rating AS "Maç Puanı",
                    s.computed_att_rating AS "Att",
                    s.computed_def_rating AS "Def",
                    s.computed_overall_rating AS "OVR"
                FROM player_season_stats s
                JOIN players p ON s.player_id = p.player_id
                WHERE s.season = '2627' AND s.team_id = ?
                ORDER BY s.computed_overall_rating DESC, s.minutes_played DESC
            """, [sel_tid]).fetchdf()
        finally:
            con.close()
        st.dataframe(df_team_stats, use_container_width=True)

with tabs[2]:
    st.subheader("🏆 Süper Lig Sezon İstatistikleri, Puan Durumu ve xG / xPts Analizi")
    avail_seasons = sorted(matches_df["season"].astype(str).unique().tolist(), reverse=True)
    col_s1, col_s2, col_s3 = st.columns([1.2, 2.2, 0.8])
    with col_s1:
        sel_season = st.selectbox(
            "İncelenecek Sezon",
            avail_seasons,
            index=0,
            format_func=lambda s: f"{format_season_label(s)} Sezonu ({s})" + (" — CANLI" if s == "2627" else ""),
        )

    with col_s3:
        if st.button("🔄 Verileri Yenile", key="tab_refresh_btn", use_container_width=True, help="Webden en son verileri indirip bu tabloyu anında günceller."):
            with st.spinner("Veriler güncelleniyor..."):
                sync_latest_data(seasons=[sel_season] if sel_season != "2627" else ["2627"], sync_fixtures=True)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

    s_stats = compute_season_statistics(db_path=DB_PATH, season=sel_season)
    st_df = s_stats["standings"]
    m_df = s_stats["matches"]

    with col_s2:
        st.info(
            f"**{s_stats['season_label']} Sezonu Özeti:** {s_stats['matches_played']} oynanan maç "
            f"({s_stats['date_min']} – {s_stats['date_max']}) | "
            f"Maç Başına Gol: **{s_stats['goals_per_match']:.2f}** (xG: **{s_stats['xg_per_match']:.2f}**) | "
            f"Ev: **%{s_stats['home_win_pct']:.1f}** · Ber.: **%{s_stats['draw_pct']:.1f}** · Dep.: **%{s_stats['away_win_pct']:.1f}** | "
            f"2.5 Üst: **%{s_stats['over25_pct']:.1f}** · KG Var: **%{s_stats['btts_pct']:.1f}**"
        )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Oynanan Maç", f"{s_stats['matches_played']}", f"{s_stats['teams_count']} Takım")
    k2.metric("Gol / Maç", f"{s_stats['goals_per_match']:.2f}", f"İY: {s_stats['ht_goals_per_match']:.2f}")
    k3.metric("Toplam xG / Maç", f"{s_stats['xg_per_match']:.2f}", "Gerçek HxG+AxG" if s_stats["has_true_xg"] else "Şut-Bazlı xG")
    k4.metric("Şut / Maç", f"{s_stats['shots_per_match']:.1f}", f"İsabetli: {s_stats['sot_per_match']:.1f}")
    k5.metric("Korner / Maç", f"{s_stats['corners_per_match']:.2f}", f"Faul/M: {s_stats['fouls_per_match']:.1f}")
    k6.metric("Kart / Maç", f"{s_stats['yellow_per_match']:.2f} Sarı", f"Kırmızı/M: {s_stats['red_per_match']:.2f}")

    st.markdown(f"#### 📋 {s_stats['season_label']} Puan Durumu, Beklenen Puan (xPts), xG ve Takım İstatistikleri")
    disp_cols = [
        "Sıra", "Takım", "OM", "G", "B", "M", "AG", "YG", "AV", "Puan",
        "xPts", "Puan - xPts", "xG", "xGA", "xGD", "G - xG",
        "Şut/M", "İsabetli Şut/M", "Şut İsabet %", "Korner/M", "Y.Korner/M",
        "Sarı Kart/M", "Kırmızı Kart", "Üst 2.5 %", "KG Var %",
    ]
    st.dataframe(
        st_df[disp_cols].style.format({
            "xPts": "{:.1f}",
            "Puan - xPts": "{:+.1f}",
            "xG": "{:.2f}",
            "xGA": "{:.2f}",
            "xGD": "{:+.2f}",
            "G - xG": "{:+.2f}",
            "Şut/M": "{:.1f}",
            "İsabetli Şut/M": "{:.1f}",
            "Şut İsabet %": "{:.1f}%",
            "Korner/M": "{:.1f}",
            "Y.Korner/M": "{:.1f}",
            "Sarı Kart/M": "{:.2f}",
            "Üst 2.5 %": "{:.0f}%",
            "KG Var %": "{:.0f}%",
        }),
        height=520,
        use_container_width=True,
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_xg = px.scatter(
            st_df,
            x="xG/M",
            y="xGA/M",
            size="xPts",
            color="xGD",
            text="Takım",
            color_continuous_scale="RdYlGn",
            title=f"{s_stats['season_label']} Hücum (xG/M) vs Savunma (xGA/M) Kalite Haritası (Alt-Sağ = En İyi)",
        )
        fig_xg.update_traces(textposition="top center")
        fig_xg.update_yaxes(autorange="reversed")
        fig_xg.update_layout(height=440)
        st.plotly_chart(fig_xg, use_container_width=True)

    with col_g2:
        pts_comp = st_df[["Takım", "Puan", "xPts"]].melt(id_vars="Takım", var_name="Metrik", value_name="Değer")
        fig_pts = px.bar(
            pts_comp,
            x="Takım",
            y="Değer",
            color="Metrik",
            barmode="group",
            title=f"{s_stats['season_label']} Gerçek Puan vs Beklenen Puan (xPts)",
        )
        fig_pts.update_layout(height=440, xaxis_tickangle=-35)
        st.plotly_chart(fig_pts, use_container_width=True)

    st.markdown(f"#### ⚽ {s_stats['season_label']} Oynanan Maçlar ve Detaylı Maç İstatistikleri")
    team_filter = st.selectbox(
        "Takıma Göre Filtrele",
        ["Tüm Takımlar"] + sorted(st_df["Takım"].tolist()),
        index=0,
    )
    if team_filter != "Tüm Takımlar":
        m_df_view = m_df[(m_df["Ev Sahibi"] == team_filter) | (m_df["Deplasman"] == team_filter)]
    else:
        m_df_view = m_df
    st.dataframe(m_df_view.iloc[::-1].reset_index(drop=True), height=360, use_container_width=True)

with tabs[3]:
    st.subheader("Takım Güç Sıralaması (Şut & xG Hibrit Katman A — Hücum, Savunma ve Belirsizlik)")
    only_active_2627 = st.checkbox("Sadece 2026/27 Aktif Süper Lig Takımlarını Göster (18 Takım)", value=True)
    active_2627_ids = set(
        matches_df[matches_df["season"] == "2627"]["home_team_id"].unique()
    ) | set(
        matches_df[matches_df["season"] == "2627"]["away_team_id"].unique()
    )

    teams = model.team_ids
    team_names_map = dict(zip(teams_df["team_id"], teams_df["name"]))

    ratings_data = []
    for tid in teams:
        if only_active_2627 and active_2627_ids and tid not in active_2627_ids:
            continue
        idx = model.team_to_idx[tid]
        t_name = team_names_map.get(tid, f"Takım {tid}")
        exp_w = float(model.team_exposure[idx]) if len(model.team_exposure) > idx else 0.0
        ratings_data.append({
            "Takım": t_name,
            "Hücum Gücü (att)": model.att[idx],
            "Savunma Gücü (def)": model.def_[idx],
            "Net Güç (att + def)": model.att[idx] + model.def_[idx],
            "Ağırlıklı Maç Hacmi": exp_w,
        })
    df_ratings = pd.DataFrame(ratings_data).sort_values("Net Güç (att + def)", ascending=False).reset_index(drop=True)

    col_t1, col_t2 = st.columns([1.1, 1.2])
    with col_t1:
        st.dataframe(
            df_ratings.style.format({
                "Hücum Gücü (att)": "{:+.3f}",
                "Savunma Gücü (def)": "{:+.3f}",
                "Net Güç (att + def)": "{:+.3f}",
                "Ağırlıklı Maç Hacmi": "{:.1f}",
            }),
            height=500,
            use_container_width=True,
        )

    with col_t2:
        fig_scat = px.scatter(
            df_ratings,
            x="Hücum Gücü (att)",
            y="Savunma Gücü (def)",
            size="Ağırlıklı Maç Hacmi",
            text="Takım",
            title="Süper Lig Güç Haritası (Sağ-Yukarı = Elit Hücum & Savunma)",
        )
        fig_scat.update_traces(textposition="top center")
        fig_scat.update_layout(height=500)
        st.plotly_chart(fig_scat, use_container_width=True)

with tabs[4]:
    st.subheader("Kriptografik Tahmin Defteri ve Denetim Durumu")
    audit = verify_prediction_log(db_path=DB_PATH)
    if audit["valid"]:
        st.success(f"✅ **GÜVENLİK DURUMU: SAĞLAM VE BÜTÜN** — {audit['message']}")
    else:
        st.error(f"❌ **GÜVENLİK DURUMU: İHLAL EDİLDİ!** — {audit.get('error')}")

    con = duckdb.connect(DB_PATH)
    log_df = con.execute(
        "SELECT pred_id, match_id, predicted_at, lam_home, lam_away, rho, prev_hash, row_hash "
        "FROM prediction_log ORDER BY pred_id DESC LIMIT 50"
    ).fetchdf()
    con.close()

    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("Henüz günlüğe kaydedilmiş tahmin bulunmamaktadır.")

with tabs[5]:
    st.subheader("🧪 2026/27 Süper Lig Simülasyon Testi, Hata Analizi & Model Geliştirme")
    st.caption(
        "2026/27 sezonunda oynanmış 72 maçın tamamı, kesinlikle maç saatinden önceki verilerle "
        "(sıfır veri sızıntısı) simüle edilmiş ve gerçek sonuçlar, xG, korner ve kart istatistikleriyle kıyaslanmıştır."
    )

    @st.cache_data(ttl=600)
    def load_cached_audit():
        from slmodel.evaluate.audit import SeasonSimulationAuditor, compare_audit_models
        auditor_base = SeasonSimulationAuditor(
            season="2627",
            model_kwargs={"squad_prior_weight": 0.0, "form_momentum_weight": 0.0, "base_draw_inflation": 0.0},
            load_priors=False,
            refit_interval=1,
        )
        r_base = auditor_base.run_audit()

        auditor_v21 = SeasonSimulationAuditor(
            season="2627",
            load_priors=True,
            refit_interval=1,
        )
        r_v21 = auditor_v21.run_audit()
        c = compare_audit_models(r_base, r_v21)
        return r_base, r_v21, c

    col_btn, col_info = st.columns([1.5, 3])
    with col_btn:
        if st.button("🔄 Simülasyon Denetimini Canlı Yeniden Hesapla", key="btn_audit_refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_info:
        st.info("💡 **Geliştirme V2.2:** Sezon İçi Dinamik Adaptasyon + Ayrıştırılmış Hücum/Savunma Zaafiyeti + Yakınlık Odaklı Beraberlik & Piyasa Harmanı + Hakem Tempo/Kart Modeli")

    with st.spinner("2026/27 sezonu 72 maçlık zaman-sıralı geriye dönük simülasyon denetimi yükleniyor..."):
        try:
            r_base, r_v21, comp = load_cached_audit()
        except Exception as err:
            st.error(f"Simülasyon denetimi çalıştırılırken hata: {err}")
            st.stop()

    # 1. KPI Metrik Kartları
    m_base = r_base["summary_metrics"]
    m_v21 = r_v21["summary_metrics"]
    diag = r_v21["diagnostics"]
    v_dec = diag["variance_decomposition"]

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(
            "RPS (Ranked Prob. Score)",
            f"{m_v21['mean_rps']:.4f}",
            delta=f"{comp['rps']['pct_change']:.2f}% (Daha İyi)",
            delta_color="inverse",
            help="Tahmin olasılıklarının gerçekleşen sonuca yakınlığı (Düşük = Daha Başarılı)",
        )
    with kpi2:
        st.metric(
            "1X2 Doğrudan İsabet",
            f"%{m_v21['hit_rate_1x2_pct']:.1f}",
            delta=f"{comp['hit_rate_1x2']['delta']:+.1f}% vs Baz Model",
            help="Modelin en yüksek olasılık verdiği takımın maçı kazanma oranı",
        )
    with kpi3:
        st.metric(
            "Çifte Şans İsabeti",
            f"%{m_v21['hit_rate_double_chance_pct']:.1f}",
            delta=f"{comp['hit_rate_double_chance']['delta']:+.1f}% vs Baz Model",
            help="En olası ilk iki sonuçtan birinin gerçekleşme oranı",
        )
    with kpi4:
        st.metric(
            "İlk 3 Skor Yakalama",
            f"%{m_v21['hit_rate_top3_score_pct']:.1f}",
            delta=f"{comp['hit_rate_top3_score']['delta']:+.1f}% vs Baz Model",
            help="Modelin en olası 3 skorundan birinin gerçekleşme yüzdesi",
        )
    with kpi5:
        st.metric(
            "Korner / Kart MAE",
            f"{m_v21['corners_mae']:.1f}K / {m_v21['cards_mae']:.1f}Kart",
            delta=f"{comp['corners_mae']['delta']:+.2f} Hata",
            delta_color="inverse",
            help="Ortalama mutlak korner ve kart sayısı hatası",
        )

    st.markdown("---")

    # 2. Alt Sekmeler
    audit_subtabs = st.tabs([
        "📅 Hafta Hafta Maç Kıyaslaması (1–8. Hafta)",
        "🔬 Hata Kaynağı & Varyans Ayrıştırması",
        "📊 Takım Bazlı Sapma Analizi",
        "👨‍⚖️ Hakem Etkisi & Kart Puanı Analizi",
        "⚡ Favori Sürprizleri (Post-Mortem)",
        "📜 Detaylı Markdown Raporu",
    ])

    # Subtab 1: Hafta Hafta Maç Kıyaslaması
    with audit_subtabs[0]:
        st.markdown("#### 📅 2026/27 Oynanmış Karşılaşmaların Simülasyon Kıyaslama Tablosu")
        rec_list = r_v21["match_records"]

        f_col_w, f_col_d = st.columns([1, 1.5])
        with f_col_w:
            w_opts = ["Tüm Haftalar (1–8. Hafta)"] + [f"{i}. Hafta" for i in range(1, 9)]
            sel_w_filter = st.selectbox("Hafta Filtresi:", w_opts, index=0)
        with f_col_d:
            all_tags = ["Tüm Teşhisler"] + sorted(list(set(r["diagnosis_tag"] for r in rec_list)))
            sel_d_filter = st.selectbox("Teşhis Rozeti Filtresi:", all_tags, index=0)

        filtered_recs = rec_list
        if sel_w_filter != "Tüm Haftalar (1–8. Hafta)":
            w_num = int(sel_w_filter.split(".")[0])
            filtered_recs = [r for r in filtered_recs if r["matchday"] == w_num]

        if sel_d_filter != "Tüm Teşhisler":
            filtered_recs = [r for r in filtered_recs if r["diagnosis_tag"] == sel_d_filter]

        table_data = []
        for r in filtered_recs:
            top_s_txt = ", ".join([f"{s[0]}-{s[1]} (%{s[2]*100:.0f})" for s in r["top_5_scores"][:3]])
            table_data.append({
                "Hafta": f"{r['matchday']}. H",
                "Karşılaşma": f"{r['home_team']} - {r['away_team']}",
                "Gerçek Skor": f"{r['actual_home_goals']} - {r['actual_away_goals']}",
                "Gerçek xG": f"{r['actual_hxg']:.2f} - {r['actual_axg']:.2f}",
                "Model Bekl. (λ - μ)": f"{r['pred_lam_h']:.2f} - {r['pred_mu_a']:.2f}",
                "1X2 % (Ev/B/Dep)": f"%{r['p_home']*100:.0f} / %{r['p_draw']*100:.0f} / %{r['p_away']*100:.0f}",
                "En Olası 3 Skor": top_s_txt,
                "Korner (Tah./Ger.)": f"{r['pred_corners_tot']:.1f} / {r['actual_corners_tot']}",
                "Kart (Tah./Ger.)": f"{r['pred_cards_tot']:.1f} / {r['actual_cards_tot']}",
                "Hata Teşhisi": r["diagnosis_tag"],
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=450)
        st.caption(f"Filtrelenen Maç Sayısı: **{len(table_data)}** | Toplam 2026/27 Maçı: **{len(rec_list)}**")

    # Subtab 2: Hata Kaynağı & Varyans Ayrıştırması
    with audit_subtabs[1]:
        st.markdown("#### 🔬 Hata Kaynağı Ayrıştırması: Model Hatası vs. Bitiricilik / Şans Varyansı")
        st.latex(r"\text{Toplam Skor Hatası} = \underbrace{\big((\lambda_H - \mu_A) - (xG_H - xG_A)\big)}_{\text{Model Kurgu Hatası}} + \underbrace{\big((xG_H - xG_A) - (G_H - G_A)\big)}_{\text{Şans / Bitiricilik Varyansı}}")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"##### 🎯 Varyans Katkı Oranları (MSE)")
            v_df = pd.DataFrame([
                {"Hata Türü": "Bitiricilik & Şans Varyansı (Finishing Luck)", "MSE Payı (%)": v_dec["finishing_luck_pct"], "MSE Değeri": v_dec["finishing_luck_mse"]},
                {"Hata Türü": "Model Kurgu Hatası (Model Setup Error)", "MSE Payı (%)": v_dec["model_error_pct"], "MSE Değeri": v_dec["model_error_mse"]},
            ])
            fig_v = px.pie(
                v_df, names="Hata Türü", values="MSE Payı (%)",
                color="Hata Türü",
                color_discrete_map={"Bitiricilik & Şans Varyansı (Finishing Luck)": "#10B981", "Model Kurgu Hatası (Model Setup Error)": "#EF4444"},
                hole=0.45,
            )
            fig_v.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_v, use_container_width=True)

        with col_v2:
            st.markdown(f"##### 🏷️ Teşhis Dağılımı")
            d_df = pd.DataFrame([
                {"Teşhis": k, "Maç Sayısı": v} for k, v in diag["diagnosis_distribution"].items()
            ]).sort_values("Maç Sayısı", ascending=False)
            fig_d = px.bar(
                d_df, x="Maç Sayısı", y="Teşhis", orientation="h",
                color="Maç Sayısı", color_continuous_scale="Viridis",
                text="Maç Sayısı",
            )
            fig_d.update_layout(yaxis={"autorange": "reversed"}, height=320)
            st.plotly_chart(fig_d, use_container_width=True)

        # Hafta Hafta Öğrenme İlerlemesi
        st.markdown("##### 📈 Hafta Hafta Öğrenme ve İsabet İlerlemesi (1–8. Hafta)")
        prog_df = pd.DataFrame(diag["round_progression"])
        prog_df["Hafta"] = prog_df["matchday"].apply(lambda x: f"{x}. Hafta")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_rps = px.line(
                prog_df, x="Hafta", y="rps", markers=True,
                title="Haftalık RPS (Düşük = Daha Başarılı)",
                labels={"rps": "RPS Skoru"}
            )
            fig_rps.update_traces(line_color="#3B82F6", line_width=3)
            st.plotly_chart(fig_rps, use_container_width=True)
        with col_p2:
            fig_hit = px.line(
                prog_df, x="Hafta", y=["hit_rate_1x2_pct", "double_chance_pct"], markers=True,
                title="Haftalık 1X2 & Çifte Şans İsabet Yüzdesi (%)",
                labels={"value": "İsabet %", "variable": "Metrik"}
            )
            st.plotly_chart(fig_hit, use_container_width=True)

    # Subtab 3: Takım Bazlı Sapma Analizi
    with audit_subtabs[2]:
        st.markdown("#### 📊 Takım Bazlı Sistematik Sapma ve İsabet Tablosu")
        st.caption("Pozitif hücum sapması: Model takımı fazla tahmin etti. Negatif hücum sapması: Takım beklentiden daha fazla gol attı.")
        t_df = pd.DataFrame(diag["team_bias_table"])
        st.dataframe(
            t_df[[
                "team_name", "matches", "goals_scored", "goals_conceded",
                "xg_for", "xg_against", "pred_goals_for", "pred_goals_against",
                "attack_bias_vs_goals", "defense_bias_vs_goals", "accuracy_1x2_pct"
            ]].rename(columns={
                "team_name": "Takım",
                "matches": "Maç",
                "goals_scored": "Gol (A)",
                "goals_conceded": "Gol (Y)",
                "xg_for": "xG (A)",
                "xg_against": "xG (Y)",
                "pred_goals_for": "Model Bekl. (A)",
                "pred_goals_against": "Model Bekl. (Y)",
                "attack_bias_vs_goals": "Hücum Sapması",
                "defense_bias_vs_goals": "Savunma Sapması",
                "accuracy_1x2_pct": "1X2 İsabet %",
            }),
            use_container_width=True,
            height=450,
        )

    # Subtab 4: Hakem Etkisi & Kart Puanı Analizi
    with audit_subtabs[3]:
        st.markdown("#### 👨‍⚖️ Hakemlerin Maç Akışı, Kart Puanı ve Tempo Analizi")
        st.caption("2026/27 Süper Lig karşılaşmalarında düdük çalan hakemlerin maç başına beklenen vs. gerçekleşen kart puanları ve xG akışı:")
        ref_table = diag.get("referee_impact_table", [])
        if ref_table:
            ref_df = pd.DataFrame(ref_table)
            st.dataframe(
                ref_df[[
                    "referee", "matches", "avg_actual_bp", "avg_pred_bp", "bp_mae",
                    "avg_actual_cards", "avg_pred_cards", "avg_actual_xg", "avg_pred_xg"
                ]].rename(columns={
                    "referee": "Hakem Adı",
                    "matches": "Yönettiği Maç",
                    "avg_actual_bp": "Gerçek Kart Pts",
                    "avg_pred_bp": "Model Kart Pts",
                    "bp_mae": "Kart Pts Hata (MAE)",
                    "avg_actual_cards": "Gerçek Kart Sayısı",
                    "avg_pred_cards": "Model Kart Sayısı",
                    "avg_actual_xg": "Gerçek xG Toplam",
                    "avg_pred_xg": "Model xG Toplam",
                }),
                use_container_width=True,
                height=450,
            )
        else:
            st.info("Hakem bazlı denetim verisi henüz bulunamadı.")

    # Subtab 5: Favori Sürprizleri
    with audit_subtabs[4]:
        st.markdown("#### ⚡ Yüksek Güvenli Favori Sürprizleri Analizi (Upset Post-Mortem)")
        st.caption("Modelin kazanma ihtimalini %52'den yüksek verdiği halde favorinin puan kaybettiği maçlar:")
        u_list = diag["upset_analysis"]
        if u_list:
            u_df = pd.DataFrame(u_list)[[
                "matchday", "home_team", "away_team", "actual_score",
                "favorite_team", "fav_win_prob_pct", "actual_xg_diff", "post_mortem_cause"
            ]].rename(columns={
                "matchday": "Hafta",
                "home_team": "Ev Sahibi",
                "away_team": "Deplasman",
                "actual_score": "Skor",
                "favorite_team": "Favori",
                "fav_win_prob_pct": "Favori %",
                "actual_xg_diff": "Gerçek xG Farkı",
                "post_mortem_cause": "Sürprizin Temel Nedeni",
            })
            st.dataframe(u_df, use_container_width=True)
        else:
            st.info("Yüksek güvenli favori sürprizi tespit edilmedi.")

    # Subtab 6: Tam Rapor
    with audit_subtabs[5]:
        rep_p = "reports/simulation_audit_2627.md"
        try:
            with open(rep_p, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception:
            st.info("Detaylı simülasyon raporu henüz üretilmedi.")

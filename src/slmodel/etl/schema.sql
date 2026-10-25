-- DuckDB Database Schema for Süper Lig Model

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    country TEXT DEFAULT 'TR'
);

CREATE TABLE IF NOT EXISTS team_aliases (
    source TEXT NOT NULL,
    source_name TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    PRIMARY KEY (source, source_name)
);

CREATE TABLE IF NOT EXISTS stadiums (
    stadium_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    lat DOUBLE,
    lon DOUBLE,
    altitude_m DOUBLE DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    matchday INTEGER,
    kickoff_utc TIMESTAMP NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals SMALLINT,
    away_goals SMALLINT,
    ht_home_goals SMALLINT,
    ht_away_goals SMALLINT,
    referee TEXT,
    stadium_id INTEGER,
    status TEXT NOT NULL CHECK (status IN ('played', 'forfeit', 'void', 'scheduled')),
    crowd TEXT DEFAULT 'full' CHECK (crowd IN ('full', 'limited', 'closed', 'unknown')),
    is_derby BOOLEAN DEFAULT FALSE,
    src_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_match_stats (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    is_home BOOLEAN NOT NULL,
    shots SMALLINT,
    shots_on_target SMALLINT,
    corners SMALLINT,
    fouls SMALLINT,
    yellow SMALLINT,
    red SMALLINT,
    xg DOUBLE,
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id BIGINT PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,          -- '1x2', 'ou', 'ah'
    selection TEXT NOT NULL,       -- '1', 'X', '2', 'over', 'under', 'home', 'away'
    line DOUBLE,                   -- e.g. 2.5 for O/U, -0.5 for AH
    odds DOUBLE NOT NULL,
    taken_at TIMESTAMP,
    kind TEXT NOT NULL CHECK (kind IN ('open', 't48', 't24', 't2', 'closing', 'file_main')),
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_log (
    pred_id BIGINT PRIMARY KEY,
    match_id INTEGER NOT NULL,
    predicted_at TIMESTAMP NOT NULL,
    as_of TIMESTAMP NOT NULL,
    config_hash TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    git_commit TEXT,
    model_version TEXT NOT NULL,
    lam_home DOUBLE NOT NULL,
    lam_away DOUBLE NOT NULL,
    rho DOUBLE NOT NULL,
    score_matrix_json TEXT NOT NULL,
    odds_snapshot_id BIGINT,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    config_json TEXT NOT NULL,
    git_commit TEXT,
    seed INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    name TEXT NOT NULL,
    position TEXT NOT NULL CHECK (position IN ('GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'AM', 'RW', 'LW', 'CF')),
    nationality TEXT,
    age SMALLINT,
    att_rating DOUBLE NOT NULL DEFAULT 0.0,
    def_rating DOUBLE NOT NULL DEFAULT 0.0,
    overall_rating DOUBLE NOT NULL DEFAULT 70.0,
    market_value_eur BIGINT DEFAULT 1000000,
    is_starter BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS team_synergy_units (
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    unit_type TEXT NOT NULL CHECK (unit_type IN ('defense', 'midfield', 'attack')),
    player1_id INTEGER NOT NULL REFERENCES players(player_id),
    player2_id INTEGER NOT NULL REFERENCES players(player_id),
    synergy_score DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (team_id, unit_type, player1_id, player2_id)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id BIGINT PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    is_starter BOOLEAN NOT NULL DEFAULT TRUE,
    minutes_played SMALLINT NOT NULL DEFAULT 90,
    goals SMALLINT NOT NULL DEFAULT 0,
    assists SMALLINT NOT NULL DEFAULT 0,
    shots SMALLINT NOT NULL DEFAULT 0,
    shots_on_target SMALLINT NOT NULL DEFAULT 0,
    key_passes SMALLINT NOT NULL DEFAULT 0,
    tackles SMALLINT NOT NULL DEFAULT 0,
    interceptions SMALLINT NOT NULL DEFAULT 0,
    clearances SMALLINT NOT NULL DEFAULT 0,
    saves SMALLINT NOT NULL DEFAULT 0,
    clean_sheet SMALLINT NOT NULL DEFAULT 0,
    goals_conceded SMALLINT NOT NULL DEFAULT 0,
    fouls_committed SMALLINT NOT NULL DEFAULT 0,
    yellow_cards SMALLINT NOT NULL DEFAULT 0,
    red_cards SMALLINT NOT NULL DEFAULT 0,
    xg DOUBLE NOT NULL DEFAULT 0.0,
    xa DOUBLE NOT NULL DEFAULT 0.0,
    match_rating DOUBLE NOT NULL DEFAULT 6.8
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    season VARCHAR NOT NULL,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    matches_played INTEGER NOT NULL DEFAULT 0,
    matches_started INTEGER NOT NULL DEFAULT 0,
    minutes_played INTEGER NOT NULL DEFAULT 0,
    goals INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    shots INTEGER NOT NULL DEFAULT 0,
    shots_on_target INTEGER NOT NULL DEFAULT 0,
    key_passes INTEGER NOT NULL DEFAULT 0,
    tackles INTEGER NOT NULL DEFAULT 0,
    interceptions INTEGER NOT NULL DEFAULT 0,
    saves INTEGER NOT NULL DEFAULT 0,
    clean_sheets INTEGER NOT NULL DEFAULT 0,
    goals_conceded INTEGER NOT NULL DEFAULT 0,
    yellow_cards INTEGER NOT NULL DEFAULT 0,
    red_cards INTEGER NOT NULL DEFAULT 0,
    xg DOUBLE NOT NULL DEFAULT 0.0,
    xa DOUBLE NOT NULL DEFAULT 0.0,
    avg_match_rating DOUBLE NOT NULL DEFAULT 6.8,
    computed_att_rating DOUBLE NOT NULL,
    computed_def_rating DOUBLE NOT NULL,
    computed_overall_rating DOUBLE NOT NULL,
    PRIMARY KEY (season, player_id)
);

CREATE TABLE IF NOT EXISTS referees (
    referee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    matches_officiated INTEGER DEFAULT 0,
    avg_fouls DOUBLE DEFAULT 25.4,
    avg_yellow DOUBLE DEFAULT 4.55,
    avg_red DOUBLE DEFAULT 0.25,
    avg_booking_pts DOUBLE DEFAULT 51.8,
    whistle_tempo_index DOUBLE DEFAULT 0.0,
    card_strictness_mult DOUBLE DEFAULT 1.0,
    red_card_mult DOUBLE DEFAULT 1.0,
    home_card_bias DOUBLE DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- External fixtures (UEFA Champions League, Europa League, Conference League, Turkish Cup)
CREATE TABLE IF NOT EXISTS external_fixtures (
    ext_id TEXT PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    competition TEXT NOT NULL,
    competition_season_id INTEGER,
    kickoff_utc TIMESTAMP NOT NULL,
    opponent TEXT,
    is_home BOOLEAN,
    status TEXT,
    source TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL
);

-- Match weather from Open-Meteo (archive and forecast)
CREATE TABLE IF NOT EXISTS match_weather (
    match_id INTEGER PRIMARY KEY REFERENCES matches(match_id),
    temp_c DOUBLE,
    precip_mm DOUBLE,
    wind_kmh DOUBLE,
    humidity_pct DOUBLE,
    kind TEXT CHECK (kind IN ('archive', 'forecast')),
    fetched_at TIMESTAMP NOT NULL
);

-- Player availability snapshots (API-Football or derived with strict timestamp)
CREATE TABLE IF NOT EXISTS availability_snapshots (
    snapshot_id BIGINT PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    player_id INTEGER,
    player_name TEXT NOT NULL,
    status TEXT CHECK (status IN ('injured', 'suspended', 'doubtful', 'out', 'available')),
    reason TEXT,
    expected_return DATE,
    source TEXT CHECK (source IN ('api_football', 'derived', 'manual')),
    fetched_at TIMESTAMP NOT NULL
);

-- Manual player availability overrides via dashboard / CLI
CREATE TABLE IF NOT EXISTS availability_overrides (
    override_id BIGINT PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    status TEXT CHECK (status IN ('out', 'doubtful', 'available')),
    match_id INTEGER,
    note TEXT,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);

-- Ingestion audit & quota log
CREATE TABLE IF NOT EXISTS ingest_log (
    log_id BIGINT PRIMARY KEY,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    requested_at TIMESTAMP NOT NULL,
    http_status INTEGER,
    quota_remaining INTEGER,
    detail TEXT
);



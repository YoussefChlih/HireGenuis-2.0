-- ============================================================
-- HireGenius — PostgreSQL Schema
-- Run this once to initialize the database
-- ============================================================

-- Create database (run as superuser)
-- CREATE DATABASE hiregenius;

-- Connect to hiregenius database then run:

CREATE TABLE IF NOT EXISTS match_results (
    id              SERIAL PRIMARY KEY,
    cv_filename     VARCHAR(255) NOT NULL,
    job_title       VARCHAR(500),
    job_description_snippet TEXT,
    score           FLOAT NOT NULL,
    semantic_score  FLOAT NOT NULL,
    keyword_score   FLOAT NOT NULL,
    matched_skills  TEXT,           -- JSON array stored as text
    missing_skills  TEXT,           -- JSON array stored as text
    language        VARCHAR(10),
    gemini_feedback TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_match_results_created_at ON match_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_match_results_score ON match_results(score DESC);

CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    full_name       VARCHAR(120) NOT NULL,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(30) NOT NULL DEFAULT 'candidate',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      VARCHAR(64) NOT NULL UNIQUE,
    expires_at      TIMESTAMP NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_token_hash ON auth_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at ON auth_sessions(expires_at);

CREATE TABLE IF NOT EXISTS jobs (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    description     TEXT NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'draft',
    created_by      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS applications (
    id              SERIAL PRIMARY KEY,
    job_id          INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cv_filename     VARCHAR(255) NOT NULL,
    cv_text         TEXT NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'submitted',
    matching_score  FLOAT NOT NULL DEFAULT 0,
    semantic_score  FLOAT NOT NULL DEFAULT 0,
    keyword_score   FLOAT NOT NULL DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(job_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS application_ai_reports (
    id                  SERIAL PRIMARY KEY,
    application_id      INTEGER NOT NULL UNIQUE REFERENCES applications(id) ON DELETE CASCADE,
    recruiter_summary   TEXT,
    candidate_summary   TEXT,
    strengths           JSONB,
    gaps                JSONB,
    recommendations     JSONB,
    score_breakdown     JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_by ON jobs(created_by);
CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_applications_candidate_id ON applications(candidate_id);
CREATE INDEX IF NOT EXISTS idx_applications_score ON applications(job_id, matching_score DESC);

-- Optional: view for quick analytics
CREATE OR REPLACE VIEW match_stats AS
SELECT
    COUNT(*)                    AS total_analyses,
    AVG(score)                  AS avg_score,
    AVG(semantic_score)         AS avg_semantic,
    AVG(keyword_score)          AS avg_keyword,
    COUNT(*) FILTER (WHERE score >= 75) AS excellent_matches,
    COUNT(*) FILTER (WHERE score >= 55 AND score < 75) AS good_matches,
    COUNT(*) FILTER (WHERE score < 55) AS poor_matches,
    COUNT(*) FILTER (WHERE language = 'fr') AS french_cvs,
    COUNT(*) FILTER (WHERE language = 'en') AS english_cvs
FROM match_results;

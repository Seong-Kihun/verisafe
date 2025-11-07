-- Migration: Enhance Reports Table for Advanced Features (SQLite)
-- Created: 2025-11-05

-- Add new columns to reports table
-- SQLite doesn't support adding multiple columns in one statement
ALTER TABLE reports ADD COLUMN severity VARCHAR(20) DEFAULT 'medium';
ALTER TABLE reports ADD COLUMN reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE reports ADD COLUMN photos TEXT DEFAULT '[]';
ALTER TABLE reports ADD COLUMN is_draft BOOLEAN DEFAULT 0;
ALTER TABLE reports ADD COLUMN accuracy REAL;
ALTER TABLE reports ADD COLUMN conditional_data TEXT DEFAULT '{}';
ALTER TABLE reports ADD COLUMN impact_count INTEGER DEFAULT 0;

-- SQLite doesn't support CREATE INDEX IF NOT EXISTS in older versions
-- But we'll try anyway
CREATE INDEX IF NOT EXISTS idx_reports_is_draft ON reports(is_draft);
CREATE INDEX IF NOT EXISTS idx_reports_reported_at ON reports(reported_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_severity ON reports(severity);

-- Update existing rows to have default values (SQLite doesn't have UPDATE ... WHERE ... IS NULL in ALTER)
UPDATE reports SET
  severity = 'medium',
  reported_at = created_at,
  photos = '[]',
  is_draft = 0,
  conditional_data = '{}',
  impact_count = 0
WHERE id IN (SELECT id FROM reports);

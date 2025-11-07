-- Migration: Enhance Reports Table for Advanced Features
-- Created: 2025-11-05

-- Add new columns to reports table
ALTER TABLE reports
ADD COLUMN IF NOT EXISTS severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high')),
ADD COLUMN IF NOT EXISTS reported_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS photos JSONB DEFAULT '[]'::jsonb,
ADD COLUMN IF NOT EXISTS is_draft BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS accuracy DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS conditional_data JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS impact_count INTEGER DEFAULT 0;

-- Create index for draft reports
CREATE INDEX IF NOT EXISTS idx_reports_is_draft ON reports(is_draft);

-- Create index for reported_at
CREATE INDEX IF NOT EXISTS idx_reports_reported_at ON reports(reported_at DESC);

-- Create index for severity
CREATE INDEX IF NOT EXISTS idx_reports_severity ON reports(severity);

-- Update existing rows to have default values
UPDATE reports SET
  severity = 'medium',
  reported_at = created_at,
  photos = '[]'::jsonb,
  is_draft = FALSE,
  conditional_data = '{}'::jsonb,
  impact_count = 0
WHERE severity IS NULL;

-- Add comment to explain columns
COMMENT ON COLUMN reports.severity IS 'Hazard severity: low, medium, high';
COMMENT ON COLUMN reports.reported_at IS 'When the incident actually occurred (different from created_at)';
COMMENT ON COLUMN reports.photos IS 'Array of photo URLs/paths in JSON format';
COMMENT ON COLUMN reports.is_draft IS 'Whether this is a draft (not submitted yet)';
COMMENT ON COLUMN reports.accuracy IS 'GPS accuracy in meters';
COMMENT ON COLUMN reports.conditional_data IS 'Additional data based on hazard type (JSON)';
COMMENT ON COLUMN reports.impact_count IS 'Number of users who found this report useful';

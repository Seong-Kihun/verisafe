-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Verify PostGIS installation
SELECT PostGIS_version();

-- Create spatial reference system index for performance
CREATE INDEX IF NOT EXISTS idx_spatial_ref_sys_srid ON spatial_ref_sys(srid);

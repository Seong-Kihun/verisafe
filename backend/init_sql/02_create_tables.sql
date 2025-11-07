-- VeriSafe Database Schema with PostGIS
-- Created: 2025-11-05

-- ==========================================
-- 1. Users Table
-- ==========================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'mapper')),
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ==========================================
-- 2. Hazards Table (PostGIS POINT)
-- ==========================================
CREATE TABLE IF NOT EXISTS hazards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hazard_type VARCHAR(50) NOT NULL,
    risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    
    -- Coordinates
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    radius DOUBLE PRECISION NOT NULL,
    
    -- PostGIS POINT
    geometry GEOMETRY(POINT, 4326) NOT NULL,
    
    -- Metadata
    source VARCHAR(50),
    description TEXT,
    start_date TIMESTAMP NOT NULL DEFAULT NOW(),
    end_date TIMESTAMP,
    verified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hazards_type ON hazards(hazard_type);
CREATE INDEX IF NOT EXISTS idx_hazards_geometry ON hazards USING GIST(geometry);


-- ==========================================
-- 3. Reports Table
-- ==========================================
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    hazard_type VARCHAR(50) NOT NULL,
    description TEXT,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geometry GEOMETRY(POINT, 4326),
    image_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'pending',
    verified_by UUID REFERENCES users(id) ON DELETE SET NULL,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_geometry ON reports USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);

-- ==========================================
-- 4. Roads Table  
-- ==========================================
CREATE TABLE IF NOT EXISTS roads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    osm_id BIGINT UNIQUE,
    name VARCHAR(200),
    geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
    road_type VARCHAR(50),
    length_km DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roads_geometry ON roads USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_roads_osm_id ON roads(osm_id);

-- ==========================================
-- 5. Landmarks Table
-- ==========================================
CREATE TABLE IF NOT EXISTS landmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geometry GEOMETRY(POINT, 4326) NOT NULL,
    description TEXT,
    address VARCHAR(500),
    osm_id BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_landmarks_geometry ON landmarks USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_landmarks_category ON landmarks(category);


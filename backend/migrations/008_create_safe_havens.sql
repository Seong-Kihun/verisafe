-- Safe havens table for emergency shelters, embassies, hospitals, etc.
CREATE TABLE IF NOT EXISTS safe_havens (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  category VARCHAR(50) NOT NULL, -- embassy, hospital, un, police, hotel, shelter
  latitude DECIMAL(10, 7) NOT NULL,
  longitude DECIMAL(10, 7) NOT NULL,
  address TEXT,
  phone VARCHAR(50),
  hours TEXT, -- Operating hours (e.g., "24/7" or "Mon-Fri 9-17")
  capacity INTEGER, -- Max people for shelters
  verified BOOLEAN DEFAULT FALSE, -- Verified by admins
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for geospatial queries
CREATE INDEX idx_safe_havens_location ON safe_havens(latitude, longitude);
CREATE INDEX idx_safe_havens_category ON safe_havens(category);

-- Sample data for Juba, South Sudan
INSERT INTO safe_havens (name, category, latitude, longitude, address, phone, hours, verified, notes) VALUES
-- Embassies
('미국 대사관 (US Embassy)', 'embassy', 4.8594, 31.5713, 'Kololo Road, Juba', '+211-912-105-188', 'Mon-Fri 8:00-17:00', true, 'Emergency services available 24/7'),
('영국 대사관 (UK Embassy)', 'embassy', 4.8520, 31.5820, 'Thong Ping, EU Compound', '+211-912-105-111', 'Mon-Thu 7:30-15:30', true, 'Consular assistance'),
('케냐 대사관 (Kenya Embassy)', 'embassy', 4.8601, 31.5890, 'Hai Referendum, Juba', '+211-955-061-000', 'Mon-Fri 8:00-16:30', true, null),

-- Hospitals
('Juba Teaching Hospital', 'hospital', 4.8512, 31.5580, 'Airport Road, Juba', '+211-928-888-888', '24/7', true, 'Main public hospital, emergency services'),
('International Hospital Kampala (IHK) Juba', 'hospital', 4.8650, 31.5920, 'Kololo, Juba', '+211-922-000-000', '24/7', true, 'Private hospital, high quality care'),
('Al-Sabah Children''s Hospital', 'hospital', 4.8490, 31.5600, 'Gudele, Juba', '+211-920-000-000', '24/7', true, 'Pediatric emergency care'),

-- UN Facilities
('UNMISS Juba Base', 'un', 4.8780, 31.6010, 'Juba International Airport', '+211-912-177-777', '24/7', true, 'Protection of Civilians site'),
('UNMISS Tomping Base', 'un', 4.8420, 31.5730, 'Tomping, Juba', '+211-912-177-888', '24/7', true, 'POC site, civilian protection'),

-- Police Stations
('Juba Central Police Station', 'police', 4.8530, 31.5800, 'Juba Town Center', '+211-955-000-777', '24/7', true, 'Main police station'),
('Gudele Police Station', 'police', 4.8450, 31.5550, 'Gudele Block 1', '+211-955-000-888', '24/7', true, null),

-- Safe Hotels
('Juba Grand Hotel', 'hotel', 4.8580, 31.5850, 'Hai Referendum, Juba', '+211-928-000-111', '24/7', true, 'Secure compound, 24/7 security'),
('Acacia Village Hotel', 'hotel', 4.8620, 31.5900, 'Kololo Road, Juba', '+211-928-000-222', '24/7', true, 'High security, expat-friendly'),

-- Emergency Shelters
('Red Cross Emergency Shelter', 'shelter', 4.8400, 31.5700, 'Munuki, Juba', '+211-920-111-000', '24/7', true, 'Capacity: 500 people'),
('UN Emergency Shelter - Tomping', 'shelter', 4.8410, 31.5720, 'Tomping, Juba', '+211-912-177-999', '24/7', true, 'Capacity: 1000 people');

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_safe_havens_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER safe_havens_updated_at
BEFORE UPDATE ON safe_havens
FOR EACH ROW
EXECUTE FUNCTION update_safe_havens_updated_at();

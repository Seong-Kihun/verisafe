/**
 * VeriSafe Web Portal Types
 */

// User & Auth Types
export interface User {
  id: string;
  username: string;
  email: string;
  role: 'user' | 'mapper' | 'admin';
  verified: boolean;
  created_at: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}

// Feature Types (매퍼가 생성하는 지리 정보)
export type FeatureType =
  | 'building'
  | 'road'
  | 'bridge'
  | 'hospital'
  | 'school'
  | 'police'
  | 'fire_station'
  | 'safe_haven'
  | 'shelter'
  | 'landmark';

export type GeometryType = 'point' | 'line' | 'polygon';

export type ReviewStatus = 'pending' | 'under_review' | 'approved' | 'rejected';

export type DetectionSource = 'mapper_created' | 'ai' | 'satellite_ai' | 'hybrid' | 'yolo_v8' | 'microsoft_buildings';

export interface GeoJSONGeometry {
  type: 'Point' | 'LineString' | 'Polygon';
  coordinates: number[] | number[][] | number[][][];
}

export interface DetectedFeature {
  id: string;
  feature_type: FeatureType;
  latitude: number;
  longitude: number;
  geometry_type: GeometryType;
  geometry_data: GeoJSONGeometry;
  confidence: number;
  detection_source: DetectionSource;
  review_status: ReviewStatus;
  created_by_user_id?: string;
  reviewed_by_user_id?: string;
  review_comment?: string;
  reviewed_at?: string;
  verified: boolean;
  verification_count: number;
  name?: string;
  description?: string;
  properties?: Record<string, any>;
  satellite_image_url?: string;
  user_photo_url?: string;
  created_at: string;
  updated_at: string;
  last_verified_at?: string;
}

// Mapper API Types
export interface MapperFeatureCreate {
  feature_type: FeatureType;
  geometry_type: GeometryType;
  geometry_data: GeoJSONGeometry;
  name?: string;
  description?: string;
  properties?: Record<string, any>;
  user_photo_url?: string;
}

export interface MapperFeatureUpdate {
  name?: string;
  description?: string;
  properties?: Record<string, any>;
  geometry_data?: GeoJSONGeometry;
}

export interface MapperContributionSummary {
  total_submissions: number;
  pending: number;
  under_review: number;
  approved: number;
  rejected: number;
  by_type: Record<string, number>;
}

// Reviewer API Types
export interface ReviewAction {
  comment?: string;
}

export interface ReviewerDashboardStats {
  pending_count: number;
  under_review_count: number;
  approved_today: number;
  rejected_today: number;
  pending_by_source: {
    ai: number;
    mapper: number;
  };
}

// API Response Types
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ApiError {
  detail: string;
  userMessage?: string;
}

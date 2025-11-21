/**
 * Reviewer API
 * 검수자가 지리 정보를 검토/승인/거부하는 API
 */

import apiClient from './client';
import type {
  DetectedFeature,
  ReviewAction,
  ReviewerDashboardStats,
  PaginatedResponse,
} from '@/types';

export const reviewerAPI = {
  /**
   * 검수 대기 목록 조회
   */
  getPendingReviews: async (params?: {
    source?: 'ai' | 'mapper' | 'all';
    feature_type?: string;
    page?: number;
    page_size?: number;
  }): Promise<PaginatedResponse<DetectedFeature>> => {
    const response = await apiClient.get<PaginatedResponse<DetectedFeature>>(
      '/api/review/pending',
      { params }
    );
    return response.data;
  },

  /**
   * 검수 시작 (under_review로 변경)
   */
  startReview: async (featureId: string): Promise<DetectedFeature> => {
    const response = await apiClient.post<DetectedFeature>(
      `/api/review/${featureId}/start-review`
    );
    return response.data;
  },

  /**
   * 승인
   */
  approveFeature: async (
    featureId: string,
    action?: ReviewAction
  ): Promise<DetectedFeature> => {
    const response = await apiClient.post<DetectedFeature>(
      `/api/review/${featureId}/approve`,
      action || {}
    );
    return response.data;
  },

  /**
   * 거부
   */
  rejectFeature: async (
    featureId: string,
    action: ReviewAction
  ): Promise<DetectedFeature> => {
    const response = await apiClient.post<DetectedFeature>(
      `/api/review/${featureId}/reject`,
      action
    );
    return response.data;
  },

  /**
   * 지도 영역 내 feature 조회 (검수용)
   */
  getFeaturesInArea: async (params: {
    min_lat: number;
    min_lng: number;
    max_lat: number;
    max_lng: number;
    status?: 'pending' | 'under_review' | 'approved' | 'rejected';
  }): Promise<DetectedFeature[]> => {
    const response = await apiClient.get<DetectedFeature[]>('/api/review/area', { params });
    return response.data;
  },

  /**
   * 검수자 대시보드 통계
   */
  getDashboardStats: async (): Promise<ReviewerDashboardStats> => {
    const response = await apiClient.get<ReviewerDashboardStats>('/api/review/dashboard');
    return response.data;
  },
};

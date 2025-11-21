/**
 * Mapper API
 * 매퍼가 지리 정보를 생성/수정/삭제하는 API
 */

import apiClient from './client';
import type {
  DetectedFeature,
  MapperFeatureCreate,
  MapperFeatureUpdate,
  MapperContributionSummary,
  PaginatedResponse,
} from '@/types';

export const mapperAPI = {
  /**
   * 새 지리 정보 생성
   */
  createFeature: async (data: MapperFeatureCreate): Promise<DetectedFeature> => {
    const response = await apiClient.post<DetectedFeature>('/api/mapper/features', data);
    return response.data;
  },

  /**
   * 지리 정보 수정
   */
  updateFeature: async (
    featureId: string,
    data: MapperFeatureUpdate
  ): Promise<DetectedFeature> => {
    const response = await apiClient.put<DetectedFeature>(
      `/api/mapper/features/${featureId}`,
      data
    );
    return response.data;
  },

  /**
   * 지리 정보 삭제 (pending 상태만 가능)
   */
  deleteFeature: async (featureId: string): Promise<{ message: string }> => {
    const response = await apiClient.delete<{ message: string }>(
      `/api/mapper/features/${featureId}`
    );
    return response.data;
  },

  /**
   * 내 기여 목록 조회
   */
  getMyContributions: async (params?: {
    status?: 'pending' | 'under_review' | 'approved' | 'rejected';
    page?: number;
    page_size?: number;
  }): Promise<PaginatedResponse<DetectedFeature>> => {
    const response = await apiClient.get<PaginatedResponse<DetectedFeature>>(
      '/api/mapper/my-contributions',
      { params }
    );
    return response.data;
  },

  /**
   * 내 기여 통계
   */
  getContributionSummary: async (): Promise<MapperContributionSummary> => {
    const response = await apiClient.get<MapperContributionSummary>('/api/mapper/my-summary');
    return response.data;
  },
};

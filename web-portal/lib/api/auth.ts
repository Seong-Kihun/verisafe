/**
 * Authentication API
 */

import apiClient from './client';
import type { LoginCredentials, AuthToken, User } from '@/types';

export const authAPI = {
  /**
   * 로그인
   */
  login: async (credentials: LoginCredentials): Promise<AuthToken> => {
    const formData = new URLSearchParams();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);

    const response = await apiClient.post<AuthToken>('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    return response.data;
  },

  /**
   * 회원가입
   */
  register: async (userData: {
    username: string;
    email: string;
    password: string;
  }): Promise<User> => {
    const response = await apiClient.post<User>('/api/auth/register', userData);
    return response.data;
  },

  /**
   * 현재 사용자 정보 가져오기 (구현 필요시)
   */
  getCurrentUser: async (): Promise<User> => {
    // 백엔드에 /api/auth/me 엔드포인트가 있다면 사용
    const response = await apiClient.get<User>('/api/auth/me');
    return response.data;
  },
};

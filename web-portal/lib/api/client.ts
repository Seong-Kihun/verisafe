/**
 * API Client
 * 백엔드 FastAPI와 통신하는 Axios 클라이언트
 */

import axios, { AxiosError } from 'axios';
import type { ApiError } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터 - JWT 토큰 추가
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('verisafe_auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    if (process.env.NODE_ENV === 'development') {
      console.log('[API] 요청:', config.method?.toUpperCase(), config.url);
    }

    return config;
  },
  (error) => {
    console.error('[API] 요청 오류:', error);
    return Promise.reject(error);
  }
);

// 응답 인터셉터 - 에러 처리 및 재시도
apiClient.interceptors.response.use(
  (response) => {
    if (process.env.NODE_ENV === 'development') {
      console.log('[API] 응답:', response.config.url, response.status);
    }
    return response;
  },
  async (error: AxiosError<ApiError>) => {
    const originalRequest = error.config;

    // 네트워크 오류
    if (!error.response) {
      console.error('[API] 네트워크 오류 - 서버에 연결할 수 없습니다');
      error.message = '서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.';
      return Promise.reject(error);
    }

    // 인증 오류 (401) - 로그인 페이지로 리다이렉트
    if (error.response.status === 401) {
      localStorage.removeItem('verisafe_auth_token');
      localStorage.removeItem('verisafe_user');

      if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
        window.location.href = '/login';
      }
    }

    // 서버 오류 (5xx) - 재시도 로직
    if (error.response.status >= 500 && originalRequest) {
      const retryCount = (originalRequest as any)._retryCount || 0;

      if (retryCount < 2) {
        (originalRequest as any)._retryCount = retryCount + 1;
        console.log(`[API] 서버 오류 - 재시도 중 (${retryCount + 1}/2)`);

        await new Promise((resolve) => setTimeout(resolve, 1000));
        return apiClient(originalRequest);
      }

      console.error('[API] 서버 오류 - 재시도 실패');
      error.message = '서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.';
    }

    // 사용자 친화적 에러 메시지
    if (error.response.data?.detail) {
      error.message = error.response.data.detail;
    }

    return Promise.reject(error);
  }
);

export default apiClient;
export { API_BASE_URL };

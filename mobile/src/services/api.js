/** API 서비스 */
import axios from 'axios';
import { Platform } from 'react-native';
import Constants from 'expo-constants';

// API 기본 URL (환경별 설정)
const getApiBaseUrl = () => {
  // 환경 변수에서 API URL 가져오기 (app.json의 extra 설정)
  const envApiUrl = Constants.expoConfig?.extra?.EXPO_PUBLIC_API_URL || process.env.EXPO_PUBLIC_API_URL;

  if (envApiUrl) {
    return envApiUrl;
  }

  // 프로덕션 환경
  if (!__DEV__) {
    return 'https://api.verisafe.com';
  }

  // 개발 환경 - 웹은 localhost, 모바일은 네트워크 IP 사용
  if (Platform.OS === 'web') {
    return 'http://localhost:8000';
  }

  // 모바일 개발 환경 - 기본값 사용 (테스트/시연용)
  // 실제 배포 시: EXPO_PUBLIC_API_URL 환경 변수 설정 또는 아래 IP를 PC IP로 변경
  return 'http://172.20.10.3:8000';
};

const API_BASE_URL = getApiBaseUrl();

// 개발 환경에서만 로깅
if (__DEV__) {
  console.log('[API] API_BASE_URL:', API_BASE_URL);
  console.log('[API] Platform.OS:', Platform.OS);
}

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30초로 증가 (외부 API 호출 고려)
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터 - 개발 환경에서만 로깅
api.interceptors.request.use(
  (config) => {
    if (__DEV__) {
      console.log('[API] 요청:', config.method?.toUpperCase(), config.url);
      if (config.params) console.log('[API] 파라미터:', config.params);
    }
    return config;
  },
  (error) => {
    if (__DEV__) {
      console.error('[API] 요청 오류:', error);
    }
    return Promise.reject(error);
  }
);

// 응답 인터셉터 - 에러 처리 및 재시도
api.interceptors.response.use(
  (response) => {
    if (__DEV__) {
      console.log('[API] 응답:', response.config.url, response.status);
    }
    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    // 에러 로깅 (프로덕션에서도 중요)
    console.error('[API] 요청 실패:', error.config?.url);

    // 네트워크 오류인 경우
    if (!error.response) {
      console.error('[API] 네트워크 오류 - 서버에 연결할 수 없습니다');
      error.userMessage = '서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.';
      return Promise.reject(error);
    }

    // 타임아웃 오류
    if (error.code === 'ECONNABORTED') {
      console.error('[API] 타임아웃 - 서버 응답 시간 초과');
      error.userMessage = '서버 응답 시간이 초과되었습니다. 다시 시도해주세요.';
      return Promise.reject(error);
    }

    // 서버 오류 (5xx) - 재시도 로직
    if (error.response.status >= 500 && error.response.status < 600) {
      // 재시도 횟수 확인 (최대 2회)
      originalRequest._retryCount = originalRequest._retryCount || 0;

      if (originalRequest._retryCount < 2) {
        originalRequest._retryCount += 1;
        console.log(`[API] 서버 오류 - 재시도 중 (${originalRequest._retryCount}/2)`);

        // 1초 대기 후 재시도
        await new Promise(resolve => setTimeout(resolve, 1000));
        return api(originalRequest);
      }

      console.error('[API] 서버 오류 - 재시도 실패');
      error.userMessage = '서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.';
    }

    // 클라이언트 오류 (4xx)
    if (error.response.status >= 400 && error.response.status < 500) {
      const detail = error.response?.data?.detail || error.message;
      console.error('[API] 클라이언트 오류:', detail);

      // 사용자 친화적 메시지
      if (error.response.status === 401) {
        error.userMessage = '인증이 만료되었습니다. 다시 로그인해주세요.';
      } else if (error.response.status === 404) {
        error.userMessage = '요청한 정보를 찾을 수 없습니다.';
      } else if (error.response.status === 429) {
        error.userMessage = '요청이 너무 많습니다. 잠시 후 다시 시도해주세요.';
      } else {
        error.userMessage = detail;
      }
    }

    return Promise.reject(error);
  }
);

// JWT 인증 토큰 인터셉터 (필요시 구현)
// import AsyncStorage from '@react-native-async-storage/async-storage';
//
// api.interceptors.request.use(async (config) => {
//   try {
//     const token = await AsyncStorage.getItem('@verisafe:auth_token');
//     if (token) {
//       config.headers.Authorization = `Bearer ${token}`;
//     }
//   } catch (error) {
//     console.error('[API] Failed to get auth token:', error);
//   }
//   return config;
// });

// API 함수들
export const mapAPI = {
  getBounds: (minLat, minLng, maxLat, maxLng, country = null) =>
    api.get('/api/map/bounds', {
      params: { min_lat: minLat, min_lng: minLng, max_lat: maxLat, max_lng: maxLng, country }
    }),
  getLandmarks: (lat, lng, radius) =>
    api.get('/api/map/landmarks', { params: { lat, lng, radius } }),
  getHazards: (lat, lng, radius, country = null) =>
    api.get('/api/map/hazards', { params: { lat, lng, radius, country } }),
  autocomplete: (q) =>
    api.get('/api/map/search/autocomplete', { params: { q } }),
  getPlaceDetail: (id) =>
    api.get('/api/map/places/detail', { params: { id } }),
  reverseGeocode: (lat, lng) =>
    api.get('/api/map/places/reverse', { params: { lat, lng } }),
  getCountries: () =>
    api.get('/api/map/countries'),
};

export const reportAPI = {
  create: (reportData) => api.post('/api/reports/create', reportData),
  list: (lat, lng, radius) =>
    api.get('/api/reports/list', { params: { lat, lng, radius } }),
  getNearby: ({ latitude, longitude, radius, hours }) =>
    api.get('/api/reports/list', {
      params: {
        lat: latitude,
        lng: longitude,
        radius,
        hours
      }
    }),
  verify: (reportId) => api.post(`/api/reports/${reportId}/verify`),
};

export const authAPI = {
  login: (username, password) =>
    api.post('/api/auth/login', new URLSearchParams({ username, password }).toString(), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    }),
  register: (userData) => api.post('/api/auth/register', userData),
};

export const routeAPI = {
  calculate: (start, end, preference, transportation_mode = 'car', excluded_hazard_types = []) =>
    api.post('/api/route/calculate', {
      start,
      end,
      preference,
      transportation_mode,
      excluded_hazard_types
    }),
  getRouteHazards: (routeId, polyline) =>
    api.get(`/api/route/${routeId}/hazards`, {
      params: { polyline: JSON.stringify(polyline) }
    }),
};

export const emergencyAPI = {
  /**
   * Trigger emergency SOS
   * @param {Object} sosData - {user_id, latitude, longitude, message}
   */
  triggerSOS: async (sosData) => {
    const response = await api.post('/api/emergency/sos', sosData);
    return response.data;
  },

  /**
   * Cancel SOS
   * @param {number} sosId - SOS event ID
   * @param {number} userId - User ID
   */
  cancelSOS: async (sosId, userId) => {
    const response = await api.post(`/api/emergency/sos/${sosId}/cancel`, {
      user_id: userId
    });
    return response.data;
  },

  /**
   * Get user SOS history
   * @param {number} userId - User ID
   * @param {string} status - Filter by status (active, cancelled, resolved) - optional
   * @param {number} limit - Maximum number of records (default: 50)
   */
  getUserSOSHistory: async (userId, status = null, limit = 50) => {
    const params = { limit };
    if (status) params.status = status;

    const response = await api.get(`/api/emergency/sos/user/${userId}`, { params });
    return response.data;
  },

  /**
   * Get SOS event detail
   * @param {number} sosId - SOS event ID
   */
  getSOSDetail: async (sosId) => {
    const response = await api.get(`/api/emergency/sos/${sosId}`);
    return response.data;
  },
};

/**
 * Safety Check-in API
 */
export const safetyCheckinAPI = {
  /**
   * Register safety check-in
   * @param {Object} checkinData - {user_id, route_id?, estimated_arrival_time, destination_lat?, destination_lon?}
   */
  register: async (checkinData) => {
    const response = await api.post('/api/safety-checkin/register', checkinData);
    return response.data;
  },

  /**
   * Confirm safe arrival
   * @param {number} checkinId - Check-in ID
   * @param {number} userId - User ID
   */
  confirm: async (checkinId, userId) => {
    const response = await api.post(`/api/safety-checkin/${checkinId}/confirm`, {
      user_id: userId
    });
    return response.data;
  },
};

export default api;

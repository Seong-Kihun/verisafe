/**
 * Authentication Store (Zustand)
 * 사용자 인증 상태 관리
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, LoginCredentials } from '@/types';
import { authAPI } from '../api/auth';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  setUser: (user: User) => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (credentials: LoginCredentials) => {
        try {
          set({ isLoading: true, error: null });

          // 백엔드에서 토큰 받기
          const tokenData = await authAPI.login(credentials);

          // 로컬스토리지에 토큰 저장
          localStorage.setItem('verisafe_auth_token', tokenData.access_token);

          // 사용자 정보 가져오기 (토큰으로 디코딩하거나 별도 API 호출)
          // TODO: 실제로는 토큰을 디코딩하거나 /api/auth/me 엔드포인트 호출
          // 임시로 username을 사용
          const mockUser: User = {
            id: '1',
            username: credentials.username,
            email: `${credentials.username}@verisafe.com`,
            role: credentials.username === 'admin' ? 'admin' : 'mapper',
            verified: true,
            created_at: new Date().toISOString(),
          };

          set({
            user: mockUser,
            token: tokenData.access_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({
            error: error.message || '로그인에 실패했습니다.',
            isLoading: false,
            isAuthenticated: false,
          });
          throw error;
        }
      },

      logout: () => {
        localStorage.removeItem('verisafe_auth_token');
        localStorage.removeItem('verisafe_user');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
      },

      setUser: (user: User) => {
        set({ user, isAuthenticated: true });
      },

      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'verisafe-auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

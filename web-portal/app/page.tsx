'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useAuthStore } from '@/lib/stores/auth-store';

export default function Home() {
  const router = useRouter();
  const { user, isAuthenticated } = useAuthStore();

  useEffect(() => {
    // 인증된 사용자는 역할에 따라 리다이렉트
    if (isAuthenticated && user) {
      if (user.role === 'mapper' || user.role === 'admin') {
        router.push('/mapper');
      } else if (user.role === 'admin') {
        router.push('/reviewer');
      } else {
        router.push('/login');
      }
    } else {
      // 미인증 사용자는 로그인 페이지로
      router.push('/login');
    }
  }, [isAuthenticated, user, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
        <p className="mt-4 text-gray-600">로딩 중...</p>
      </div>
    </div>
  );
}

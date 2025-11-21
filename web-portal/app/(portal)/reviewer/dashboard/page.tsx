'use client';

import { useQuery } from '@tanstack/react-query';
import Header from '@/components/layout/Header';
import { reviewerAPI } from '@/lib/api/reviewer';

export default function ReviewerDashboardPage() {
  // 대시보드 통계 조회
  const { data: stats, isLoading } = useQuery({
    queryKey: ['reviewer', 'dashboard'],
    queryFn: () => reviewerAPI.getDashboardStats(),
  });

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="검수 대시보드" subtitle="검수 통계와 현황을 확인하세요" />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-gray-600">로딩 중...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="검수 대시보드" subtitle="검수 통계와 현황을 확인하세요" />

      <div className="flex-1 overflow-auto p-6">
        {/* 통계 카드 */}
        {stats && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-yellow-500">
                <p className="text-sm text-gray-600 mb-2">검수 대기</p>
                <p className="text-4xl font-bold text-gray-900">{stats.pending_count}</p>
              </div>

              <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
                <p className="text-sm text-gray-600 mb-2">검수 중</p>
                <p className="text-4xl font-bold text-gray-900">{stats.under_review_count}</p>
              </div>

              <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-500">
                <p className="text-sm text-gray-600 mb-2">오늘 승인</p>
                <p className="text-4xl font-bold text-green-600">{stats.approved_today}</p>
              </div>

              <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-red-500">
                <p className="text-sm text-gray-600 mb-2">오늘 거부</p>
                <p className="text-4xl font-bold text-red-600">{stats.rejected_today}</p>
              </div>
            </div>

            {/* 출처별 대기 현황 */}
            <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
              <h2 className="text-xl font-bold mb-4">출처별 검수 대기 현황</h2>
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <p className="text-sm text-blue-700 mb-1">AI 탐지</p>
                  <p className="text-3xl font-bold text-blue-900">
                    {stats.pending_by_source.ai}
                  </p>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                  <p className="text-sm text-purple-700 mb-1">매퍼 제출</p>
                  <p className="text-3xl font-bold text-purple-900">
                    {stats.pending_by_source.mapper}
                  </p>
                </div>
              </div>
            </div>

            {/* 작업 효율성 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4">오늘의 작업 현황</h2>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600">승인 비율</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stats.approved_today + stats.rejected_today > 0
                        ? (
                            (stats.approved_today /
                              (stats.approved_today + stats.rejected_today)) *
                            100
                          ).toFixed(1)
                        : 0}
                      %
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-600 h-2 rounded-full"
                      style={{
                        width: `${
                          stats.approved_today + stats.rejected_today > 0
                            ? (stats.approved_today /
                                (stats.approved_today + stats.rejected_today)) *
                              100
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600">처리 건수</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stats.approved_today + stats.rejected_today} 건
                    </span>
                  </div>
                  <div className="flex gap-2 text-xs text-gray-600">
                    <span>승인: {stats.approved_today}건</span>
                    <span>거부: {stats.rejected_today}건</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

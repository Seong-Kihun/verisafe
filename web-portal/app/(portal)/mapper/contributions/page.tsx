'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Header from '@/components/layout/Header';
import { mapperAPI } from '@/lib/api/mapper';
import type { DetectedFeature, ReviewStatus } from '@/types';

const statusLabels: Record<ReviewStatus, string> = {
  pending: '검수 대기',
  under_review: '검수 중',
  approved: '승인됨',
  rejected: '거부됨',
};

const statusColors: Record<ReviewStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  under_review: 'bg-blue-100 text-blue-800',
  approved: 'bg-green-100 text-green-800',
  rejected: 'bg-red-100 text-red-800',
};

export default function ContributionsPage() {
  const [selectedStatus, setSelectedStatus] = useState<ReviewStatus | 'all'>('all');

  // 기여 통계 조회
  const { data: summary } = useQuery({
    queryKey: ['mapper', 'summary'],
    queryFn: () => mapperAPI.getContributionSummary(),
  });

  // 기여 목록 조회
  const { data: contributions, isLoading } = useQuery({
    queryKey: ['mapper', 'contributions', selectedStatus],
    queryFn: () =>
      mapperAPI.getMyContributions({
        status: selectedStatus === 'all' ? undefined : selectedStatus,
        page: 1,
        page_size: 50,
      }),
  });

  return (
    <div className="flex flex-col h-full">
      <Header title="내 기여" subtitle="제출한 지리 정보와 검수 상태를 확인하세요" />

      <div className="flex-1 overflow-auto p-6">
        {/* 통계 카드 */}
        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
              <p className="text-sm text-gray-600">총 제출</p>
              <p className="text-2xl font-bold text-gray-900">{summary.total_submissions}</p>
            </div>
            <div className="bg-yellow-50 rounded-lg shadow p-4 border border-yellow-200">
              <p className="text-sm text-yellow-700">검수 대기</p>
              <p className="text-2xl font-bold text-yellow-900">{summary.pending}</p>
            </div>
            <div className="bg-blue-50 rounded-lg shadow p-4 border border-blue-200">
              <p className="text-sm text-blue-700">검수 중</p>
              <p className="text-2xl font-bold text-blue-900">{summary.under_review}</p>
            </div>
            <div className="bg-green-50 rounded-lg shadow p-4 border border-green-200">
              <p className="text-sm text-green-700">승인됨</p>
              <p className="text-2xl font-bold text-green-900">{summary.approved}</p>
            </div>
            <div className="bg-red-50 rounded-lg shadow p-4 border border-red-200">
              <p className="text-sm text-red-700">거부됨</p>
              <p className="text-2xl font-bold text-red-900">{summary.rejected}</p>
            </div>
          </div>
        )}

        {/* 필터 */}
        <div className="mb-4">
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedStatus('all')}
              className={`px-4 py-2 rounded-md ${
                selectedStatus === 'all'
                  ? 'bg-primary text-white'
                  : 'bg-white text-gray-700 border border-gray-300'
              }`}
            >
              전체
            </button>
            {(['pending', 'under_review', 'approved', 'rejected'] as const).map((status) => (
              <button
                key={status}
                onClick={() => setSelectedStatus(status)}
                className={`px-4 py-2 rounded-md ${
                  selectedStatus === status
                    ? 'bg-primary text-white'
                    : 'bg-white text-gray-700 border border-gray-300'
                }`}
              >
                {statusLabels[status]}
              </button>
            ))}
          </div>
        </div>

        {/* 기여 목록 */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-gray-600">로딩 중...</p>
          </div>
        ) : contributions && contributions.items.length > 0 ? (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    객체 유형
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    이름
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    상태
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    제출일
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    검수 코멘트
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {contributions.items.map((item: DetectedFeature) => (
                  <tr key={item.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {item.feature_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {item.name || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded-full ${
                          statusColors[item.review_status]
                        }`}
                      >
                        {statusLabels[item.review_status]}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {new Date(item.created_at).toLocaleDateString('ko-KR')}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {item.review_comment || '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <p className="text-gray-500">아직 제출한 기여가 없습니다.</p>
            <p className="text-sm text-gray-400 mt-2">지도 편집 메뉴에서 새로운 정보를 추가하세요.</p>
          </div>
        )}
      </div>
    </div>
  );
}

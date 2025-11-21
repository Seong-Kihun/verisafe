'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import Header from '@/components/layout/Header';
import { reviewerAPI } from '@/lib/api/reviewer';
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

export default function ReviewerQueuePage() {
  const queryClient = useQueryClient();
  const [selectedFeature, setSelectedFeature] = useState<DetectedFeature | null>(null);
  const [filterSource, setFilterSource] = useState<'ai' | 'mapper' | 'all'>('all');
  const [reviewComment, setReviewComment] = useState('');

  // 검수 대기열 조회
  const { data: pendingReviews, isLoading } = useQuery({
    queryKey: ['reviewer', 'pending', filterSource],
    queryFn: () =>
      reviewerAPI.getPendingReviews({
        source: filterSource,
        page: 1,
        page_size: 50,
      }),
  });

  // 검수 시작
  const startReviewMutation = useMutation({
    mutationFn: (featureId: string) => reviewerAPI.startReview(featureId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['reviewer', 'pending'] });
      setSelectedFeature(data);
    },
  });

  // 승인
  const approveMutation = useMutation({
    mutationFn: ({ featureId, comment }: { featureId: string; comment?: string }) =>
      reviewerAPI.approveFeature(featureId, comment ? { comment } : undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reviewer'] });
      alert('승인되었습니다.');
      setSelectedFeature(null);
      setReviewComment('');
    },
    onError: (error: any) => {
      alert(`승인 실패: ${error.message}`);
    },
  });

  // 거부
  const rejectMutation = useMutation({
    mutationFn: ({ featureId, comment }: { featureId: string; comment: string }) =>
      reviewerAPI.rejectFeature(featureId, { comment }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reviewer'] });
      alert('거부되었습니다.');
      setSelectedFeature(null);
      setReviewComment('');
    },
    onError: (error: any) => {
      alert(`거부 실패: ${error.message}`);
    },
  });

  const handleStartReview = (feature: DetectedFeature) => {
    if (feature.review_status === 'pending') {
      startReviewMutation.mutate(feature.id);
    } else {
      setSelectedFeature(feature);
    }
  };

  const handleApprove = () => {
    if (!selectedFeature) return;
    approveMutation.mutate({
      featureId: selectedFeature.id,
      comment: reviewComment || undefined,
    });
  };

  const handleReject = () => {
    if (!selectedFeature) return;
    if (!reviewComment.trim()) {
      alert('거부 사유를 입력해주세요.');
      return;
    }
    rejectMutation.mutate({
      featureId: selectedFeature.id,
      comment: reviewComment,
    });
  };

  return (
    <div className="flex flex-col h-full">
      <Header title="검수 대기열" subtitle="제출된 지리 정보를 검토하고 승인/거부하세요" />

      <div className="flex flex-1 overflow-hidden">
        {/* 대기열 목록 */}
        <div className="flex-1 overflow-auto p-6">
          {/* 필터 */}
          <div className="mb-4">
            <div className="flex gap-2">
              <button
                onClick={() => setFilterSource('all')}
                className={`px-4 py-2 rounded-md ${
                  filterSource === 'all'
                    ? 'bg-primary text-white'
                    : 'bg-white text-gray-700 border border-gray-300'
                }`}
              >
                전체
              </button>
              <button
                onClick={() => setFilterSource('mapper')}
                className={`px-4 py-2 rounded-md ${
                  filterSource === 'mapper'
                    ? 'bg-primary text-white'
                    : 'bg-white text-gray-700 border border-gray-300'
                }`}
              >
                매퍼 제출
              </button>
              <button
                onClick={() => setFilterSource('ai')}
                className={`px-4 py-2 rounded-md ${
                  filterSource === 'ai'
                    ? 'bg-primary text-white'
                    : 'bg-white text-gray-700 border border-gray-300'
                }`}
              >
                AI 탐지
              </button>
            </div>
          </div>

          {/* 목록 */}
          {isLoading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
              <p className="mt-4 text-gray-600">로딩 중...</p>
            </div>
          ) : pendingReviews && pendingReviews.items.length > 0 ? (
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
                      출처
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      신뢰도
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      상태
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      제출일
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      액션
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {pendingReviews.items.map((item: DetectedFeature) => (
                    <tr
                      key={item.id}
                      className={`hover:bg-gray-50 ${
                        selectedFeature?.id === item.id ? 'bg-blue-50' : ''
                      }`}
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {item.feature_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {item.name || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {item.detection_source}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {(item.confidence * 100).toFixed(0)}%
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
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button
                          onClick={() => handleStartReview(item)}
                          className="text-primary hover:text-primary-700 font-medium"
                        >
                          검수하기
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <p className="text-gray-500">검수 대기 중인 항목이 없습니다.</p>
            </div>
          )}
        </div>

        {/* 상세 검수 패널 */}
        {selectedFeature && (
          <div className="w-96 bg-white border-l border-gray-200 p-6 overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">검수 상세</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">객체 유형</label>
                <p className="mt-1 text-gray-900">{selectedFeature.feature_type}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">이름</label>
                <p className="mt-1 text-gray-900">{selectedFeature.name || '-'}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">설명</label>
                <p className="mt-1 text-gray-900">{selectedFeature.description || '-'}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">출처</label>
                <p className="mt-1 text-gray-900">{selectedFeature.detection_source}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">신뢰도</label>
                <p className="mt-1 text-gray-900">
                  {(selectedFeature.confidence * 100).toFixed(0)}%
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">좌표</label>
                <p className="mt-1 text-sm text-gray-600">
                  위도: {selectedFeature.latitude.toFixed(6)}
                  <br />
                  경도: {selectedFeature.longitude.toFixed(6)}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  검수 코멘트 {selectedFeature.review_status === 'pending' && '(선택사항)'}
                </label>
                <textarea
                  value={reviewComment}
                  onChange={(e) => setReviewComment(e.target.value)}
                  placeholder="검수 의견을 입력하세요"
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary focus:border-primary"
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleReject}
                  className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
                  disabled={rejectMutation.isPending}
                >
                  거부
                </button>
                <button
                  onClick={handleApprove}
                  className="flex-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
                  disabled={approveMutation.isPending}
                >
                  승인
                </button>
              </div>

              <button
                onClick={() => setSelectedFeature(null)}
                className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
              >
                닫기
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

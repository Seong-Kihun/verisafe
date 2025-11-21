'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import Header from '@/components/layout/Header';
import type { FeatureType, GeoJSONGeometry, MapperFeatureCreate } from '@/types';
import { mapperAPI } from '@/lib/api/mapper';
import { useMutation } from '@tanstack/react-query';

// Leaflet은 SSR을 지원하지 않으므로 dynamic import 사용
const MapEditor = dynamic(() => import('@/components/map/MapEditor'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-gray-100">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
        <p className="mt-4 text-gray-600">지도 로딩 중...</p>
      </div>
    </div>
  ),
});

export default function MapperPage() {
  const [showForm, setShowForm] = useState(false);
  const [currentGeometry, setCurrentGeometry] = useState<{
    geometry: GeoJSONGeometry;
    geometryType: 'point' | 'line' | 'polygon';
  } | null>(null);

  const [formData, setFormData] = useState<{
    feature_type: FeatureType;
    name: string;
    description: string;
  }>({
    feature_type: 'building',
    name: '',
    description: '',
  });

  const createFeatureMutation = useMutation({
    mutationFn: (data: MapperFeatureCreate) => mapperAPI.createFeature(data),
    onSuccess: () => {
      alert('지리 정보가 성공적으로 제출되었습니다. 검수 후 반영됩니다.');
      setShowForm(false);
      setCurrentGeometry(null);
      setFormData({
        feature_type: 'building',
        name: '',
        description: '',
      });
    },
    onError: (error: any) => {
      alert(`제출 실패: ${error.message}`);
    },
  });

  const handleFeatureCreated = (geometry: GeoJSONGeometry, geometryType: 'point' | 'line' | 'polygon') => {
    setCurrentGeometry({ geometry, geometryType });
    setShowForm(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!currentGeometry) {
      alert('지도에서 객체를 그려주세요.');
      return;
    }

    const data: MapperFeatureCreate = {
      feature_type: formData.feature_type,
      geometry_type: currentGeometry.geometryType,
      geometry_data: currentGeometry.geometry,
      name: formData.name,
      description: formData.description,
      properties: {},
    };

    createFeatureMutation.mutate(data);
  };

  const handleCancel = () => {
    setShowForm(false);
    setCurrentGeometry(null);
  };

  return (
    <div className="flex flex-col h-full">
      <Header
        title="지도 편집"
        subtitle="위성지도를 보고 건물, 도로 등 객체를 그려 정보를 추가하세요"
      />

      <div className="flex flex-1 overflow-hidden">
        {/* 지도 영역 */}
        <div className="flex-1 relative">
          <MapEditor onFeatureCreated={handleFeatureCreated} />
        </div>

        {/* 사이드 패널 (폼) */}
        {showForm && currentGeometry && (
          <div className="w-96 bg-white border-l border-gray-200 p-6 overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">객체 정보 입력</h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  객체 유형
                </label>
                <select
                  value={formData.feature_type}
                  onChange={(e) =>
                    setFormData({ ...formData, feature_type: e.target.value as FeatureType })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary focus:border-primary"
                  required
                >
                  <option value="building">건물</option>
                  <option value="road">도로</option>
                  <option value="bridge">교량</option>
                  <option value="hospital">병원</option>
                  <option value="school">학교</option>
                  <option value="police">경찰서</option>
                  <option value="fire_station">소방서</option>
                  <option value="safe_haven">안전 대피소</option>
                  <option value="shelter">쉼터</option>
                  <option value="landmark">랜드마크</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  이름 (선택사항)
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="예: 서울시청"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary focus:border-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  설명 (선택사항)
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="추가 정보를 입력하세요"
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary focus:border-primary"
                />
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
                <p className="text-sm text-blue-800">
                  <strong>현재 geometry:</strong> {currentGeometry.geometryType}
                </p>
                <p className="text-xs text-blue-600 mt-1">
                  검수 승인 후 공개 지도에 반영됩니다.
                </p>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleCancel}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
                  disabled={createFeatureMutation.isPending}
                >
                  취소
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
                  disabled={createFeatureMutation.isPending}
                >
                  {createFeatureMutation.isPending ? '제출 중...' : '제출'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* 안내 메시지 (폼이 없을 때) */}
        {!showForm && (
          <div className="absolute top-4 right-4 bg-white shadow-lg rounded-lg p-4 max-w-sm border border-gray-200">
            <h3 className="font-bold text-lg mb-2">사용 방법</h3>
            <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside">
              <li>지도 우측 상단의 도구를 사용하세요</li>
              <li>마커, 선, 다각형을 그려 객체를 표시하세요</li>
              <li>그리기 완료 후 정보를 입력하세요</li>
              <li>제출하면 검수 후 반영됩니다</li>
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}

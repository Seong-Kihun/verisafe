'use client';

import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet-draw';
import type { FeatureType, GeoJSONGeometry } from '@/types';

// Leaflet 아이콘 기본 설정 (Next.js에서 필요)
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

interface MapEditorProps {
  onFeatureCreated?: (geometry: GeoJSONGeometry, geometryType: 'point' | 'line' | 'polygon') => void;
}

export default function MapEditor({ onFeatureCreated }: MapEditorProps) {
  const mapRef = useRef<L.Map | null>(null);
  const drawnItemsRef = useRef<L.FeatureGroup>(new L.FeatureGroup());
  const [isMapReady, setIsMapReady] = useState(false);

  useEffect(() => {
    // 이미 지도가 초기화되었다면 스킵
    if (mapRef.current) return;

    // 지도 초기화
    const map = L.map('map-editor', {
      center: [37.5665, 126.9780], // 서울 기본 위치
      zoom: 13,
    });

    mapRef.current = map;

    // 타일 레이어 추가 (위성 이미지)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri',
      maxZoom: 19,
    }).addTo(map);

    // OpenStreetMap 레이블 레이어 (선택사항)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      opacity: 0.5,
      maxZoom: 19,
    }).addTo(map);

    // 그려진 객체들을 담을 레이어
    const drawnItems = drawnItemsRef.current;
    map.addLayer(drawnItems);

    // Leaflet.draw 컨트롤 추가
    const drawControl = new L.Control.Draw({
      edit: {
        featureGroup: drawnItems,
        remove: true,
      },
      draw: {
        polygon: {
          allowIntersection: false,
          showArea: true,
          metric: ['km', 'm'],
        },
        polyline: {
          metric: ['km', 'm'],
        },
        rectangle: {
          showArea: true,
          metric: ['km', 'm'],
        },
        circle: false, // 원은 GeoJSON 표준이 아니므로 비활성화
        circlemarker: false,
        marker: true,
      },
    });

    map.addControl(drawControl);

    // 그리기 완료 이벤트
    map.on(L.Draw.Event.CREATED, (event: any) => {
      const layer = event.layer;
      const type = event.layerType;

      drawnItems.addLayer(layer);

      // GeoJSON으로 변환
      const geoJSON = layer.toGeoJSON();
      const geometry = geoJSON.geometry as GeoJSONGeometry;

      let geometryType: 'point' | 'line' | 'polygon';
      if (type === 'marker') {
        geometryType = 'point';
      } else if (type === 'polyline') {
        geometryType = 'line';
      } else {
        geometryType = 'polygon';
      }

      // 콜백 호출
      onFeatureCreated?.(geometry, geometryType);
    });

    setIsMapReady(true);

    // Cleanup
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [onFeatureCreated]);

  return (
    <div className="relative w-full h-full">
      <div id="map-editor" className="w-full h-full" />
      {!isMapReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-gray-600">지도 로딩 중...</p>
          </div>
        </div>
      )}
    </div>
  );
}

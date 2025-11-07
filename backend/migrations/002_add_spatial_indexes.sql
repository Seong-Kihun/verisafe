-- PostGIS 공간 인덱스 추가 (성능 최적화)
-- 작성일: 2025-11-05
-- 목적: 경로 계산 및 위험 정보 검색 최적화

-- PostGIS extension 활성화 (이미 있으면 스킵)
CREATE EXTENSION IF NOT EXISTS postgis;

-- roads 테이블 geometry 컬럼에 GIST 인덱스 추가
-- 용도: 최근접 노드 탐색 (route_calculator.py)
CREATE INDEX IF NOT EXISTS idx_roads_geometry_gist
ON roads USING GIST(geometry);

-- roads 테이블 geography 변환용 GIST 인덱스
-- 용도: 미터 단위 정확도가 필요한 거리 계산
CREATE INDEX IF NOT EXISTS idx_roads_geography_gist
ON roads USING GIST(geography(geometry));

-- hazards 테이블 geometry 컬럼에 GIST 인덱스 추가
-- 용도: 경로 근방 위험 정보 검색 (route.py)
CREATE INDEX IF NOT EXISTS idx_hazards_geometry_gist
ON hazards USING GIST(geometry);

-- hazards 테이블 geography 변환용 GIST 인덱스
-- 용도: ST_DWithin을 사용한 공간 쿼리 최적화
CREATE INDEX IF NOT EXISTS idx_hazards_geography_gist
ON hazards USING GIST(geography(geometry));

-- 인덱스 확인
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('roads', 'hazards')
    AND indexname LIKE '%gist%'
ORDER BY tablename, indexname;

-- 성능 통계 업데이트
ANALYZE roads;
ANALYZE hazards;

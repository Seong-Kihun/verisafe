# VeriSafe MVP 개발 계획서

**프로젝트명**: VeriSafe - 안전한 경로 추천 네비게이션  
**버전**: 2.0  
**작성일**: 2025년 1월  
**수정일**: 2025년 1월 20일 (선임 개발자 피드백 반영)  
**문서 목적**: 선임 개발자 검토 및 승인

---

## 1. 프로젝트 개요

### 1.1 목적
KOICA 사업 수행인력 및 구호활동가를 위한 안전 경로 추천 네비게이션 앱 개발. 기존 "가장 빠른 경로"가 아닌 "가장 안전한 경로"를 제공하여 안전 위협을 최소화.

### 1.2 MVP 범위
- ✅ 지도 기반 네비게이션 (안전/빠른 경로 비교)
- ✅ 사용자 제보 시스템 (위험 정보 등록)
- ✅ 실시간 위험도 계산
- ✅ 관리자 제보 검증 기능
- ❌ AI 기반 예측 모델 (추후)
- ❌ 오프라인 모드 (추후)
- ❌ 외부 API 실시간 연동 (추후)

### 1.3 시연 대상지
**남수단 주바(Juba)**
- 좌표: 4.8594°N, 31.5713°E
- 반경: 약 15km
- 주요 지점: 주바 국제공항, 시청, 대학병원

---

## 2. 기술 스택

### 2.1 프론트엔드
```
React Native + Expo
├── React Navigation (화면 전환)
├── React Native Maps (지도 표시)
├── AsyncStorage (로컬 저장)
├── Axios (API 통신)
└── React Native Vector Icons (아이콘)
```

### 2.2 백엔드
```
Python 3.11+
├── FastAPI (API 프레임워크)
├── SQLAlchemy (ORM)
├── Alembic (마이그레이션)
├── Pydantic (데이터 검증)
├── JWT (인증)
├── bcrypt (비밀번호 암호화)
└── Python-dotenv (환경 변수)
```

### 2.3 데이터베이스
```
PostgreSQL 14+
├── PostGIS (공간 데이터)
└── pgAdmin (관리 도구)
```

### 2.4 캐시 레이어
```
Redis 7+
├── 경로 계산 결과 캐싱
├── 위험도 데이터 캐싱
└── 세션 정보 저장
```

### 2.5 유틸리티
```
Python 라이브러리
├── OSMnx (OSM 데이터 처리)
├── NetworkX (그래프 알고리즘)
└── Shapely (공간 연산)
```

### 2.6 개발 환경
- 로컬 개발 환경
- Git 버전 관리
- Windows 10 환경
- Docker (선택사항, 로컬 환경 구성 간소화)

---

## 3. 시스템 아키텍처

### 3.1 전체 구조
```
┌─────────────────────────────────────────────────────────┐
│                 모바일 앱 (React Native)                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────────┐       │
│  │ 네비게이션 │  │ 제보 등록 │  │  제보 목록   │       │
│  └─────┬─────┘  └─────┬─────┘  └──────┬────────┘       │
│        └──────────────┼────────────────┘                │
└───────────────────────┼─────────────────────────────────┘
                        │ HTTP/HTTPS
┌───────────────────────┼─────────────────────────────────┐
│                API 서버 (FastAPI)                       │
│  ┌────────────────────────────────────────────────┐    │
│  │  REST API                                      │    │
│  │  - /api/auth/*      (인증)                    │    │
│  │  - /api/route/*     (경로 계산)                │    │
│  │  - /api/reports/*   (제보 관리)                │    │
│  │  - /api/map/*       (지도 데이터)              │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┼─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼─────────┐  ┌──▼────────┐  ┌──▼────────────┐
│ PostgreSQL      │  │   Redis   │  │ 파일 저장소    │
│ + PostGIS       │  │ (캐시)    │  │ (제보 이미지)  │
│ - 도로 데이터   │  │           │  │               │
│ - 위험 정보     │  │           │  │               │
│ - 제보 데이터   │  │           │  │               │
│ - 사용자        │  │           │  │               │
└─────────────────┘  └───────────┘  └───────────────┘
```

### 3.2 데이터 흐름

**서버 시작 시 (1회)**:
```
1. OSM 데이터 로드 → NetworkX 그래프 생성
2. 메모리에 그래프 저장 (GraphManager)
3. 위험도 계산 스케줄러 시작 (5분마다 실행)
```

**사용자 경로 요청 시**:
```
사용자 입력 → API 요청 → 메모리 그래프 조회 → 
A* 알고리즘 (이미 계산된 위험도 사용) → 경로 결과 반환
응답 시간: 1-2초
```

**제보 등록 흐름**:
```
1. 사용자 제보 → Presigned URL 요청
2. 이미지 업로드 (직접 저장소로)
3. 제보 데이터 DB 저장 (pending)
4. 관리자 검증 → 승인/거부
5. 승인 시 hazards 테이블에 추가 → 다음 주기(5분)에 위험도 반영
```

---

## 4. 데이터베이스 스키마

### 4.1 테이블 목록

| 테이블명 | 설명 |
|---------|------|
| users | 사용자 정보 |
| roads | 도로 네트워크 |
| hazards | 위험 정보 |
| reports | 사용자 제보 |
| route_calculations | 경로 계산 기록 |
| landmarks | 주요 지점 (랜드마크) |

### 4.2 주요 테이블 상세

#### users (사용자)
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user', -- user, admin, mapper
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
```

#### roads (도로)
```sql
CREATE TABLE roads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    osm_id BIGINT UNIQUE,
    name VARCHAR(200),
    geometry LINESTRING NOT NULL, -- PostGIS
    road_type VARCHAR(50), -- highway, primary, secondary 등
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_roads_geometry ON roads USING GIST(geometry);
CREATE INDEX idx_roads_road_type ON roads(road_type);
```

#### hazards (위험 정보)
```sql
CREATE TABLE hazards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- road_id 제거: hazards는 영역(Area) 기반이며 여러 도로에 영향을 줄 수 있음
    hazard_type VARCHAR(50) NOT NULL, -- armed_conflict, protest_riot, checkpoint 등
    risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    radius FLOAT NOT NULL, -- 영향 반경 (km)
    geometry POINT NOT NULL, -- PostGIS 공간 데이터
    source VARCHAR(50), -- external_api, user_report, system
    description TEXT,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP, -- NULL이면 영구적
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hazards_location ON hazards USING GIST(geometry);
CREATE INDEX idx_hazards_active ON hazards(end_date) WHERE end_date > CURRENT_TIMESTAMP;
CREATE INDEX idx_hazards_type ON hazards(hazard_type);
```

**설계 변경 사유**:
- `road_id` 컬럼 제거: 위험 정보는 특정 하나의 도로가 아닌 **영역(Area)** 단위로 정의됨
- 하나의 위험(예: 시위, 교전)이 여러 도로에 영향을 줄 수 있음
- 경로 계산 시 PostGIS의 `ST_DWithin` 공간 쿼리로 영향받는 도로를 동적으로 판단

#### reports (사용자 제보)
```sql
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    hazard_type VARCHAR(50) NOT NULL,
    description TEXT,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    image_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'pending', -- pending, verified, rejected
    verified_by UUID REFERENCES users(id),
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_reports_user ON reports(user_id);
```

#### landmarks (랜드마크)
```sql
CREATE TABLE landmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50), -- airport, government, hospital, hotel 등
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    geometry POINT NOT NULL, -- PostGIS
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_landmarks_geometry ON landmarks USING GIST(geometry);
```

### 4.3 위험 유형 정의

```python
HAZARD_TYPES = {
    'armed_conflict': {'risk_range': (80, 100), 'icon': '🔫'},
    'protest_riot': {'risk_range': (70, 85), 'icon': '👥'},
    'checkpoint': {'risk_range': (50, 70), 'icon': '⚠️'},
    'road_damage': {'risk_range': (60, 80), 'icon': '🚧'},
    'natural_disaster': {'risk_range': (70, 90), 'icon': '💥'},
    'other': {'risk_range': (40, 60), 'icon': '❓'}
}
```

---

## 5. API 엔드포인트 설계

### 5.1 인증 API

#### POST /api/auth/register
회원가입
```json
Request:
{
  "username": "user123",
  "email": "user@example.com",
  "password": "secure_password"
}

Response:
{
  "id": "uuid",
  "username": "user123",
  "token": "jwt_token"
}
```

#### POST /api/auth/login
로그인
```json
Request:
{
  "username": "user123",
  "password": "secure_password"
}

Response:
{
  "id": "uuid",
  "username": "user123",
  "token": "jwt_token"
}
```

#### GET /api/auth/me
현재 사용자 정보 조회
```json
Response:
{
  "id": "uuid",
  "username": "user123",
  "role": "user"
}
```

### 5.2 경로 계산 API

#### POST /api/route/calculate
경로 계산
```json
Request:
{
  "start": {"lat": 4.8594, "lng": 31.5713},
  "end": {"lat": 4.8450, "lng": 31.5850},
  "preference": "safe"  // safe or fast
}

Response:
{
  "routes": [
    {
      "id": "route_1",
      "type": "safe",
      "distance": 3.2,
      "duration": 12,
      "risk_score": 25,
      "waypoints": [
        {
          "name": "주바 공항",
          "lat": 4.8594,
          "lng": 31.5713,
          "risk": "low"
        }
      ],
      "polyline": [[4.8594, 31.5713], [4.8500, 31.5750], ...]
    },
    {
      "id": "route_2",
      "type": "fast",
      "distance": 2.1,
      "duration": 8,
      "risk_score": 75,
      "warnings": ["검문소 2개 경유", "위험도 높음"],
      "polyline": [...]
    }
  ]
}
```

### 5.3 제보 API

#### POST /api/reports/upload-url
이미지 업로드 URL 요청
```json
Request:
{
  "filename": "image.jpg",
  "content_type": "image/jpeg"
}

Response:
{
  "upload_url": "https://storage.example.com/presigned-url...",
  "file_url": "https://storage.example.com/reports/uuid/image.jpg",
  "expires_in": 300
}
```

#### POST /api/reports/create
제보 등록
```json
Request:
{
  "hazard_type": "checkpoint",
  "description": "불법 검문소 설치됨",
  "latitude": 4.8594,
  "longitude": 31.5713,
  "image_url": "https://storage.example.com/reports/uuid/image.jpg" (선택사항)
}

Response:
{
  "id": "uuid",
  "status": "pending",
  "message": "제보가 접수되었습니다"
}
```

**이미지 업로드 방식**:
- **방식 1 (MVP)**: 로컬 파일 저장소 (빠른 구현)
- **방식 2 (권장)**: Presigned URL 방식 (확장성)
  1. 클라이언트가 `/api/reports/upload-url`로 업로드 URL 요청
  2. 서버가 Presigned URL 발급 (1회용, 만료 시간 5분)
  3. 클라이언트가 Presigned URL로 직접 이미지 업로드
  4. 업로드 완료 후 제보 데이터와 함께 파일 URL 전송

#### GET /api/reports/list
제보 목록 조회
```json
Query Parameters:
- lat (float): 중심 위도
- lng (float): 중심 경도
- radius (float): 반경 (km), 기본값 5
- status (string): pending/verified/rejected

Response:
{
  "reports": [
    {
      "id": "uuid",
      "hazard_type": "checkpoint",
      "description": "불법 검문소",
      "latitude": 4.8594,
      "longitude": 31.5713,
      "status": "pending",
      "created_at": "2025-01-20T10:30:00Z"
    }
  ]
}
```

#### POST /api/reports/{id}/verify
관리자 승인
```json
Response:
{
  "id": "uuid",
  "status": "verified",
  "message": "제보가 승인되었습니다"
}
```

### 5.4 지도 데이터 API

#### GET /api/map/bounds
영역 내 도로/위험 데이터
```json
Query Parameters:
- min_lat, min_lng, max_lat, max_lng

Response:
{
  "roads": [...],
  "hazards": [...],
  "landmarks": [...]
}
```

---

## 6. UI/UX 디자인 시스템

### 6.1 색상 팔레트

```javascript
const Colors = {
  // Primary (KOICA 블루)
  primary: '#0066CC',
  primaryLight: '#66B3FF',
  primaryDark: '#004D99',
  
  // Secondary
  accent: '#F4D160', // 강조 버튼용
  background: '#F8F9FA',
  surface: '#FFFFFF',
  
  // Text
  textPrimary: '#1E293B',
  textSecondary: '#64748B',
  
  // Status
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#3B82F6',
  
  // Risk Levels
  riskVeryLow: '#10B981',   // 0-20
  riskLow: '#F4D160',      // 21-50
  riskMedium: '#F59E0B',    // 51-70
  riskHigh: '#EF4444'       // 71-100
};
```

### 6.2 타이포그래피

```javascript
const Typography = {
  h1: { fontSize: 28, fontWeight: 'bold', fontFamily: 'Pretendard-Bold' },
  h2: { fontSize: 24, fontWeight: 'bold', fontFamily: 'Pretendard-Bold' },
  h3: { fontSize: 20, fontWeight: '600', fontFamily: 'Pretendard-SemiBold' },
  body: { fontSize: 16, fontWeight: '400', fontFamily: 'Pretendard-Regular' },
  caption: { fontSize: 14, fontWeight: '400', fontFamily: 'Pretendard-Regular' },
  small: { fontSize: 12, fontWeight: '400', fontFamily: 'Pretendard-Regular' }
};
```

### 6.3 컴포넌트 스타일

```javascript
const Styles = {
  container: {
    padding: 16,
    backgroundColor: Colors.background
  },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 16,
    marginVertical: 8,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4
  },
  button: {
    borderRadius: 20,
    paddingVertical: 14,
    paddingHorizontal: 24,
    elevation: 2
  },
  input: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.textSecondary,
    padding: 12
  }
};
```

### 6.4 주요 화면 구조

#### Home Screen (메인 지도)
- 상단: 헤더 (VeriSafe 로고, 현위치 버튼, 설정)
- 중앙: 지도 영역 (70% 화면)
- 하단: 출발지/목적지 입력 → 경로 결과 표시

#### Report Create Screen (제보 등록)
- 상단: 헤더 (뒤로가기, 제목)
- 위치 선택: 지도 미리보기 + 현재 위치 버튼
- 위험 유형: 6개 카드 버튼 (그리드 2x3)
- 설명 입력: 멀티라인 텍스트 입력
- 사진 첨부: 카메라/갤러리 버튼
- 하단: 제보 등록 버튼

#### Report List Screen (제보 목록)
- 상단: 검색 바 + 필터 버튼
- 목록: 카드형 리스트 (위험 아이콘, 위치, 시간, 상태)

---

## 7. 핵심 알고리즘 (성능 최적화)

### 7.1 아키텍처 설계 원칙

**핵심 성능 최적화 전략**:
1. 도로 그래프는 서버 시작 시 메모리에 미리 로드 (Singleton 패턴)
2. 위험도 계산은 경로 탐색과 완전 분리 (별도 스케줄러)
3. 사용자 요청 시에는 이미 계산된 위험도를 읽어만 사용

### 7.2 서버 시작 시 그래프 로딩

```python
# app/services/graph_manager.py
from typing import Optional
import networkx as nx

class GraphManager:
    """도로 네트워크 그래프를 메모리에 유지하는 Singleton"""
    
    _instance = None
    _graph: Optional[nx.DiGraph] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """서버 시작 시 한 번만 실행"""
        if self._graph is None:
            print("로딩 중: 주바 도로 네트워크 그래프")
            self._graph = await self._load_from_database()
            
            # 모든 엣지에 위험도 속성 초기화
            for u, v, data in self._graph.edges(data=True):
                data['risk_score'] = 0
            
            print(f"완료: {len(self._graph.nodes)}개 노드, {len(self._graph.edges)}개 엣지")
    
    async def _load_from_database(self):
        """PostGIS에서 도로 데이터를 로드하여 NetworkX 그래프 생성"""
        # OSMnx를 사용하여 그래프 생성
        import osmnx as ox
        G = ox.graph_from_place("Juba, South Sudan", network_type='drive')
        return G
    
    def get_graph(self) -> nx.DiGraph:
        """메모리의 그래프 반환"""
        return self._graph
```

### 7.3 위험도 계산 프로세스 (별도 스케줄러)

```python
# app/services/hazard_scorer.py
import asyncio
from datetime import datetime, timedelta

class HazardScorer:
    """도로별 위험도를 주기적으로 계산하여 그래프에 업데이트"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.update_interval = 300  # 5분마다 업데이트
    
    async def start_scheduler(self):
        """별도 태스크로 위험도 계산 스케줄러 시작"""
        while True:
            await self.update_all_risk_scores()
            await asyncio.sleep(self.update_interval)
    
    async def update_all_risk_scores(self):
        """모든 도로 엣지의 위험도를 재계산"""
        print(f"[{datetime.now()}] 위험도 업데이트 시작")
        
        graph = self.graph_manager.get_graph()
        hazards = await self._get_active_hazards()
        
        # 각 엣지의 위험도 계산
        for u, v, data in graph.edges(data=True):
            risk_score = await self._calculate_edge_risk(data, hazards)
            data['risk_score'] = risk_score
        
        print(f"[{datetime.now()}] 위험도 업데이트 완료")
    
    async def _calculate_edge_risk(self, edge_data, hazards):
        """
        특정 엣지의 위험도 계산
        
        Args:
            edge_data: NetworkX 엣지 데이터 (geometry 정보 포함)
            hazards: 활성 위험 정보 리스트
        
        Returns:
            risk_score: 0-100 사이의 위험도
        """
        total_risk = 0
        
        # 엣지의 중간점 계산
        edge_midpoint = self._get_edge_midpoint(edge_data)
        
        # 영향 반경 내 위험 정보 필터링
        for hazard in hazards:
            distance = self._calculate_distance(edge_midpoint, 
                                                 (hazard['lat'], hazard['lng']))
            
            # 영향 반경 내에 있는지 확인
            if distance <= hazard['radius']:
                # 거리에 따른 가중치 (가까울수록 높은 영향)
                weight = 1 / (1 + distance)
                total_risk += hazard['risk_score'] * weight
        
        # 정규화 (0-100)
        return min(int(total_risk), 100)
    
    async def _get_active_hazards(self):
        """활성화된 위험 정보 조회 (PostGIS 쿼리)"""
        # PostgreSQL + PostGIS 쿼리
        # SELECT * FROM hazards WHERE end_date > NOW() OR end_date IS NULL
        pass
```

### 7.4 경로 계산 API (최적화 후)

```python
# app/routes/route.py
from fastapi import APIRouter
import networkx as nx
from networkx.algorithms.shortest_paths import astar_path

router = APIRouter()

@router.post("/calculate")
async def calculate_route(request: RouteRequest):
    """
    사용자 요청 시: 이미 계산된 위험도를 읽어만 사용
    응답 시간: 1-2초 이내
    """
    graph = graph_manager.get_graph()
    
    # 1. 출발지/목적지 최근접 노드 찾기
    start_node = find_nearest_node(graph, request.start)
    end_node = find_nearest_node(graph, request.end)
    
    # 2. 엣지 가중치 계산 (이미 메모리에 있는 risk_score 사용)
    def weight_function(u, v, data):
        distance = data['length']
        risk = data.get('risk_score', 0)
        
        if request.preference == 'safe':
            return distance * 1000 + risk * 100  # 안전 우선
        else:
            return distance * 1000  # 빠르기 우선
    
    # 3. A* 알고리즘으로 경로 탐색
    try:
        safe_route = astar_path(graph, start_node, end_node, 
                                weight=weight_function)
        
        # 4. 빠른 경로도 계산
        fast_route = astar_path(graph, start_node, end_node, 
                               weight=lambda u, v, d: d['length'] * 1000)
        
        return {
            "routes": [
            {"type": "safe", "route": safe_route},
            {"type": "fast", "route": fast_route}
        ]
    except nx.NetworkXNoPath:
        return {"error": "경로를 찾을 수 없습니다"}
```

### 7.5 성능 개선 효과

| 항목 | 최적화 전 | 최적화 후 |
|------|-----------|-----------|
| 그래프 로딩 | 요청마다 10-30초 | 서버 시작 시 1회 (5초) |
| 위험도 계산 | 요청마다 5-15초 | 5분마다 비동기 (0초) |
| 경로 탐색 | 10-30초 | 1-2초 |
| 총 응답 시간 | 25-60초 | 1-2초 |

**왜 이렇게 빠른가?**
- 그래프가 이미 메모리에 로드되어 있음
- 위험도가 미리 계산되어 있어 추가 쿼리 불필요
- 사용자 요청 시 A* 알고리즘만 실행하면 됨

**트레이드오프**:
- 위험 정보가 5분 지연될 수 있음 (실시간이 아님)
- 하지만 MVP 목표인 "안전 경로 제공"에는 충분히 적합

---

## 8. 데이터 준비

### 8.1 OpenStreetMap 데이터 처리

```python
# 1. OSM 데이터 다운로드 (남수단 주바)
import osmnx as ox

# 주바 지역 데이터 추출
place = "Juba, South Sudan"
G = ox.graph_from_place(place, network_type='drive')

# 2. 도로 네트워크를 그래프로 변환
# 3. PostGIS 형식으로 변환
# 4. DB에 저장
```

### 8.2 더미 데이터 생성

**위험 정보 더미 데이터**:
- 무장 충돌: 3건
- 검문소: 5건
- 도로 파손: 8건
- 시위: 2건

**랜드마크 더미 데이터**:
- 주바 국제공항
- 주바 시청
- Juba University Hospital
- 주요 호텔 3개

---

## 9. 개발 일정

### Phase 1: 프로젝트 구축 (1주)
- Day 1: 프로젝트 초기화 (React Native, FastAPI)
- Day 2: 데이터베이스 설정 (PostgreSQL + PostGIS + Redis)
- Day 3: 기본 UI 컴포넌트 구성
- Day 4: 네비게이션 구조
- Day 5: 스타일 시스템 적용

### Phase 2: 지도 및 데이터 (1주)
- Day 6-7: OSM 데이터 처리 및 저장
- Day 8: 더미 데이터 생성
- Day 9: 지도 표시 구현
- Day 10: 마커 및 정보창

### Phase 3: 경로 계산 (2주) ⚠️ 확대
- Week 3
  - Day 11-12: GraphManager 구현 (Singleton 패턴)
  - Day 13-14: 위험도 계산 스케줄러 구현
  - Day 15: A* 알고리즘 최적화
- Week 4
  - Day 16: 경로 계산 API 개발
  - Day 17-18: 프론트엔드 연동
  - Day 19-20: 성능 테스트 및 최적화

### Phase 4: 제보 시스템 (1주)
- Day 21-22: 제보 등록 화면
- Day 23: 이미지 업로드 기능 (Presigned URL 또는 로컬 저장)
- Day 24: 제보 목록 화면
- Day 25: API 연동

### Phase 5: 통합 및 최적화 (1주)
- Day 26-27: 전체 통합 테스트
- Day 28: 버그 수정
- Day 29: 성능 최종 검증
- Day 30: 시연 준비

**총 개발 기간**: 6주 (약 1.5개월)

---

## 10. 시연 시나리오

### 시나리오 1: 안전 경로 찾기
1. 앱 실행 → 로그인 (데모 계정)
2. 홈 화면에서 주바(Juba) 지도 확인
3. 출발지: "Juba Airport" 선택
4. 목적지: "Juba City Hall" 입력
5. "안전 경로 찾기" 버튼 클릭
6. 두 가지 경로 비교:
   - 안전 경로: 12분, 3.2km, 위험도 낮음 (녹색)
   - 빠른 경로: 8분, 2.1km, 위험도 높음 (빨간색, 경고 표시)
7. 안전 경로 상세 보기 클릭
8. 경로 상 위험 정보 확인

### 시나리오 2: 제보 등록
1. 홈 화면에서 "+ 제보하기" 버튼 클릭
2. 위험 유형 선택: "검문소" (⚠️)
3. 위치 선택: 지도에서 특정 지점 클릭
4. 설명 입력: "불법 검문소 설치되어 있음, 주의 필요"
5. 사진 첨부: (선택사항)
6. "제보 등록하기" 버튼 클릭
7. 제보 목록 화면에서 확인 (상태: 대기중)

### 시나리오 3: 관리자 검증 (데모)
1. 관리자 계정으로 로그인
2. 제보 목록 화면에서 방금 등록한 제보 확인
3. 제보 상세 정보 확인
4. "승인" 버튼 클릭
5. hazards 테이블에 위험 정보 추가 확인

### 시나리오 4: 통합 시연
1. 시나리오 1의 동일한 경로를 다시 요청
2. 이전에 제보한 검문소가 경로에 반영되었는지 확인
3. 해당 지점을 회피하는 새로운 경로 제안 확인

---

## 11. 기술적 도전 과제

### 11.1 도로 네트워크 그래프 구축
**문제**: OpenStreetMap 데이터를 네트워크 그래프로 변환  
**해결책**: NetworkX + OSMnx 라이브러리 활용

### 11.2 실시간 위험도 계산
**문제**: 매 경로 요청마다 복잡한 공간 쿼리  
**해결책**: Redis 캐싱 + PostGIS 공간 인덱스 활용

### 11.3 모바일 성능 최적화
**문제**: 큰 도로 네트워크 데이터 로딩  
**해결책**: Viewport 기반 부분 로딩 + 압축 (polyline)

### 11.4 위치 정확도
**문제**: 위험 정보의 영향 반경 계산  
**해결책**: PostGIS ST_DWithin 활용한 공간 쿼리

---

## 12. 검토 사항

### 12.1 아키텍처
- [ ] 데이터베이스 스키마 설계 적합성
- [ ] API 엔드포인트 구조 검토
- [ ] 보안 고려사항 (인증, 권한)

### 12.2 기술 스택
- [ ] React Native vs Flutter 선택 의견
- [ ] PostgreSQL + PostGIS vs MongoDB 검토
- [ ] Redis 캐시 적용 필수 여부

### 12.3 성능
- [ ] 경로 계산 응답 시간 목표 설정
- [ ] 동시 사용자 수 고려
- [ ] 모바일 앱 메모리 사용량

### 12.4 확장성
- [ ] 추후 AI 모델 통합 가능 여부
- [ ] 다국가 확장 시 구조
- [ ] 외부 API 연동 설계

---

## 13. 리스크 관리

| 리스크 | 영향도 | 완화 방안 | 현재 상태 |
|--------|--------|----------|----------|
| 경로 계산 느림 | 높음 → 중 | GraphManager + 별도 스케줄러 구현 | 해결됨 |
| OSM 데이터 부족 | 중 | 더미 데이터로 보완 | 모니터링 |
| 모바일 성능 저하 | 중 | Viewport 기반 부분 로딩 | 최적화 필요 |
| 지도 API 제한 | 낮 | 마커 수 제한 | 제한사항 인지 |
| 위험 정보 지연 | 낮 | 5분 주기 업데이트 (트레이드오프) | 수용 가능 |

---

## 14. 다음 단계

✅ **선임 개발자 피드백 반영 완료**

현재 상태:
- ✅ 모든 주요 아키텍처 개선사항 반영
- ✅ 성능 최적화 전략 수립
- ✅ 현실적인 개발 일정 확정
- ✅ 기술 스택 명확화

다음 액션:
1. ✅ **이 문서 최종 검토 완료**
2. Git 저장소 생성
3. 프로젝트 초기화 (Phase 1 시작)
4. 개발 착수

---

## 부록

### A. 참고 자료
- [OpenStreetMap](https://www.openstreetmap.org/)
- [PostGIS Documentation](https://postgis.net/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### B. 키워드
- 네비게이션, 안전 경로, OpenStreetMap, PostGIS, React Native, FastAPI

### C. 선임 개발자 피드백 반영 내역

### 반영된 주요 개선사항

#### 1. 성능 최적화 (최우선)
- ✅ **GraphManager 도입**: 도로 그래프를 서버 시작 시 메모리에 미리 로드
- ✅ **위험도 계산 분리**: 별도 스케줄러로 5분마다 비동기 처리
- ✅ **응답 시간 개선**: 25-60초 → 1-2초

#### 2. API 설계 개선
- ✅ **Presigned URL 방식**: 이미지 업로드 성능 향상 (Base64 제거)
- ✅ **파일 저장소 명확화**: 로컬 또는 S3 호환 스토리지

#### 3. 데이터베이스 스키마 수정
- ✅ **hazards 테이블**: road_id 컬럼 제거 (영역 기반 설계)
- ✅ **PostGIS 활용**: ST_DWithin 공간 쿼리로 영향받는 도로 동적 판단

#### 4. 기술 스택 명확화
- ✅ **Redis 추가**: 캐시 레이어 명시 (2.4 섹션)
- ✅ **파일 저장소 정의**: 아키텍처에서 명확히 구분

#### 5. 개발 일정 현실화
- ✅ **Phase 3 확대**: 경로 계산 1주 → 2주
- ✅ **총 기간**: 5주 → 6주

---

**문서 버전**: 2.0  
**작성일**: 2025-01-20  
**최종 수정일**: 2025-01-20 (선임 개발자 피드백 반영)

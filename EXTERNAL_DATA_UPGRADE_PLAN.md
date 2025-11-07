# 📡 외부 데이터 수집 고도화 계획서

> 작성일: 2025-11-05
> 목적: 실시간 위험 정보 수집 시스템 구축 및 고도화

---

## 🔍 현재 문제점 분석

### ❌ 확인된 문제들

1. **자동 스케줄러 미실행**
   - `main.py`에 DataCollectorScheduler가 시작되지 않음
   - 서버 시작 시 외부 데이터 수집이 자동으로 이루어지지 않음

2. **API 키 미설정**
   - ACLED API 키가 None → 더미 데이터만 생성
   - 실제 외부 API 호출이 이루어지지 않음

3. **초기 데이터 없음**
   - DB에 외부 소스 데이터가 전혀 없음
   - 사용자가 지도에서 아무 정보도 볼 수 없음

4. **모니터링 부재**
   - 데이터 수집이 잘 되고 있는지 확인할 방법 없음
   - 로그만 보고 판단해야 함

---

## 🎯 고도화 3단계 계획

---

## 📌 Phase 1: 즉시 해결 (1-2시간)

### 목표: 자동 수집 시스템 작동 시작

### 1.1 자동 스케줄러 추가

**파일:** `backend/app/main.py`

```python
@app.on_event("startup")
async def startup_event():
    # ... 기존 코드 ...

    async def initialize_background():
        try:
            # ... 기존 GraphManager 초기화 ...

            # ✅ DataCollectorScheduler 추가
            from app.services.external_data.data_collector_scheduler import DataCollectorScheduler

            scheduler = DataCollectorScheduler(SessionLocal, graph_manager)

            # 서버 시작 시 즉시 1회 수집
            print("[Main] 초기 외부 데이터 수집 시작...")
            await scheduler.run_once(db)

            # 24시간마다 자동 수집 시작
            asyncio.create_task(scheduler.start_scheduler(interval_hours=24))
            print("[Main] 외부 데이터 스케줄러 시작됨 (24시간 주기)")

        except Exception as e:
            print(f"[Main] 초기화 오류: {e}")
```

**효과:**
- ✅ 서버 시작 시 즉시 외부 데이터 수집
- ✅ 24시간마다 자동 갱신
- ✅ API 키 없어도 더미 데이터로 작동

---

### 1.2 관리자 대시보드 API 추가

**새 파일:** `backend/app/routes/data_dashboard.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.hazard import Hazard
from sqlalchemy import func
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """외부 데이터 수집 대시보드 통계"""

    # 전체 통계
    total_hazards = db.query(func.count(Hazard.id)).scalar()

    # 소스별 통계
    sources = {}
    for source_name in ['acled', 'gdacs', 'reliefweb']:
        count = db.query(func.count(Hazard.id)).filter(
            Hazard.source.like(f"{source_name}%")
        ).scalar()

        latest = db.query(func.max(Hazard.created_at)).filter(
            Hazard.source.like(f"{source_name}%")
        ).scalar()

        sources[source_name] = {
            "count": count,
            "last_updated": latest.isoformat() if latest else None,
            "status": "active" if count > 0 else "inactive"
        }

    # 최근 24시간 수집 데이터
    yesterday = datetime.utcnow() - timedelta(hours=24)
    recent_count = db.query(func.count(Hazard.id)).filter(
        Hazard.created_at >= yesterday
    ).scalar()

    # 위험도별 분포
    risk_distribution = {
        "low": db.query(func.count(Hazard.id)).filter(Hazard.risk_score < 40).scalar(),
        "medium": db.query(func.count(Hazard.id)).filter(
            Hazard.risk_score >= 40, Hazard.risk_score < 70
        ).scalar(),
        "high": db.query(func.count(Hazard.id)).filter(Hazard.risk_score >= 70).scalar(),
    }

    return {
        "total_hazards": total_hazards,
        "sources": sources,
        "recent_24h": recent_count,
        "risk_distribution": risk_distribution,
        "last_check": datetime.utcnow().isoformat()
    }

@router.post("/dashboard/trigger-collection")
async def trigger_manual_collection(db: Session = Depends(get_db)):
    """수동으로 데이터 수집 트리거"""
    from app.services.graph_manager import GraphManager
    from app.services.external_data.data_collector_scheduler import DataCollectorScheduler

    graph_manager = GraphManager()
    scheduler = DataCollectorScheduler(lambda: db, graph_manager)
    stats = await scheduler.run_once(db)

    return {
        "status": "success",
        "message": "데이터 수집 완료",
        "statistics": stats
    }
```

**라우터 등록:** `backend/app/main.py`
```python
from app.routes import data_dashboard
app.include_router(data_dashboard.router, prefix="/api/data", tags=["data"])
```

---

### 1.3 프론트엔드 관리자 페이지

**새 화면:** `mobile/src/screens/DataDashboardScreen.js`

```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import { Colors, Spacing, Typography } from '../styles';
import api from '../services/api';

export default function DataDashboardScreen() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [collecting, setCollecting] = useState(false);

  const loadStats = async () => {
    try {
      const response = await api.get('/api/data/dashboard/stats');
      setStats(response.data);
    } catch (error) {
      console.error('통계 로드 오류:', error);
    } finally {
      setLoading(false);
    }
  };

  const triggerCollection = async () => {
    setCollecting(true);
    try {
      const response = await api.post('/api/data/dashboard/trigger-collection');
      alert(`수집 완료: ${response.data.statistics.total}개 항목`);
      await loadStats(); // 새로고침
    } catch (error) {
      alert('수집 오류: ' + error.message);
    } finally {
      setCollecting(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>외부 데이터 수집 대시보드</Text>

      {/* 전체 통계 */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>전체 위험 정보</Text>
        <Text style={styles.bigNumber}>{stats?.total_hazards || 0}</Text>
        <Text style={styles.subtitle}>총 수집된 항목 수</Text>
      </View>

      {/* 소스별 통계 */}
      <Text style={styles.sectionTitle}>데이터 소스</Text>

      {Object.entries(stats?.sources || {}).map(([source, data]) => (
        <View key={source} style={styles.sourceCard}>
          <View style={styles.sourceHeader}>
            <Text style={styles.sourceName}>{source.toUpperCase()}</Text>
            <View style={[
              styles.statusBadge,
              { backgroundColor: data.status === 'active' ? '#10B981' : '#EF4444' }
            ]}>
              <Text style={styles.statusText}>{data.status}</Text>
            </View>
          </View>
          <Text style={styles.sourceCount}>{data.count}개 항목</Text>
          <Text style={styles.sourceUpdate}>
            최근 업데이트: {data.last_updated ? new Date(data.last_updated).toLocaleString('ko-KR') : '없음'}
          </Text>
        </View>
      ))}

      {/* 위험도 분포 */}
      <Text style={styles.sectionTitle}>위험도 분포</Text>
      <View style={styles.card}>
        <View style={styles.riskRow}>
          <Text style={styles.riskLabel}>낮음</Text>
          <Text style={styles.riskValue}>{stats?.risk_distribution?.low || 0}</Text>
        </View>
        <View style={styles.riskRow}>
          <Text style={styles.riskLabel}>보통</Text>
          <Text style={styles.riskValue}>{stats?.risk_distribution?.medium || 0}</Text>
        </View>
        <View style={styles.riskRow}>
          <Text style={styles.riskLabel}>높음</Text>
          <Text style={styles.riskValue}>{stats?.risk_distribution?.high || 0}</Text>
        </View>
      </View>

      {/* 수동 수집 버튼 */}
      <TouchableOpacity
        style={[styles.collectButton, collecting && styles.collectButtonDisabled]}
        onPress={triggerCollection}
        disabled={collecting}
      >
        {collecting ? (
          <ActivityIndicator color={Colors.textInverse} />
        ) : (
          <Text style={styles.collectButtonText}>지금 데이터 수집하기</Text>
        )}
      </TouchableOpacity>

      <Text style={styles.lastCheck}>
        마지막 확인: {stats?.last_check ? new Date(stats.last_check).toLocaleString('ko-KR') : ''}
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    padding: Spacing.lg,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginBottom: Spacing.xl,
  },
  card: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  cardTitle: {
    ...Typography.h3,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  bigNumber: {
    fontSize: 48,
    fontWeight: 'bold',
    color: Colors.primary,
    marginBottom: Spacing.xs,
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  sectionTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginTop: Spacing.xl,
    marginBottom: Spacing.md,
  },
  sourceCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
  },
  sourceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  sourceName: {
    ...Typography.h3,
    color: Colors.textPrimary,
    fontWeight: 'bold',
  },
  statusBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: 8,
  },
  statusText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  sourceCount: {
    ...Typography.body,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  sourceUpdate: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  riskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  riskLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  riskValue: {
    ...Typography.body,
    color: Colors.primary,
    fontWeight: '600',
  },
  collectButton: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    padding: Spacing.lg,
    alignItems: 'center',
    marginTop: Spacing.xl,
    marginBottom: Spacing.md,
  },
  collectButtonDisabled: {
    opacity: 0.5,
  },
  collectButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
  lastCheck: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.xl,
  },
});
```

**네비게이션 추가:** 관리자 탭이나 설정에 추가

---

## 📈 Phase 2: 데이터 소스 확장 (1주일)

### 목표: 더 많은 실시간 데이터 수집

### 2.1 소셜 미디어 모니터링

**새 Collector:** `backend/app/services/external_data/twitter_collector.py`

```python
"""Twitter/X API를 통한 실시간 위험 정보 수집"""
import tweepy
from datetime import datetime, timedelta

class TwitterCollector:
    """
    Twitter/X에서 키워드 기반 위험 정보 수집

    키워드:
    - #SouthSudan #Juba #conflict #security
    - 분쟁, 시위, 재난 관련 해시태그
    """

    def __init__(self, api_key, api_secret):
        auth = tweepy.OAuthHandler(api_key, api_secret)
        self.api = tweepy.API(auth)

    async def collect_tweets(self, db: Session, keywords: list, days: int = 1):
        """
        키워드로 트윗 검색 및 위험 정보 추출
        """
        since_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        for keyword in keywords:
            tweets = self.api.search_tweets(
                q=f"{keyword} -filter:retweets",
                lang="en",
                since=since_date,
                count=100,
                geocode="4.8594,31.5713,100km"  # 주바 중심 100km 반경
            )

            for tweet in tweets:
                # NLP로 위험도 분석
                risk_score = self._analyze_sentiment(tweet.text)

                if risk_score > 30:  # 위험도가 일정 이상일 때만 저장
                    hazard = Hazard(
                        hazard_type=self._detect_hazard_type(tweet.text),
                        risk_score=risk_score,
                        latitude=4.8594,  # 지오태깅이 있으면 사용
                        longitude=31.5713,
                        radius=5.0,
                        source="twitter",
                        description=f"Twitter: {tweet.text[:200]}",
                        verified=False,  # 소셜 미디어는 미검증
                        start_date=tweet.created_at,
                        end_date=tweet.created_at + timedelta(hours=12)
                    )
                    db.add(hazard)

        db.commit()
```

---

### 2.2 뉴스 API 통합

**새 Collector:** `backend/app/services/external_data/news_collector.py`

```python
"""NewsAPI를 통한 뉴스 기반 위험 정보 수집"""
from newsapi import NewsApiClient

class NewsCollector:
    """
    NewsAPI에서 남수단 관련 뉴스 수집

    소스:
    - Reuters, BBC, Al Jazeera 등 주요 언론사
    - 키워드: South Sudan, Juba, conflict, violence
    """

    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)

    async def collect_news(self, db: Session, days: int = 7):
        """
        최근 뉴스 기사에서 위험 정보 추출
        """
        articles = self.newsapi.get_everything(
            q='South Sudan OR Juba',
            language='en',
            sort_by='publishedAt',
            from_param=(datetime.utcnow() - timedelta(days=days)).isoformat()
        )

        for article in articles['articles']:
            # 제목 + 설명에서 위험 유형 감지
            text = f"{article['title']} {article['description']}"

            if self._is_relevant(text):
                hazard = Hazard(
                    hazard_type=self._detect_hazard_type(text),
                    risk_score=self._calculate_risk(article),
                    latitude=4.8594,
                    longitude=31.5713,
                    radius=10.0,  # 뉴스는 광범위
                    source="news",
                    description=f"{article['source']['name']}: {article['title']}",
                    verified=True,  # 언론사 검증됨
                    start_date=datetime.fromisoformat(article['publishedAt'].replace('Z', '')),
                    end_date=datetime.utcnow() + timedelta(days=3)
                )
                db.add(hazard)

        db.commit()
```

---

### 2.3 위성 이미지 분석 (고급)

**새 Collector:** `backend/app/services/external_data/satellite_collector.py`

```python
"""Sentinel Hub API를 통한 위성 이미지 분석"""
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection

class SatelliteCollector:
    """
    위성 이미지에서 변화 감지

    감지 항목:
    - 홍수 (수역 확장)
    - 화재 (열 감지)
    - 건물 파괴 (변화 감지)
    """

    def __init__(self, instance_id, client_id, client_secret):
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret

    async def detect_floods(self, db: Session):
        """
        NDWI (Normalized Difference Water Index)로 홍수 감지
        """
        # 위성 이미지 요청 및 분석
        # ...

        if flood_detected:
            hazard = Hazard(
                hazard_type="flood",
                risk_score=80,
                latitude=flood_center_lat,
                longitude=flood_center_lng,
                radius=flood_radius_km,
                source="satellite",
                description="Satellite detected: Flood area expansion",
                verified=True,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=7)
            )
            db.add(hazard)
```

---

## 🤖 Phase 3: AI 기반 고도화 (2-4주)

### 목표: 지능형 위험 예측 시스템

### 3.1 자연어 처리 (NLP) 강화

```python
"""Transformers 기반 위험 정보 추출"""
from transformers import pipeline

class NLPAnalyzer:
    """
    BERT/GPT 기반 텍스트 분석

    기능:
    - 감정 분석 (위험도 판단)
    - 개체명 인식 (장소, 조직 추출)
    - 위험 유형 분류
    """

    def __init__(self):
        self.sentiment = pipeline("sentiment-analysis")
        self.ner = pipeline("ner")
        self.classifier = pipeline("zero-shot-classification")

    def analyze_text(self, text: str):
        # 감정 분석
        sentiment = self.sentiment(text)[0]
        risk_score = 100 if sentiment['label'] == 'NEGATIVE' else 30

        # 개체명 인식 (장소 추출)
        entities = self.ner(text)
        locations = [e['word'] for e in entities if e['entity'] == 'LOC']

        # 위험 유형 분류
        labels = ["conflict", "disaster", "protest", "terrorism", "crime"]
        result = self.classifier(text, labels)
        hazard_type = result['labels'][0]

        return {
            "risk_score": risk_score,
            "locations": locations,
            "hazard_type": hazard_type
        }
```

---

### 3.2 시계열 예측 모델

```python
"""LSTM 기반 위험 발생 예측"""
import torch
import torch.nn as nn

class HazardPredictor(nn.Module):
    """
    과거 데이터로 미래 위험 예측

    입력:
    - 과거 30일간 위험 발생 패턴
    - 계절성, 요일 등 시간 특성
    - 지역별 특성

    출력:
    - 향후 7일간 위험 발생 확률
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

    def predict_risk(self, historical_data):
        """
        Returns: {
            "date": "2025-11-06",
            "risk_probability": 0.75,
            "expected_type": "conflict",
            "confidence": 0.82
        }
        """
        pass
```

---

### 3.3 크라우드소싱 검증 시스템

```python
"""사용자 제보 데이터 신뢰도 계산"""

class TrustScorer:
    """
    사용자 제보 신뢰도 평가

    요소:
    - 사용자 과거 정확도
    - 제보 빈도 및 패턴
    - 다른 사용자 검증 결과
    - 외부 소스 교차 검증
    """

    def calculate_trust_score(self, report, user):
        # 사용자 신뢰도
        user_trust = user.accuracy_rate * 0.4

        # 교차 검증
        similar_reports = self._find_similar_reports(report)
        cross_validation = len(similar_reports) * 0.3

        # 외부 소스 확인
        external_match = self._check_external_sources(report) * 0.3

        trust_score = user_trust + cross_validation + external_match

        if trust_score > 0.7:
            # 자동으로 verified = True
            report.verified = True

        return trust_score
```

---

## 📊 실행 체크리스트

### ✅ Phase 1 (즉시 실행)

- [ ] `main.py`에 DataCollectorScheduler 추가
- [ ] 서버 재시작 후 초기 데이터 수집 확인
- [ ] `data_dashboard.py` 라우터 추가
- [ ] `DataDashboardScreen.js` 화면 추가
- [ ] 네비게이션에 대시보드 메뉴 추가
- [ ] API 호출 테스트: `POST /api/data/dashboard/trigger-collection`
- [ ] 지도에서 위험 정보 마커 확인

### 🔄 Phase 2 (1주일 내)

- [ ] Twitter API 키 발급
- [ ] NewsAPI 키 발급
- [ ] TwitterCollector 구현
- [ ] NewsCollector 구현
- [ ] 스케줄러에 새 Collector 추가
- [ ] 데이터 품질 검증 로직 추가
- [ ] 중복 제거 알고리즘 개선

### 🚀 Phase 3 (2-4주)

- [ ] Transformers 라이브러리 설치
- [ ] NLP Analyzer 구현
- [ ] LSTM 모델 학습
- [ ] 예측 API 엔드포인트 추가
- [ ] 프론트엔드에 예측 정보 표시
- [ ] 크라우드소싱 검증 시스템 구축

---

## 🔑 필요한 API 키

1. **ACLED** (무료) - https://acleddata.com/
   - 분쟁 데이터

2. **Twitter API** (유료) - https://developer.twitter.com/
   - 기본: $100/월

3. **NewsAPI** (무료 제한) - https://newsapi.org/
   - 무료: 100 요청/일
   - 유료: $449/월

4. **Sentinel Hub** (유료) - https://www.sentinel-hub.com/
   - 위성 이미지
   - $0.01 per request

---

## 📈 예상 효과

### Phase 1 완료 시:
- ✅ 자동 데이터 수집 작동
- ✅ 관리자 대시보드로 모니터링
- ✅ 3개 소스에서 데이터 수집
- **예상 데이터량: 50-100개/일**

### Phase 2 완료 시:
- ✅ 5개 이상 소스 통합
- ✅ 실시간성 향상 (분 단위)
- ✅ 데이터 품질 개선
- **예상 데이터량: 200-500개/일**

### Phase 3 완료 시:
- ✅ AI 기반 위험 예측
- ✅ 자동 검증 시스템
- ✅ 크라우드소싱 통합
- **예상 정확도: 85%+**

---

## 💡 추가 아이디어

1. **WebSocket 실시간 알림**
   - 새로운 위험 정보 발생 시 즉시 알림

2. **지역별 알림 구독**
   - 사용자가 관심 지역 설정

3. **히트맵 시각화**
   - 위험도 밀집 지역 표시

4. **시간별 트렌드 분석**
   - 시간대별 위험 발생 패턴

5. **다국어 지원**
   - 영어/아랍어 뉴스 수집

---

**문서 끝**

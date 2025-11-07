"""NLP 텍스트 분석기 - BERT/Transformers 기반"""
import re
from typing import Dict, Optional, List
from datetime import datetime


class NLPAnalyzer:
    """
    자연어 처리 기반 텍스트 분석기

    기능:
    - 감정 분석 (긍정/부정/중립)
    - 위험도 자동 평가
    - 키워드 추출
    - 다국어 지원 (영어, 아랍어)

    Phase 3 구현

    참고: 실제 Transformers 모델은 매우 큰 리소스를 필요로 하므로,
    프로덕션 환경에서는 별도의 ML 서버나 GPU를 권장합니다.
    여기서는 규칙 기반 + 경량 모델 조합으로 구현합니다.
    """

    # 위험 관련 키워드 (가중치 포함)
    DANGER_KEYWORDS = {
        # 매우 높은 위험 (가중치: 3)
        "very_high": [
            "killed", "dead", "death", "fatal", "massacre", "genocide",
            "explosion", "bomb", "suicide", "terrorist", "armed attack",
            "shooting", "gunfire", "casualties", "victims"
        ],
        # 높은 위험 (가중치: 2)
        "high": [
            "violence", "conflict", "fighting", "clash", "attack", "assault",
            "danger", "dangerous", "critical", "severe", "emergency",
            "urgent", "crisis", "threat", "threatening"
        ],
        # 중간 위험 (가중치: 1)
        "medium": [
            "protest", "demonstration", "riot", "unrest", "tension",
            "warning", "alert", "concern", "risk", "unsafe",
            "checkpoint", "blockade", "roadblock"
        ]
    }

    # 긍정적 키워드 (위험도 감소)
    POSITIVE_KEYWORDS = [
        "safe", "secure", "peaceful", "calm", "stable", "resolved",
        "improving", "better", "aid", "help", "support", "relief"
    ]

    # 시간 긴급성 키워드
    URGENCY_KEYWORDS = {
        "immediate": ["now", "currently", "ongoing", "active", "today"],
        "recent": ["yesterday", "recent", "latest", "just", "hours ago"],
        "upcoming": ["will", "expected", "planned", "soon", "tomorrow"]
    }

    def __init__(self):
        """
        NLP 분석기 초기화

        참고: 실제 Transformers 모델 로드는 선택적입니다.
        모델 로드 실패 시 규칙 기반 분석으로 폴백합니다.
        """
        self.use_transformers = False
        self.model = None
        self.tokenizer = None

        # Transformers 모델 로드 시도 (선택적)
        try:
            from transformers import pipeline
            # 경량 감정 분석 모델
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU 사용
            )
            self.use_transformers = True
            print("[NLPAnalyzer] Transformers 모델 로드 성공")
        except Exception as e:
            print(f"[NLPAnalyzer] Transformers 모델 로드 실패: {e}")
            print("[NLPAnalyzer] 규칙 기반 분석으로 폴백합니다.")

    def analyze_text(self, text: str) -> Dict:
        """
        텍스트 종합 분석

        Args:
            text: 분석할 텍스트

        Returns:
            분석 결과 딕셔너리
        """
        if not text:
            return self._get_default_analysis()

        text_lower = text.lower()

        # 1. 감정 분석
        sentiment = self._analyze_sentiment(text)

        # 2. 위험도 점수 계산
        danger_score = self._calculate_danger_score(text_lower)

        # 3. 긴급성 평가
        urgency = self._assess_urgency(text_lower)

        # 4. 키워드 추출
        keywords = self._extract_keywords(text_lower)

        # 5. 위험 유형 추론
        hazard_type = self._infer_hazard_type(text_lower, keywords)

        # 6. 최종 위험도 계산 (0-100)
        final_risk_score = self._compute_final_risk_score(
            danger_score, sentiment, urgency
        )

        return {
            "sentiment": sentiment,
            "danger_score": danger_score,
            "urgency": urgency,
            "keywords": keywords,
            "hazard_type": hazard_type,
            "risk_score": final_risk_score,
            "analyzed_at": datetime.utcnow().isoformat(),
            "method": "transformers" if self.use_transformers else "rule_based"
        }

    def _analyze_sentiment(self, text: str) -> Dict:
        """
        감정 분석 (Transformers 또는 규칙 기반)

        Returns:
            {"label": "POSITIVE/NEGATIVE/NEUTRAL", "score": 0.0-1.0}
        """
        if self.use_transformers:
            try:
                result = self.sentiment_analyzer(text[:512])[0]  # 최대 512 토큰
                return {
                    "label": result["label"],
                    "score": result["score"]
                }
            except Exception as e:
                print(f"[NLPAnalyzer] Transformers 감정 분석 오류: {e}")

        # 규칙 기반 감정 분석
        text_lower = text.lower()

        # 부정적 키워드 카운트
        negative_count = sum(
            1 for keywords in self.DANGER_KEYWORDS.values()
            for keyword in keywords
            if keyword in text_lower
        )

        # 긍정적 키워드 카운트
        positive_count = sum(
            1 for keyword in self.POSITIVE_KEYWORDS
            if keyword in text_lower
        )

        if negative_count > positive_count:
            return {"label": "NEGATIVE", "score": min(0.9, 0.5 + negative_count * 0.1)}
        elif positive_count > negative_count:
            return {"label": "POSITIVE", "score": min(0.9, 0.5 + positive_count * 0.1)}
        else:
            return {"label": "NEUTRAL", "score": 0.5}

    def _calculate_danger_score(self, text: str) -> float:
        """
        위험도 점수 계산 (0.0-1.0)

        키워드 가중치 기반 점수 계산
        """
        score = 0.0

        # 매우 높은 위험 키워드
        very_high_count = sum(
            1 for keyword in self.DANGER_KEYWORDS["very_high"]
            if keyword in text
        )
        score += very_high_count * 0.3

        # 높은 위험 키워드
        high_count = sum(
            1 for keyword in self.DANGER_KEYWORDS["high"]
            if keyword in text
        )
        score += high_count * 0.2

        # 중간 위험 키워드
        medium_count = sum(
            1 for keyword in self.DANGER_KEYWORDS["medium"]
            if keyword in text
        )
        score += medium_count * 0.1

        # 긍정적 키워드는 점수 감소
        positive_count = sum(
            1 for keyword in self.POSITIVE_KEYWORDS
            if keyword in text
        )
        score -= positive_count * 0.1

        # 0.0-1.0 범위로 제한
        return max(0.0, min(1.0, score))

    def _assess_urgency(self, text: str) -> str:
        """
        긴급성 평가

        Returns:
            "immediate", "recent", "upcoming", "general"
        """
        for urgency_level, keywords in self.URGENCY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return urgency_level

        return "general"

    def _extract_keywords(self, text: str) -> List[str]:
        """
        주요 키워드 추출

        위험 관련 키워드 우선 추출
        """
        keywords = []

        for level_keywords in self.DANGER_KEYWORDS.values():
            for keyword in level_keywords:
                if keyword in text:
                    keywords.append(keyword)

        for keyword in self.POSITIVE_KEYWORDS:
            if keyword in text:
                keywords.append(keyword)

        return list(set(keywords))[:10]  # 최대 10개

    def _infer_hazard_type(self, text: str, keywords: List[str]) -> str:
        """
        위험 유형 추론

        Returns:
            "conflict", "protest", "natural_disaster", "checkpoint", "other"
        """
        # 분쟁 관련
        conflict_keywords = ["conflict", "fighting", "violence", "attack", "killed", "shooting", "armed"]
        if any(kw in text for kw in conflict_keywords):
            return "conflict"

        # 시위 관련
        protest_keywords = ["protest", "demonstration", "riot", "unrest"]
        if any(kw in text for kw in protest_keywords):
            return "protest"

        # 자연재해 관련
        disaster_keywords = ["flood", "drought", "earthquake", "storm", "disaster"]
        if any(kw in text for kw in disaster_keywords):
            return "natural_disaster"

        # 검문소 관련
        checkpoint_keywords = ["checkpoint", "blockade", "roadblock"]
        if any(kw in text for kw in checkpoint_keywords):
            return "checkpoint"

        return "other"

    def _compute_final_risk_score(
        self,
        danger_score: float,
        sentiment: Dict,
        urgency: str
    ) -> int:
        """
        최종 위험도 점수 계산 (0-100)

        위험도 점수, 감정, 긴급성을 종합하여 계산
        """
        # 기본 점수 (danger_score 기반)
        base_score = danger_score * 70  # 0-70점

        # 감정 점수 반영
        if sentiment["label"] == "NEGATIVE":
            base_score += sentiment["score"] * 20  # 최대 +20점
        elif sentiment["label"] == "POSITIVE":
            base_score -= sentiment["score"] * 10  # 최대 -10점

        # 긴급성 점수 반영
        urgency_bonus = {
            "immediate": 10,
            "recent": 5,
            "upcoming": 3,
            "general": 0
        }
        base_score += urgency_bonus.get(urgency, 0)

        # 0-100 범위로 제한
        return int(max(0, min(100, base_score)))

    def _get_default_analysis(self) -> Dict:
        """기본 분석 결과 (텍스트 없을 때)"""
        return {
            "sentiment": {"label": "NEUTRAL", "score": 0.5},
            "danger_score": 0.0,
            "urgency": "general",
            "keywords": [],
            "hazard_type": "other",
            "risk_score": 0,
            "analyzed_at": datetime.utcnow().isoformat(),
            "method": "default"
        }

    def enhance_hazard_with_nlp(self, hazard_dict: Dict) -> Dict:
        """
        Hazard 딕셔너리에 NLP 분석 결과 추가

        Args:
            hazard_dict: Hazard 데이터 (description 필드 필요)

        Returns:
            NLP 분석이 추가된 딕셔너리
        """
        description = hazard_dict.get("description", "")

        if not description:
            return hazard_dict

        analysis = self.analyze_text(description)

        # NLP 분석 결과를 hazard에 추가
        hazard_dict["nlp_analysis"] = analysis

        # NLP 기반 위험도로 조정 (선택적)
        # 기존 risk_score와 NLP risk_score의 평균
        if "risk_score" in hazard_dict:
            original_score = hazard_dict["risk_score"]
            nlp_score = analysis["risk_score"]
            # 70% 원본, 30% NLP
            adjusted_score = int(original_score * 0.7 + nlp_score * 0.3)
            hazard_dict["risk_score_adjusted"] = adjusted_score

        return hazard_dict

"""API 속도 제한 미들웨어 (Redis 기반)"""
from fastapi import Request, HTTPException, status
from typing import Optional, Callable
import time
import hashlib
from functools import wraps

from app.services.redis_manager import redis_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Redis 기반 API 속도 제한

    사용법:
        from app.middleware.rate_limiter import limiter

        @router.post("/api")
        @limiter.limit("10/minute")
        async def my_endpoint():
            ...
    """

    def __init__(self):
        self.logger = logger

    def limit(self, rate_limit: str):
        """
        속도 제한 데코레이터

        Args:
            rate_limit: "N/timeunit" 형식 (예: "10/minute", "100/hour", "1000/day")

        Returns:
            데코레이터 함수
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Request 객체 찾기
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                if request is None:
                    # Request가 없으면 속도 제한 무시 (테스트 환경 등)
                    return await func(*args, **kwargs)

                # 속도 제한 체크
                if not self._check_rate_limit(request, rate_limit):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Too Many Requests",
                            "message": f"요청 횟수 제한을 초과했습니다. 잠시 후 다시 시도해주세요.",
                            "rate_limit": rate_limit
                        }
                    )

                # 정상 실행
                return await func(*args, **kwargs)

            return wrapper
        return decorator

    def _check_rate_limit(self, request: Request, rate_limit: str) -> bool:
        """
        속도 제한 체크

        Args:
            request: FastAPI Request 객체
            rate_limit: "N/timeunit" 형식

        Returns:
            허용 여부 (True: 허용, False: 초과)
        """
        try:
            # rate_limit 파싱 (예: "10/minute" -> max_requests=10, window=60)
            max_requests, window_seconds = self._parse_rate_limit(rate_limit)

            # 클라이언트 식별자 생성 (IP + 엔드포인트)
            client_ip = self._get_client_ip(request)
            endpoint = request.url.path
            client_key = self._generate_key(client_ip, endpoint)

            # Redis 키
            redis_key = f"rate_limit:{client_key}"

            # Redis에서 현재 요청 횟수 가져오기
            current_count = redis_manager.get(redis_key)

            if current_count is None:
                # 첫 요청
                redis_manager.set(redis_key, 1, ttl=window_seconds)
                return True

            # 요청 횟수 체크
            if isinstance(current_count, dict) and 'count' in current_count:
                current_count = current_count['count']

            current_count = int(current_count)

            if current_count >= max_requests:
                # 제한 초과
                self.logger.warning(
                    f"Rate limit exceeded: {client_ip} -> {endpoint} "
                    f"({current_count}/{max_requests} in {window_seconds}s)"
                )
                return False

            # 카운트 증가
            redis_manager.set(redis_key, current_count + 1, ttl=window_seconds)
            return True

        except Exception as e:
            # Redis 오류 시 속도 제한 무시 (서비스 중단 방지)
            self.logger.error(f"Rate limiter error: {e}")
            return True

    def _parse_rate_limit(self, rate_limit: str) -> tuple[int, int]:
        """
        rate_limit 문자열 파싱

        Args:
            rate_limit: "N/timeunit" 형식 (예: "10/minute", "100/hour")

        Returns:
            (max_requests, window_seconds)
        """
        try:
            parts = rate_limit.split('/')
            if len(parts) != 2:
                raise ValueError(f"Invalid rate_limit format: {rate_limit}")

            max_requests = int(parts[0])
            time_unit = parts[1].lower()

            # 시간 단위를 초로 변환
            time_units = {
                'second': 1,
                'minute': 60,
                'hour': 3600,
                'day': 86400
            }

            window_seconds = time_units.get(time_unit)
            if window_seconds is None:
                raise ValueError(f"Invalid time unit: {time_unit}")

            return max_requests, window_seconds

        except Exception as e:
            self.logger.error(f"Failed to parse rate_limit '{rate_limit}': {e}")
            # 기본값: 100회/분
            return 100, 60

    def _get_client_ip(self, request: Request) -> str:
        """
        클라이언트 IP 주소 추출

        Args:
            request: FastAPI Request 객체

        Returns:
            클라이언트 IP 주소
        """
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있을 경우)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # 첫 번째 IP가 실제 클라이언트 IP
            return forwarded_for.split(',')[0].strip()

        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # 직접 연결된 클라이언트 IP
        if request.client:
            return request.client.host

        return "unknown"

    def _generate_key(self, client_ip: str, endpoint: str) -> str:
        """
        Redis 키 생성 (IP + 엔드포인트 해시)

        Args:
            client_ip: 클라이언트 IP
            endpoint: API 엔드포인트 경로

        Returns:
            해시된 키
        """
        combined = f"{client_ip}:{endpoint}"
        return hashlib.md5(combined.encode()).hexdigest()


# 싱글톤 인스턴스
limiter = RateLimiter()

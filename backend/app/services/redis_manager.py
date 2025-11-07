"""Redis 캐싱 관리자"""
import redis
import json
from typing import Optional, Any
from app.config import settings


class RedisManager:
    """Redis 캐싱 관리 (Singleton)"""

    _instance = None
    _client: Optional[redis.Redis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Redis 클라이언트 초기화"""
        if self._client is None:
            try:
                self._client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,  # 자동으로 bytes → str 변환
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # 연결 테스트
                self._client.ping()
                print(f"[RedisManager] 연결 성공: {settings.redis_host}:{settings.redis_port}")
            except Exception as e:
                print(f"[RedisManager] 연결 실패: {e}")
                print("[RedisManager] 캐싱 없이 계속 진행합니다.")
                self._client = None

    def get_client(self) -> Optional[redis.Redis]:
        """Redis 클라이언트 반환"""
        return self._client

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 가져오기

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 (JSON 자동 디코딩) 또는 None
        """
        if self._client is None:
            return None

        try:
            value = self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"[RedisManager] get 오류: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        캐시에 값 저장

        Args:
            key: 캐시 키
            value: 저장할 값 (자동으로 JSON 인코딩)
            ttl: Time-To-Live (초), None이면 settings.redis_cache_ttl 사용

        Returns:
            성공 여부
        """
        if self._client is None:
            return False

        try:
            ttl = ttl or settings.redis_cache_ttl
            serialized = json.dumps(value, ensure_ascii=False)
            self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            print(f"[RedisManager] set 오류: {e}")
            return False

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        if self._client is None:
            return False

        try:
            self._client.delete(key)
            return True
        except Exception as e:
            print(f"[RedisManager] delete 오류: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        패턴에 맞는 모든 키 삭제

        Args:
            pattern: Redis 패턴 (예: "route:*")

        Returns:
            삭제된 키 개수
        """
        if self._client is None:
            return 0

        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            print(f"[RedisManager] delete_pattern 오류: {e}")
            return 0

    def flush_all(self) -> bool:
        """모든 캐시 삭제 (주의!)"""
        if self._client is None:
            return False

        try:
            self._client.flushdb()
            print("[RedisManager] 모든 캐시 삭제됨")
            return True
        except Exception as e:
            print(f"[RedisManager] flush_all 오류: {e}")
            return False

    def get_stats(self) -> dict:
        """Redis 통계 정보"""
        if self._client is None:
            return {"status": "disconnected"}

        try:
            info = self._client.info()
            return {
                "status": "connected",
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses")
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Singleton 인스턴스
redis_manager = RedisManager()

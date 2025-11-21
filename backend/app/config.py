"""애플리케이션 설정"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """환경 변수 기반 설정"""

    # Application
    app_name: str = "VeriSafe API"
    version: str = "1.0.0"
    debug: bool = True

    # Database
    database_type: str = "postgresql"  # postgresql or sqlite
    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "verisafe_user"
    database_password: str = Field(
        default="verisafe_pass_2025",
        description="Database password - 프로덕션에서는 환경 변수로 설정 필요"
    )
    database_name: str = "verisafe_db"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = Field(
        default="verisafe_redis_2025",
        description="Redis password - 프로덕션에서는 환경 변수로 설정 필요"
    )
    redis_db: int = 0
    redis_cache_ttl: int = 300  # 5분

    # JWT (환경 변수 필수!)
    secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION",
        description="JWT secret key - 프로덕션에서 반드시 변경 필요"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    def validate_production_secrets(self):
        """프로덕션 환경에서 기본 비밀 키 사용 방지 (개발 환경은 경고만)"""
        warnings = []

        # JWT Secret Key 검증
        if self.secret_key == "CHANGE-ME-IN-PRODUCTION":
            warnings.append("SECRET_KEY가 기본값입니다")

        # Database Password 검증
        if self.database_password == "verisafe_pass_2025":
            warnings.append("DATABASE_PASSWORD가 기본값입니다")

        # Redis Password 검증
        if self.redis_password == "verisafe_redis_2025":
            warnings.append("REDIS_PASSWORD가 기본값입니다")

        if warnings:
            if self.debug:
                # 개발 환경: 경고만 출력
                print(f"\n[WARN] 보안 경고 (테스트/개발용): {', '.join(warnings)}")
                print("       프로덕션 배포 시 반드시 .env 파일에서 변경하세요!\n")
            else:
                # 프로덕션 환경: 에러 발생
                raise ValueError(
                    f"프로덕션 환경에서 기본 비밀번호 사용 불가: {', '.join(warnings)}\n"
                    "Please set proper values in .env file!"
                )

    # CORS - 개발 환경에서는 모든 localhost 포트 허용
    allowed_origins: str = "http://localhost:8081,http://172.20.10.3:8081,http://192.168.45.177:8081,http://192.168.0.24:8081,http://localhost:19006,http://localhost:19000,http://localhost:8000,http://127.0.0.1:8081,http://127.0.0.1:19006,http://127.0.0.1:19000,http://localhost:3000,http://127.0.0.1:3000,http://172.20.10.3:3000,http://192.168.45.177:3000,http://192.168.0.24:3000"

    # File Storage
    upload_dir: str = "./uploads"
    max_upload_size: int = 10485760

    # External APIs - Phase 1
    acled_api_key: Optional[str] = None
    gdacs_api_url: str = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    reliefweb_api_url: str = "https://api.reliefweb.int/v1"

    # External APIs - Phase 2
    twitter_bearer_token: Optional[str] = None
    news_api_key: Optional[str] = None
    sentinel_client_id: Optional[str] = None
    sentinel_client_secret: Optional[str] = None

    # Background Tasks
    hazard_update_interval_seconds: int = 300  # 5분마다 위험도 업데이트
    external_data_collection_interval_hours: int = 6  # 6시간마다 외부 데이터 수집

    @property
    def database_url(self) -> str:
        """데이터베이스 연결 URL"""
        if self.database_type == "postgresql":
            return f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
        else:
            return "sqlite:///./verisafe.db"

    @property
    def redis_url(self) -> str:
        """Redis 연결 URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # .env 파일의 추가 필드 허용
    }


settings = Settings()

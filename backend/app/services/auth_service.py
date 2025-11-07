"""인증 서비스 (보안 강화: bcrypt 해싱)"""
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.models import User
from app.config import settings


# Passlib 컨텍스트 설정 (bcrypt, 12 라운드)
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)

# OAuth2 스킴 (JWT 토큰 추출용)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT 토큰 생성"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    비밀번호 검증 (bcrypt)

    Args:
        plain_password: 평문 비밀번호
        hashed_password: bcrypt 해시

    Returns:
        일치 여부
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    비밀번호 해싱 (bcrypt, 12 라운드)

    Args:
        password: 평문 비밀번호

    Returns:
        bcrypt 해시 ($2b$12$...)
    """
    return pwd_context.hash(password)


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """사용자 인증"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(lambda: None)  # DB dependency는 라우터에서 주입
) -> User:
    """
    JWT 토큰으로부터 현재 사용자 추출

    Args:
        token: JWT access token
        db: Database session

    Returns:
        User 객체

    Raises:
        HTTPException: 인증 실패 시 401
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보를 확인할 수 없습니다",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # DB에서 사용자 조회
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise credentials_exception
        return user
    finally:
        db.close()


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    현재 활성 사용자 확인

    Args:
        current_user: get_current_user로부터 주입된 사용자

    Returns:
        활성 User 객체

    Raises:
        HTTPException: 비활성 사용자일 경우 400
    """
    # 추후 User 모델에 is_active 필드 추가 시 사용
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="비활성 사용자")
    return current_user


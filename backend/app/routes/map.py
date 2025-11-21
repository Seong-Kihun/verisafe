"""지도 데이터 API"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
import httpx
from typing import List

from app.database import get_db
from app.models import Landmark, Hazard
from app.schemas.map import MapBoundsResponse, LandmarkResponse, HazardResponse, AutocompleteResponse
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# OpenStreetMap Nominatim API 설정
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = "VeriSafe/1.0 (https://verisafe.com)"  # Nominatim 정책에 따라 User-Agent 필요


@router.get("/landmarks", response_model=list[LandmarkResponse])
async def get_landmarks(
    lat: float = Query(4.8594, description="중심 위도"),
    lng: float = Query(31.5713, description="중심 경도"),
    radius: float = Query(15.0, description="반경 (km)"),
    db: Session = Depends(get_db)
):
    """주변 랜드마크 조회"""
    landmarks = db.query(Landmark).all()
    return landmarks


@router.get("/hazards", response_model=list[HazardResponse])
async def get_hazards(
    lat: float = Query(4.8594, description="중심 위도"),
    lng: float = Query(31.5713, description="중심 경도"),
    radius: float = Query(15.0, description="반경 (km)"),
    country: str = Query(None, description="국가 코드 (ISO 3166-1 alpha-2)"),
    db: Session = Depends(get_db)
):
    """주변 위험 정보 조회 (국가별 필터링 지원)"""
    query = db.query(Hazard)

    # 국가 필터링 (우선순위)
    if country:
        query = query.filter(Hazard.country == country)

    # 위치 기반 필터링 (간단한 bounding box)
    # 1도 ≈ 111km이므로 radius를 도 단위로 변환
    degree_radius = radius / 111.0
    query = query.filter(
        Hazard.latitude.between(lat - degree_radius, lat + degree_radius),
        Hazard.longitude.between(lng - degree_radius, lng + degree_radius)
    )

    hazards = query.all()
    logger.info(f"위험 정보 조회: country={country}, 결과={len(hazards)}개")
    return hazards


@router.get("/bounds", response_model=MapBoundsResponse)
async def get_map_data(
    min_lat: float = Query(4.8, description="최소 위도"),
    min_lng: float = Query(31.5, description="최소 경도"),
    max_lat: float = Query(4.9, description="최대 위도"),
    max_lng: float = Query(31.6, description="최대 경도"),
    db: Session = Depends(get_db)
):
    """영역 내 모든 지도 데이터 조회"""
    import time
    start_time = time.time()

    try:
        logger.info(f"/bounds 요청 시작: min_lat={min_lat}, min_lng={min_lng}, max_lat={max_lat}, max_lng={max_lng}")

        # 데이터베이스 쿼리 실행 (타임아웃 방지를 위해 개별 쿼리)
        try:
            query_start = time.time()
            landmarks = db.query(Landmark).all()
            query_time = time.time() - query_start
            logger.info(f"Landmarks 쿼리 완료: {len(landmarks)}개, 소요시간: {query_time:.3f}초")
        except Exception as landmark_error:
            logger.error(f"Landmarks 쿼리 실패: {landmark_error}", exc_info=True)
            landmarks = []

        try:
            query_start = time.time()
            hazards = db.query(Hazard).all()
            query_time = time.time() - query_start
            logger.info(f"Hazards 쿼리 완료: {len(hazards)}개, 소요시간: {query_time:.3f}초")
        except Exception as hazard_error:
            logger.error(f"Hazards 쿼리 실패: {hazard_error}", exc_info=True)
            hazards = []

        total_time = time.time() - start_time
        logger.info(f"/bounds 응답 완료: landmarks={len(landmarks)}, hazards={len(hazards)}, 총 소요시간: {total_time:.3f}초")
        
        return MapBoundsResponse(
            landmarks=landmarks,
            hazards=hazards
        )
    except (ValueError, KeyError) as e:
        total_time = time.time() - start_time
        logger.error(f"/bounds 파라미터 오류 (소요시간: {total_time:.3f}초): {e}")
        raise HTTPException(status_code=400, detail=f"잘못된 요청 파라미터: {str(e)}")
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"/bounds 서버 오류 (소요시간: {total_time:.3f}초): {e}", exc_info=True)
        # 에러 발생 시 빈 결과 반환 (사용자 경험 유지)
        return MapBoundsResponse(
            landmarks=[],
            hazards=[]
        )


@router.get("/search/autocomplete", response_model=list[AutocompleteResponse])
async def autocomplete(
    q: str = Query(..., description="검색어"),
    db: Session = Depends(get_db)
):
    """장소 자동완성 검색 (OpenStreetMap Nominatim API 연동 + Redis 캐싱)"""
    import time
    import json
    from app.services.redis_manager import redis_manager
    start_time = time.time()

    if not q or len(q) < 2:
        return []

    try:
        logger.info(f"검색 요청 시작: '{q}'")

        # Redis 캐시 확인 (검색 결과를 1시간 동안 캐싱)
        cache_key = f"search:autocomplete:{q.lower().strip()}"
        redis_client = redis_manager.get_client()

        if redis_client:
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"캐시에서 검색 결과 반환: '{q}'")
                    cached_data = json.loads(cached_result)
                    # JSON을 AutocompleteResponse 객체로 변환
                    cached_results = [AutocompleteResponse(**item) for item in cached_data]
                    total_time = time.time() - start_time
                    logger.info(f"검색 완료 (캐시): {len(cached_results)}개 결과, 총 소요시간: {total_time:.3f}초")
                    return cached_results
            except Exception as cache_error:
                logger.warning(f"캐시 조회 오류 (무시): {cache_error}")

        # OpenStreetMap Nominatim API 호출 (타임아웃 35초 - 프론트엔드 30초보다 여유있게)
        async with httpx.AsyncClient(timeout=35.0) as client:
            # 검색 쿼리: 사용자 입력만 사용 (더 넓은 범위 검색)
            search_query = q.strip()

            # South Sudan 전체에서 검색, 최대 25개 결과 (더 많은 옵션 제공)
            params = {
                "q": search_query,
                "format": "json",
                "limit": 25,  # 최대 25개 결과로 증가
                "addressdetails": 1,
                "countrycodes": "ss",  # South Sudan로 제한
                "accept-language": "en",  # 영어 우선
            }

            headers = {
                "User-Agent": USER_AGENT
            }

            logger.debug(f"Nominatim API 호출 시작: {params}")
            api_start = time.time()

            try:
                response = await client.get(
                    f"{NOMINATIM_BASE_URL}/search",
                    params=params,
                    headers=headers
                )
                api_time = time.time() - api_start
                logger.debug(f"Nominatim API 응답 수신: {response.status_code}, 소요시간: {api_time:.3f}초")
                response.raise_for_status()

                results = response.json()
                logger.debug(f"Nominatim 응답 파싱 완료: {len(results)}개 결과")
            except httpx.TimeoutException:
                api_time = time.time() - api_start
                logger.error(f"Nominatim API 타임아웃 (소요시간: {api_time:.3f}초)")
                return []
            except httpx.HTTPStatusError as e:
                api_time = time.time() - api_start
                logger.error(f"Nominatim API HTTP 오류 (소요시간: {api_time:.3f}초): {e.response.status_code}")
                return []
            
            # OpenStreetMap 결과를 앱 형식으로 변환
            autocomplete_results = []
            for item in results:
                try:
                    # 주소 문자열 생성
                    address_parts = []
                    if item.get("address"):
                        addr = item["address"]
                        # 주소 구성 요소 추출
                        if addr.get("road"):
                            address_parts.append(addr["road"])
                        if addr.get("suburb") or addr.get("neighbourhood"):
                            address_parts.append(addr.get("suburb") or addr.get("neighbourhood"))
                        if addr.get("city") or addr.get("town"):
                            address_parts.append(addr.get("city") or addr.get("town"))

                    address = ", ".join(address_parts) if address_parts else item.get("display_name", "")

                    # 장소 이름 추출
                    name = item.get("name") or item.get("display_name", "")
                    if not name:
                        continue

                    # importance 점수 추출 (OpenStreetMap의 관련도 점수)
                    base_importance = item.get("importance", 0.0)

                    # 검색어와의 관련성 점수 계산 (추가 boost)
                    relevance_boost = 0.0
                    search_lower = search_query.lower()
                    name_lower = name.lower()

                    # 정확히 일치하면 큰 boost
                    if name_lower == search_lower:
                        relevance_boost = 0.5
                    # 이름이 검색어로 시작하면 중간 boost
                    elif name_lower.startswith(search_lower):
                        relevance_boost = 0.3
                    # 검색어가 이름에 포함되면 작은 boost
                    elif search_lower in name_lower:
                        relevance_boost = 0.1

                    # 최종 importance 점수 = base + relevance boost
                    final_importance = base_importance + relevance_boost

                    # 좌표 정보도 포함 (지도에 표시하기 위해)
                    autocomplete_results.append(
                        AutocompleteResponse(
                            id=f"osm_{item['place_id']}",  # OpenStreetMap place_id 사용
                            name=name,
                            address=address,
                            latitude=float(item.get("lat", 0)) if item.get("lat") else None,
                            longitude=float(item.get("lon", 0)) if item.get("lon") else None,
                            importance=final_importance,
                        )
                    )
                except Exception as item_error:
                    logger.debug(f"결과 항목 처리 오류: {item_error}")
                    continue

            logger.debug(f"변환된 결과: {len(autocomplete_results)}개")

            # 최종 importance 점수 기준으로 정렬 (높은 순서)
            autocomplete_results.sort(key=lambda x: x.importance or 0.0, reverse=True)

            # 상위 15개만 반환 (너무 많은 결과는 UX에 좋지 않음)
            autocomplete_results = autocomplete_results[:15]

            logger.debug(f"importance 기준 정렬 완료")
            if autocomplete_results:
                # Pydantic v2 호환성
                try:
                    result_dict = autocomplete_results[0].model_dump()
                except AttributeError:
                    result_dict = autocomplete_results[0].dict()
                logger.debug(f"첫 번째 결과 예시: {result_dict}")

            # Redis 캐시에 저장 (1시간 TTL)
            if redis_client and autocomplete_results:
                try:
                    # Pydantic 객체를 JSON으로 변환
                    cache_data = [result.model_dump() if hasattr(result, 'model_dump') else result.dict() for result in autocomplete_results]
                    redis_client.setex(
                        cache_key,
                        3600,  # 1시간 TTL
                        json.dumps(cache_data)
                    )
                    logger.debug(f"검색 결과 캐시 저장: '{q}'")
                except Exception as cache_error:
                    logger.warning(f"캐시 저장 오류 (무시): {cache_error}")

            total_time = time.time() - start_time
            logger.info(f"검색 완료: {len(autocomplete_results)}개 결과, 총 소요시간: {total_time:.3f}초")
            return autocomplete_results
            
    except httpx.HTTPError as e:
        total_time = time.time() - start_time
        logger.error(f"Nominatim API 호출 실패 (소요시간: {total_time:.3f}초): {e}")
        logger.error(f"응답 내용: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        # API 실패 시 빈 결과 반환
        return []
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"autocomplete 검색 오류 (소요시간: {total_time:.3f}초): {e}", exc_info=True)
        return []


@router.get("/places/reverse", response_model=LandmarkResponse)
async def reverse_geocode(
    lat: float = Query(..., description="위도"),
    lng: float = Query(..., description="경도"),
    db: Session = Depends(get_db)
):
    """좌표로 역지오코딩 (OpenStreetMap Nominatim API)"""
    import time
    import uuid
    start_time = time.time()

    try:
        logger.info(f"역지오코딩 요청: lat={lat}, lng={lng}")

        async with httpx.AsyncClient(timeout=25.0) as client:
            params = {
                "lat": lat,
                "lon": lng,
                "format": "json",
                "addressdetails": 1,
                "zoom": 18,  # 가장 상세한 주소 정보
            }

            headers = {
                "User-Agent": USER_AGENT
            }

            logger.debug(f"Nominatim reverse geocoding 호출: {params}")
            api_start = time.time()

            try:
                response = await client.get(
                    f"{NOMINATIM_BASE_URL}/reverse",
                    params=params,
                    headers=headers
                )
                api_time = time.time() - api_start
                logger.debug(f"Nominatim reverse 응답: {response.status_code}, 소요시간: {api_time:.3f}초")
                response.raise_for_status()

                result = response.json()
                logger.debug(f"역지오코딩 결과 수신")

                if not result:
                    logger.debug(f"역지오코딩 결과 없음")
                    raise HTTPException(status_code=404, detail="Location not found")
                
                # 주소 구성
                address_parts = []
                if result.get("address"):
                    addr = result["address"]
                    # 주소 구성 요소 추출
                    if addr.get("road"):
                        address_parts.append(addr["road"])
                    if addr.get("suburb") or addr.get("neighbourhood"):
                        address_parts.append(addr.get("suburb") or addr.get("neighbourhood"))
                    if addr.get("city") or addr.get("town"):
                        address_parts.append(addr.get("city") or addr.get("town"))
                    if addr.get("state"):
                        address_parts.append(addr["state"])
                    if addr.get("country"):
                        address_parts.append(addr["country"])
                
                address = ", ".join(address_parts) if address_parts else result.get("display_name", "")
                
                # 장소 이름 추출 (가능한 경우)
                name = result.get("name") or result.get("display_name", "").split(",")[0] or "선택한 위치"
                
                # 카테고리 추출 (OSM type에 따라)
                category = "other"
                osm_type = result.get("osm_type", "")
                if osm_type == "way":
                    if "road" in result.get("address", {}).keys():
                        category = "road"
                elif "building" in result.get("type", ""):
                    category = "building"
                
                total_time = time.time() - start_time
                logger.info(f"역지오코딩 완료: {name}, 소요시간: {total_time:.3f}초")

                return LandmarkResponse(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, f"reverse_{lat}_{lng}"),
                    name=name,
                    category=category,
                    latitude=lat,
                    longitude=lng,
                    description=address or result.get("display_name", "")
                )

            except httpx.TimeoutException:
                api_time = time.time() - api_start
                logger.error(f"Nominatim reverse 타임아웃 (소요시간: {api_time:.3f}초)")
                # 타임아웃 시 fallback으로 처리
                total_time = time.time() - start_time
                logger.info(f"타임아웃 fallback: 기본 위치 정보 반환 (소요시간: {total_time:.3f}초)")
                return LandmarkResponse(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, f"reverse_{lat}_{lng}"),
                    name="선택한 위치",
                    category="other",
                    latitude=lat,
                    longitude=lng,
                    description=f"{lat:.4f}, {lng:.4f}"
                )
            except httpx.HTTPStatusError as e:
                api_time = time.time() - api_start
                logger.error(f"Nominatim reverse HTTP 오류 (소요시간: {api_time:.3f}초): {e.response.status_code}")
                # HTTP 오류 시 fallback으로 처리
                total_time = time.time() - start_time
                logger.info(f"HTTP 오류 fallback: 기본 위치 정보 반환 (소요시간: {total_time:.3f}초)")
                return LandmarkResponse(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, f"reverse_{lat}_{lng}"),
                    name="선택한 위치",
                    category="other",
                    latitude=lat,
                    longitude=lng,
                    description=f"{lat:.4f}, {lng:.4f}"
                )

    except httpx.HTTPError as e:
        total_time = time.time() - start_time
        logger.error(f"Nominatim reverse API 호출 실패 (소요시간: {total_time:.3f}초): {e}")
        # API 실패 시 기본 응답 반환 (좌표만 표시)
        return LandmarkResponse(
            id=uuid.uuid5(uuid.NAMESPACE_DNS, f"reverse_{lat}_{lng}"),
            name="선택한 위치",
            category="other",
            latitude=lat,
            longitude=lng,
            description=f"{lat:.4f}, {lng:.4f}"
        )
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"역지오코딩 오류 (소요시간: {total_time:.3f}초): {e}", exc_info=True)
        # 일반 오류 시에도 기본 응답 반환
        return LandmarkResponse(
            id=uuid.uuid5(uuid.NAMESPACE_DNS, f"reverse_{lat}_{lng}"),
            name="선택한 위치",
            category="other",
            latitude=lat,
            longitude=lng,
            description=f"{lat:.4f}, {lng:.4f}"
        )


@router.get("/places/detail", response_model=LandmarkResponse)
async def get_place_detail(
    id: str = Query(..., description="장소 ID"),
    db: Session = Depends(get_db)
):
    """장소 상세 정보 조회 (OpenStreetMap 또는 DB)"""
    try:
        # OpenStreetMap 장소인 경우 (id가 osm_로 시작)
        if id.startswith("osm_"):
            place_id = id.replace("osm_", "")
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                params = {
                    "place_id": place_id,
                    "format": "json",
                    "addressdetails": 1,
                }
                
                headers = {
                    "User-Agent": USER_AGENT
                }

                logger.debug(f"장소 상세 조회: place_id={place_id}")

                response = await client.get(
                    f"{NOMINATIM_BASE_URL}/lookup",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()

                results = response.json()
                logger.debug(f"Nominatim lookup 응답: {len(results) if results else 0}개 결과")

                if not results or len(results) == 0:
                    logger.debug(f"장소를 찾을 수 없음: place_id={place_id}")
                    raise HTTPException(status_code=404, detail="Place not found")
                
                item = results[0]
                
                # 주소 구성
                address_parts = []
                if item.get("address"):
                    addr = item["address"]
                    if addr.get("road"):
                        address_parts.append(addr["road"])
                    if addr.get("suburb") or addr.get("neighbourhood"):
                        address_parts.append(addr.get("suburb") or addr.get("neighbourhood"))
                    if addr.get("city") or addr.get("town"):
                        address_parts.append(addr.get("city") or addr.get("town"))
                
                address = ", ".join(address_parts) if address_parts else item.get("display_name", "")
                
                # 카테고리 추출
                category = None
                if item.get("type"):
                    category = item["type"]
                elif item.get("class"):
                    category = item["class"]
                
                # LandmarkResponse 형식으로 변환 (UUID는 place_id를 문자열로 변환)
                import uuid
                # place_id를 UUID로 변환 (간단한 해시 사용)
                name_bytes = item.get("name", "").encode()[:16]
                place_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"osm_{place_id}")
                
                return LandmarkResponse(
                    id=place_uuid,
                    name=item.get("name") or item.get("display_name", ""),
                    category=category,
                    latitude=float(item["lat"]),
                    longitude=float(item["lon"]),
                    description=address
                )
        else:
            # DB에서 조회 (기존 Landmark)
            import uuid as uuid_lib
            place_id = uuid_lib.UUID(id)
            place = db.query(Landmark).filter(Landmark.id == place_id).first()
            if not place:
                raise HTTPException(status_code=404, detail="Place not found")
            return place
            
    except httpx.HTTPError as e:
        logger.error(f"Nominatim API 호출 실패: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch place details")
    except ValueError as e:
        logger.error(f"Invalid UUID: {e}")
        raise HTTPException(status_code=400, detail="Invalid place ID")
    except Exception as e:
        logger.error(f"get_place_detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/countries")
async def get_countries(db: Session = Depends(get_db)):
    """위험 정보가 있는 국가 목록 조회 (현재 비활성화)"""
    # Hazard 모델에 country 필드가 없으므로 빈 리스트 반환
    logger.info("국가 목록 조회: country 필드 미지원")
    return {"countries": []}

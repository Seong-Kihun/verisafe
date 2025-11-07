"""MBTiles 오프라인 지도 관리자"""
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import hashlib


class MBTilesManager:
    """
    MBTiles 오프라인 지도 관리
    
    Phase 5 구현:
    - 타일 저장/로드
    - 영역별 다운로드
    - 캐시 관리
    """
    
    def __init__(self, mbtiles_dir: str = "data/mbtiles"):
        """
        Args:
            mbtiles_dir: MBTiles 파일 저장 디렉토리
        """
        self.mbtiles_dir = mbtiles_dir
        os.makedirs(self.mbtiles_dir, exist_ok=True)
    
    def create_mbtiles(self, filename: str, metadata: Dict) -> str:
        """
        새 MBTiles 데이터베이스 생성
        
        Args:
            filename: 파일명
            metadata: 메타데이터 (name, description, bounds, etc.)
        
        Returns:
            생성된 파일 경로
        """
        filepath = os.path.join(self.mbtiles_dir, f"{filename}.mbtiles")
        
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        
        # 메타데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT,
                value TEXT
            )
        """)
        
        # 타일 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                zoom_level INTEGER,
                tile_column INTEGER,
                tile_row INTEGER,
                tile_data BLOB,
                PRIMARY KEY (zoom_level, tile_column, tile_row)
            )
        """)
        
        # 메타데이터 삽입
        for key, value in metadata.items():
            cursor.execute("INSERT INTO metadata (name, value) VALUES (?, ?)", 
                          (key, str(value)))
        
        conn.commit()
        conn.close()
        
        print(f"[MBTilesManager] MBTiles 생성: {filepath}")
        return filepath
    
    def add_tile(self, filename: str, z: int, x: int, y: int, tile_data: bytes) -> bool:
        """
        타일 추가
        
        Args:
            filename: MBTiles 파일명
            z: Zoom level
            x: Tile column
            y: Tile row
            tile_data: 타일 이미지 데이터 (PNG/JPEG)
        
        Returns:
            성공 여부
        """
        filepath = os.path.join(self.mbtiles_dir, f"{filename}.mbtiles")
        
        if not os.path.exists(filepath):
            print(f"[MBTilesManager] 파일 없음: {filepath}")
            return False
        
        try:
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data)
                VALUES (?, ?, ?, ?)
            """, (z, x, y, tile_data))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"[MBTilesManager] 타일 추가 오류: {e}")
            return False
    
    def get_tile(self, filename: str, z: int, x: int, y: int) -> Optional[bytes]:
        """
        타일 조회
        
        Args:
            filename: MBTiles 파일명
            z, x, y: 타일 좌표
        
        Returns:
            타일 데이터 or None
        """
        filepath = os.path.join(self.mbtiles_dir, f"{filename}.mbtiles")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT tile_data FROM tiles
                WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?
            """, (z, x, y))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"[MBTilesManager] 타일 조회 오류: {e}")
            return None
    
    def get_tiles_in_bounds(self, filename: str, min_lat: float, max_lat: float,
                           min_lng: float, max_lng: float, zoom: int) -> List[Tuple]:
        """
        경계 영역 내 타일 리스트 조회
        
        Args:
            filename: MBTiles 파일명
            min_lat, max_lat, min_lng, max_lng: 경계
            zoom: Zoom level
        
        Returns:
            [(z, x, y), ...] 타일 좌표 리스트
        """
        # 위경도 -> 타일 좌표 변환
        def lat_lng_to_tile(lat: float, lng: float, z: int) -> Tuple[int, int]:
            import math
            lat_rad = math.radians(lat)
            n = 2.0 ** z
            x = int((lng + 180.0) / 360.0 * n)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (x, y)
        
        min_x, max_y = lat_lng_to_tile(min_lat, min_lng, zoom)
        max_x, min_y = lat_lng_to_tile(max_lat, max_lng, zoom)
        
        tiles = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.append((zoom, x, y))
        
        return tiles
    
    def get_metadata(self, filename: str) -> Dict:
        """
        MBTiles 메타데이터 조회
        
        Args:
            filename: MBTiles 파일명
        
        Returns:
            메타데이터 딕셔너리
        """
        filepath = os.path.join(self.mbtiles_dir, f"{filename}.mbtiles")
        
        if not os.path.exists(filepath):
            return {}
        
        try:
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name, value FROM metadata")
            rows = cursor.fetchall()
            conn.close()
            
            return {row[0]: row[1] for row in rows}
            
        except Exception as e:
            print(f"[MBTilesManager] 메타데이터 조회 오류: {e}")
            return {}
    
    def get_storage_stats(self, filename: str) -> Dict:
        """
        저장소 통계
        
        Args:
            filename: MBTiles 파일명
        
        Returns:
            통계 정보
        """
        filepath = os.path.join(self.mbtiles_dir, f"{filename}.mbtiles")
        
        if not os.path.exists(filepath):
            return {"error": "파일 없음"}
        
        try:
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            
            # 타일 개수
            cursor.execute("SELECT COUNT(*) FROM tiles")
            tile_count = cursor.fetchone()[0]
            
            # Zoom level별 개수
            cursor.execute("""
                SELECT zoom_level, COUNT(*) 
                FROM tiles 
                GROUP BY zoom_level
                ORDER BY zoom_level
            """)
            zoom_stats = dict(cursor.fetchall())
            
            conn.close()
            
            # 파일 크기
            file_size = os.path.getsize(filepath)
            
            return {
                "filename": filename,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "total_tiles": tile_count,
                "zoom_levels": zoom_stats,
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).isoformat()
            }
            
        except Exception as e:
            print(f"[MBTilesManager] 통계 조회 오류: {e}")
            return {"error": str(e)}


# 싱글톤
mbtiles_manager = MBTilesManager()

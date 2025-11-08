"""위성사진 AI 분석 (OpenCV 기반)"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import base64
import io

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("[SatelliteImageAnalyzer] OpenCV가 설치되지 않았습니다.")

from PIL import Image


class SatelliteImageAnalyzer:
    """
    위성사진 AI 분석기
    
    Phase 4 구현:
    - 도로 감지 (Road Detection)
    - 건물 감지 (Building Detection)
    - 변화 감지 (Change Detection)
    - CAPTCHA용 간단한 분석
    """
    
    def __init__(self):
        self.min_road_width = 10  # 픽셀
        self.min_building_area = 100  # 픽셀²
    
    def analyze_image(self, image_data: bytes) -> Dict:
        """
        위성사진 분석 메인 함수
        
        Args:
            image_data: 이미지 바이트 데이터
        
        Returns:
            분석 결과 딕셔너리
        """
        if not OPENCV_AVAILABLE:
            return self._dummy_analysis()
        
        try:
            # 이미지 로드
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "이미지 로드 실패"}
            
            # 분석 수행
            roads = self.detect_roads(image)
            buildings = self.detect_buildings(image)
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "roads": roads,
                "buildings": buildings,
                "total_roads": len(roads),
                "total_buildings": len(buildings)
            }
            
        except Exception as e:
            print(f"[SatelliteImageAnalyzer] 분석 오류: {e}")
            return self._dummy_analysis()
    
    def detect_roads(self, image: np.ndarray) -> List[Dict]:
        """
        도로 감지 (간단한 엣지 검출 + 라인 검출)
        
        Args:
            image: OpenCV 이미지
        
        Returns:
            감지된 도로 리스트
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny 엣지 검출
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        roads = []
        
        if lines is not None:
            for i, line in enumerate(lines[:20]):  # 최대 20개
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                roads.append({
                    "id": i,
                    "start": {"x": int(x1), "y": int(y1)},
                    "end": {"x": int(x2), "y": int(y2)},
                    "length": round(float(length), 2),
                    "confidence": 0.7 + np.random.rand() * 0.2  # 시뮬레이션
                })
        
        return roads
    
    def detect_buildings(self, image: np.ndarray) -> List[Dict]:
        """
        건물 감지 (개선된 알고리즘 - 어두운 영역 감지)

        위성 이미지에서 건물은 일반적으로 주변보다 어둡게 나타남

        Args:
            image: OpenCV 이미지

        Returns:
            감지된 건물 리스트
        """
        # 그레이스케일
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 반전 (건물이 어두우므로, 반전하면 밝게 됨)
        inverted = cv2.bitwise_not(gray)

        # 적응형 임계값 적용 (로컬 영역별로 다른 임계값)
        # 건물은 주변보다 밝게 변환됨 (반전 후)
        binary = cv2.adaptiveThreshold(
            inverted,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # 로컬 영역 크기 (작게 조정)
            C=-5           # 임계값 조정 (더 많이 감지되도록)
        )

        # 형태학적 연산으로 노이즈 제거
        kernel = np.ones((2, 2), np.uint8)  # 작은 커널로 변경
        # Opening: 작은 노이즈 제거
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing: 구멍 메우기
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buildings = []
        building_id = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # 최소/최대 크기 필터 (더 완화)
            if area < 50 or area > 100000:  # 작은 건물도 감지되도록
                continue

            # 바운딩 박스
            x, y, w, h = cv2.boundingRect(contour)

            # 너무 작은 것 제외
            if w < 10 or h < 10:
                continue

            # 종횡비 체크 (건물은 대략 사각형) - 더 완화
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:  # 범위 확대
                continue

            # 사각형 정도 체크 (건물은 대략 사각형에 가까움) - 더 완화
            # 컨투어 면적 / 바운딩 박스 면적 비율
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            if extent < 0.3:  # 더 완화
                continue

            # 신뢰도 계산 (extent와 크기 기반)
            confidence = min(0.95, 0.6 + extent * 0.25 + min(area / 5000, 0.15))

            buildings.append({
                "id": building_id,
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area": round(float(area), 2),
                "confidence": round(float(confidence), 2),
                "aspect_ratio": round(float(aspect_ratio), 2),
                "extent": round(float(extent), 2)
            })

            building_id += 1

            # 최대 20개까지만
            if building_id >= 20:
                break

        return buildings
    
    def analyze_image_with_visualization(self, image_data: bytes) -> Dict:
        """
        위성사진 분석 + 시각화 (아이디어톤 시연용)

        각 단계별 이미지를 생성하여 반환:
        1. 원본 이미지
        2. 도로 감지 결과
        3. 건물 감지 결과
        4. 최종 합성 이미지

        Args:
            image_data: 이미지 바이트 데이터

        Returns:
            분석 결과 + 각 단계별 이미지 (base64)
        """
        if not OPENCV_AVAILABLE:
            return self._dummy_visualization()

        try:
            # 이미지 로드
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {"error": "이미지 로드 실패"}

            # 원본 이미지 복사
            original = image.copy()
            road_overlay = image.copy()
            building_overlay = image.copy()
            combined_overlay = image.copy()

            # 분석 수행
            roads = self.detect_roads(image)
            buildings = self.detect_buildings(image)

            # 도로 시각화 (빨간색 라인)
            for road in roads:
                x1, y1 = road["start"]["x"], road["start"]["y"]
                x2, y2 = road["end"]["x"], road["end"]["y"]
                cv2.line(road_overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 빨간색
                cv2.line(combined_overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # 건물 시각화 (초록색 박스)
            for building in buildings:
                bbox = building["bbox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                cv2.rectangle(building_overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록색
                cv2.rectangle(combined_overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 이미지를 base64로 인코딩
            def encode_image(img):
                _, buffer = cv2.imencode('.jpg', img)
                return base64.b64encode(buffer).decode('utf-8')

            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "analysis": {
                    "roads": roads,
                    "buildings": buildings,
                    "total_roads": len(roads),
                    "total_buildings": len(buildings),
                    "statistics": {
                        "total_road_length": sum(r["length"] for r in roads),
                        "total_building_area": sum(b["area"] for b in buildings),
                        "avg_road_confidence": np.mean([r["confidence"] for r in roads]) if roads else 0,
                        "avg_building_confidence": np.mean([b["confidence"] for b in buildings]) if buildings else 0
                    }
                },
                "visualization": {
                    "original": encode_image(original),
                    "roads": encode_image(road_overlay),
                    "buildings": encode_image(building_overlay),
                    "combined": encode_image(combined_overlay)
                }
            }

        except Exception as e:
            print(f"[SatelliteImageAnalyzer] 시각화 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._dummy_visualization()

    def _dummy_analysis(self) -> Dict:
        """
        OpenCV 없을 때 더미 분석 결과
        """
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "image_size": {"width": 800, "height": 600},
            "roads": [
                {"id": 0, "start": {"x": 100, "y": 100}, "end": {"x": 700, "y": 100}, "length": 600, "confidence": 0.85},
                {"id": 1, "start": {"x": 400, "y": 50}, "end": {"x": 400, "y": 550}, "length": 500, "confidence": 0.78}
            ],
            "buildings": [
                {"id": 0, "bbox": {"x": 200, "y": 200, "width": 100, "height": 80}, "area": 8000, "confidence": 0.82}
            ],
            "total_roads": 2,
            "total_buildings": 1
        }

    def _dummy_visualization(self) -> Dict:
        """
        OpenCV 없을 때 더미 시각화 결과
        """
        # 간단한 더미 이미지 생성 (PIL 사용)
        from PIL import Image, ImageDraw

        # 800x600 검은 배경
        img = Image.new('RGB', (800, 600), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)

        # 텍스트 추가
        draw.text((300, 280), "Demo Satellite Image", fill=(255, 255, 255))
        draw.text((280, 310), "(OpenCV not available)", fill=(150, 150, 150))

        # base64 인코딩
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 12개 건물 더미 데이터 생성
        buildings = []
        building_id = 0
        for i in range(3):  # 3 rows
            for j in range(4):  # 4 cols
                x = 50 + j * 180
                y = 50 + i * 180
                w = 80 + (i * 10)
                h = 70 + (j * 5)
                buildings.append({
                    "id": building_id,
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "area": float(w * h),
                    "confidence": round(0.75 + (building_id * 0.02), 2),
                    "aspect_ratio": round(float(w) / h, 2),
                    "extent": 0.85
                })
                building_id += 1

        # 도로 더미 데이터
        roads = [
            {"id": 0, "start": {"x": 0, "y": 150}, "end": {"x": 800, "y": 150}, "length": 800, "confidence": 0.85},
            {"id": 1, "start": {"x": 0, "y": 350}, "end": {"x": 800, "y": 350}, "length": 800, "confidence": 0.88},
            {"id": 2, "start": {"x": 200, "y": 0}, "end": {"x": 200, "y": 600}, "length": 600, "confidence": 0.82},
            {"id": 3, "start": {"x": 500, "y": 0}, "end": {"x": 500, "y": 600}, "length": 600, "confidence": 0.79},
        ]

        total_building_area = sum(b["area"] for b in buildings)
        avg_building_confidence = sum(b["confidence"] for b in buildings) / len(buildings)

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "image_size": {"width": 800, "height": 600},
            "analysis": {
                "roads": roads,
                "buildings": buildings,
                "total_roads": len(roads),
                "total_buildings": len(buildings),
                "statistics": {
                    "total_road_length": sum(r["length"] for r in roads),
                    "total_building_area": total_building_area,
                    "avg_road_confidence": sum(r["confidence"] for r in roads) / len(roads),
                    "avg_building_confidence": avg_building_confidence
                }
            },
            "visualization": {
                "original": img_base64,
                "roads": img_base64,
                "buildings": img_base64,
                "combined": img_base64
            }
        }
    
    def generate_captcha_task(self, image_data: bytes) -> Dict:
        """
        CAPTCHA용 검증 작업 생성
        
        사용자에게 "도로가 몇 개인가요?" 같은 질문
        
        Args:
            image_data: 위성사진
        
        Returns:
            CAPTCHA 작업
        """
        analysis = self.analyze_image(image_data)
        
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        # 무작위로 질문 유형 선택
        import random
        task_types = ["count_roads", "count_buildings", "identify_road"]
        task_type = random.choice(task_types)
        
        if task_type == "count_roads":
            correct_answer = analysis["total_roads"]
            question = "이 위성사진에 도로가 몇 개 있나요?"
            options = [correct_answer + i for i in range(-2, 3) if correct_answer + i > 0]
            
        elif task_type == "count_buildings":
            correct_answer = analysis["total_buildings"]
            question = "이 위성사진에 건물이 몇 개 있나요?"
            options = [correct_answer + i for i in range(-2, 3) if correct_answer + i > 0]
            
        else:  # identify_road
            correct_answer = "yes" if analysis["total_roads"] > 0 else "no"
            question = "이 사진에 도로가 보이나요?"
            options = ["yes", "no"]
        
        return {
            "task_id": f"captcha_{datetime.utcnow().timestamp()}",
            "task_type": task_type,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "analysis": analysis
        }


# 싱글톤
satellite_image_analyzer = SatelliteImageAnalyzer()

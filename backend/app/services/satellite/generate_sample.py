"""샘플 위성 이미지 생성 스크립트 (건물 감지 최적화)"""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random


def generate_sample_satellite_image(output_path: str):
    """
    데모용 샘플 위성 이미지 생성 (건물 감지 최적화)

    실제 아이디어톤에서는 실제 남수단 주바 지역 위성 이미지로 교체하세요.
    (Google Earth, Sentinel Hub, NASA Worldview 등에서 다운로드 가능)
    """
    # 이미지 크기 (1024x768)
    width, height = 1024, 768

    # 배경 생성 (위성사진 느낌 - 갈색/녹색)
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # 배경 노이즈 (땅 느낌 - 더 밝게)
    np.random.seed(42)
    for y in range(height):
        for x in range(width):
            # 땅 색상 (밝은 갈색/베이지 계열)
            r = int(140 + np.random.randint(-15, 15))
            g = int(130 + np.random.randint(-15, 15))
            b = int(100 + np.random.randint(-15, 15))
            pixels[x, y] = (r, g, b)

    draw = ImageDraw.Draw(img)

    # 녹지 구역 먼저 그리기 (공원, 나무)
    green_areas = [
        [400, 400, 600, 600],  # 큰 공원
        [50, 50, 140, 90],     # 작은 녹지
        [850, 600, 950, 700],  # 작은 녹지
    ]
    green_color = (80, 140, 70)
    for area in green_areas:
        draw.ellipse(area, fill=green_color)

    # 건물 그리기 (매우 어둡고 명확하게 - 위성에서 보이는 지붕)
    # 건물은 주변보다 훨씬 어두워야 감지가 잘 됨
    building_color = (40, 40, 45)  # 매우 어두운 회색 (거의 검정)
    shadow_color = (30, 30, 35)    # 그림자

    # 건물들 (더 큰 사이즈, 명확한 사각형)
    buildings = [
        (100, 100, 200, 190),   # 건물 1 - 크게
        (230, 100, 290, 190),   # 건물 2
        (350, 70, 470, 170),    # 건물 3 - 크게
        (750, 80, 870, 210),    # 건물 4 - 크게
        (550, 240, 670, 360),   # 건물 5 - 크게
        (100, 340, 230, 460),   # 건물 6 - 크게
        (750, 340, 920, 490),   # 건물 7 - 크게
        (340, 540, 470, 660),   # 건물 8 - 크게
        (500, 540, 600, 630),   # 건물 9
        (80, 550, 160, 640),    # 건물 10
        (230, 560, 320, 670),   # 건물 11
        (750, 540, 840, 630),   # 건물 12
    ]

    for bbox in buildings:
        x1, y1, x2, y2 = bbox
        # 그림자 먼저 (약간 오른쪽 아래로)
        draw.rectangle([x1+3, y1+3, x2+3, y2+3], fill=shadow_color)
        # 건물 본체
        draw.rectangle([x1, y1, x2, y2], fill=building_color, outline=(25, 25, 30), width=3)
        # 건물 디테일 (지붕 텍스처)
        for i in range(y1+10, y2-10, 20):
            draw.line([x1+5, i, x2-5, i], fill=(50, 50, 55), width=1)

    # 도로 그리기 (밝은 회색 - 아스팔트, 건물보다 나중에 그려서 덮어쓰기)
    road_color = (170, 170, 170)

    # 주요 도로 (수평)
    draw.rectangle([0, 220, width, 235], fill=road_color)
    draw.rectangle([0, 510, width, 525], fill=road_color)

    # 주요 도로 (수직)
    draw.rectangle([310, 0, 325, height], fill=road_color)
    draw.rectangle([720, 0, 735, height], fill=road_color)

    # 작은 도로들
    draw.rectangle([155, 80, 165, 620], fill=road_color)
    draw.rectangle([480, 305, 920, 315], fill=road_color)
    draw.rectangle([10, 680, 400, 690], fill=road_color)

    # 도로 중앙선 (흰색 점선)
    dash_color = (200, 200, 200)
    # 수평 도로 중앙선
    for x in range(0, width, 40):
        draw.rectangle([x, 227, x+20, 229], fill=dash_color)
        draw.rectangle([x, 517, x+20, 519], fill=dash_color)
    # 수직 도로 중앙선
    for y in range(0, height, 40):
        draw.rectangle([317, y, 319, y+20], fill=dash_color)
        draw.rectangle([727, y, 729, y+20], fill=dash_color)

    # 텍스트 추가 (위치 정보) - 배경 상자와 함께
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 상단 텍스트 배경
    draw.rectangle([5, 5, 400, 35], fill=(0, 0, 0, 180))
    draw.text((10, 10), "Juba, South Sudan - Sample Area", fill=(255, 255, 255), font=font)

    # 하단 텍스트 배경
    draw.rectangle([5, height-35, 450, height-5], fill=(0, 0, 0, 180))
    draw.text((10, height - 30), "AI Satellite Analysis Demo - VeriSafe", fill=(255, 255, 255), font=font)

    # 범례 추가 (우측 상단)
    legend_x = width - 200
    legend_y = 10
    draw.rectangle([legend_x-5, legend_y-5, width-5, legend_y+95], fill=(0, 0, 0, 180))
    draw.rectangle([legend_x, legend_y+5, legend_x+15, legend_y+20], fill=building_color)
    draw.text((legend_x+20, legend_y+5), "Buildings", fill=(255, 255, 255), font=font_small)
    draw.rectangle([legend_x, legend_y+30, legend_x+15, legend_y+45], fill=road_color)
    draw.text((legend_x+20, legend_y+30), "Roads", fill=(255, 255, 255), font=font_small)
    draw.rectangle([legend_x, legend_y+55, legend_x+15, legend_y+70], fill=green_color)
    draw.text((legend_x+20, legend_y+55), "Green Areas", fill=(255, 255, 255), font=font_small)

    # 약간의 블러 효과 (현실감)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # 저장
    img.save(output_path, 'JPEG', quality=95)
    print(f"[OK] Sample image generated: {output_path}")
    print(f"   - Buildings: 12 (clear dark rectangles)")
    print(f"   - Roads: 6 (bright gray lines)")
    print(f"   - Green areas: 3 (parks/trees)")
    print()
    print("[INFO] For the ideathon, replace with real satellite imagery!")
    print("   Recommended sources: Google Earth, Sentinel Hub, NASA Worldview")


if __name__ == "__main__":
    generate_sample_satellite_image("app/services/satellite/sample_juba.jpg")

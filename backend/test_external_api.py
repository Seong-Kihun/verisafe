"""외부 API 테스트 스크립트"""
import httpx
import asyncio


async def test_gdacs():
    """GDACS API 테스트"""
    print("=" * 60)
    print("GDACS API 테스트 중...")
    print("=" * 60)

    url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    params = {
        'country': 'South Sudan',
        'fromDate': '2024-10-01',
        'toDate': '2025-01-01'
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            print(f"[OK] Status code: {response.status_code}")
            print(f"[OK] Content-Type: {response.headers.get('content-type')}")
            print(f"[OK] Response size: {len(response.text)} chars")

            if response.status_code == 200:
                # XML 파싱
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)
                items = root.findall(".//item")
                print(f"[OK] Found {len(items)} disaster events")

                if len(items) > 0:
                    print("\nFirst event:")
                    item = items[0]
                    title = item.find("title")
                    if title is not None:
                        print(f"  Title: {title.text}")
                else:
                    print("[WARN] No disaster events in South Sudan")
                    print("[WARN] Checking worldwide events...")

                    # 전 세계 이벤트 확인
                    params2 = {'fromDate': '2024-10-01', 'toDate': '2025-01-01'}
                    response2 = await client.get(url, params=params2)
                    root2 = ET.fromstring(response2.text)
                    items2 = root2.findall(".//item")
                    print(f"  Worldwide events: {len(items2)}")

            return response.status_code == 200

    except Exception as e:
        print(f"[ERROR] GDACS API error: {e}")
        return False


async def test_acled():
    """ACLED API 테스트 (API 키 필요)"""
    print("\n" + "=" * 60)
    print("ACLED API 테스트 중...")
    print("=" * 60)

    url = "https://api.acleddata.com/acled/read"

    # API 키 확인
    from app.config import settings
    api_key = settings.acled_api_key

    if not api_key:
        print("[WARN] ACLED API key not configured")
        print("       Set ACLED_API_KEY in .env file")
        print("       Get free API key at https://acleddata.com/")
        return False

    params = {
        'key': api_key,
        'country': 'South Sudan',
        'event_date': '2024-10-01|2025-01-01',
        'event_date_where': 'BETWEEN',
        'limit': 10
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            print(f"[OK] Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    events = data['data']
                    print(f"[OK] Found {len(events)} conflict events")

                    if len(events) > 0:
                        print("\nFirst event:")
                        event = events[0]
                        print(f"  Type: {event.get('event_type')}")
                        print(f"  Date: {event.get('event_date')}")
                        print(f"  Location: {event.get('location')}")
                return True
            else:
                print(f"[ERROR] API error: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False

    except Exception as e:
        print(f"[ERROR] ACLED API error: {e}")
        return False


async def main():
    """Main test function"""
    print("\nExternal API Connection Test")
    print("=" * 60)

    gdacs_ok = await test_gdacs()
    acled_ok = await test_acled()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  GDACS: {'[OK]' if gdacs_ok else '[FAIL]'}")
    print(f"  ACLED: {'[OK]' if acled_ok else '[FAIL] (API key required)'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

"""데이터베이스 중복 데이터 확인 스크립트"""
import sqlite3

conn = sqlite3.connect('C:/Users/ki040/verisafe/backend/verisafe.db')
cursor = conn.cursor()

# 전체 hazards 개수
cursor.execute('SELECT COUNT(*) FROM hazards')
total = cursor.fetchone()[0]
print(f'Total hazards: {total}')

# Flood 개수
cursor.execute('SELECT COUNT(*) FROM hazards WHERE hazard_type = "flood"')
flood_count = cursor.fetchone()[0]
print(f'Flood hazards: {flood_count}')

# 중복 description 확인
cursor.execute('''
SELECT description, COUNT(*) as cnt
FROM hazards
WHERE hazard_type = "flood"
GROUP BY description
ORDER BY cnt DESC
LIMIT 10
''')

print('\nTop 10 duplicate flood descriptions:')
for row in cursor.fetchall():
    count, desc = row[1], row[0]
    desc_short = desc[:70] + '...' if len(desc) > 70 else desc
    print(f'  {count}x: {desc_short}')

# 모든 hazard_type 카운트
cursor.execute('''
SELECT hazard_type, COUNT(*) as cnt
FROM hazards
GROUP BY hazard_type
ORDER BY cnt DESC
''')

print('\nHazard types count:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

conn.close()

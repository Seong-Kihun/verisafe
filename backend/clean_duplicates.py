"""중복 더미 데이터 삭제 스크립트"""
import sqlite3

db_path = 'C:/Users/ki040/verisafe/backend/verisafe.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 더미 데이터만 삭제 (source에 _dummy가 포함된 것들)
cursor.execute('''
DELETE FROM hazards
WHERE source LIKE '%_dummy%'
''')

deleted = cursor.rowcount
print(f'삭제된 더미 데이터: {deleted}개')

conn.commit()

# 남은 데이터 확인
cursor.execute('SELECT COUNT(*) FROM hazards')
remaining = cursor.fetchone()[0]
print(f'남은 위험 정보: {remaining}개')

cursor.execute('SELECT COUNT(*) FROM hazards WHERE hazard_type = "flood"')
flood_count = cursor.fetchone()[0]
print(f'남은 flood 정보: {flood_count}개')

conn.close()
print('\n✅ 중복 데이터 삭제 완료!')

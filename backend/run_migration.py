"""Migration 실행 스크립트"""
import sqlite3
import os

def run_migration():
    """SQLite migration 실행"""
    db_path = os.path.join(os.path.dirname(__file__), 'verisafe.db')
    migration_path = os.path.join(os.path.dirname(__file__), 'migrations', '003_enhance_reports_sqlite.sql')

    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    if not os.path.exists(migration_path):
        print(f"[ERROR] Migration file not found: {migration_path}")
        return

    try:
        # Read migration SQL
        with open(migration_path, 'r', encoding='utf-8') as f:
            migration_sql = f.read()

        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("Running migration...")

        # Execute each statement separately (SQLite doesn't support multiple statements well)
        statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]

        for i, statement in enumerate(statements, 1):
            try:
                print(f"  [{i}/{len(statements)}] Executing statement...")
                cursor.execute(statement)
                conn.commit()
                print(f"  [OK] Statement {i} executed successfully")
            except sqlite3.OperationalError as e:
                # Column might already exist
                if 'duplicate column name' in str(e).lower() or 'already exists' in str(e).lower():
                    print(f"  [SKIP] Statement {i} - {e}")
                else:
                    raise

        print("\n[SUCCESS] Migration completed successfully!")

        # Verify the new columns
        cursor.execute("PRAGMA table_info(reports)")
        columns = cursor.fetchall()
        print("\nReports table columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")

        conn.close()

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_migration()

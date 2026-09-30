import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(__file__))

from database.aws.db_connector import DBConnect

def get_all_schemas(db: DBConnect) -> dict:
    """
    Fetch the full column-level schema for every table in the connected database.

    Returns a dict mapping table_name -> list of column dicts, e.g.:
        {
            'Doctors': [
                {'Field': 'doctor_id', 'Type': 'int', 'Null': 'NO',
                 'Key': 'PRI', 'Default': None, 'Extra': 'auto_increment'},
                ...
            ],
            ...
        }
    """
    if not db.validate_connection():
        print("❌ No active connection – cannot fetch schemas.")
        return {}

    schemas = {}

    try:
        with db.connection.cursor() as cursor:
            # 1. Get all table names
            cursor.execute("SHOW TABLES;")
            tables_rows = cursor.fetchall()

            # DictCursor returns rows like {'Tables_in_hospital_er_management': 'Doctors'}
            # Pull out just the table name regardless of the key name
            table_names = [list(row.values())[0] for row in tables_rows]

            print(f"\n📂 Found {len(table_names)} table(s): {table_names}\n")
            print("=" * 60)

            # 2. DESCRIBE each table
            for table in table_names:
                cursor.execute(f"DESCRIBE `{table}`;")
                columns = cursor.fetchall()
                schemas[table] = columns

                print(f"\n🗂  Table: {table}")
                print("-" * 60)

                # Pretty-print column info
                header = f"{'Field':<25} {'Type':<20} {'Null':<6} {'Key':<5} {'Default':<12} {'Extra'}"
                print(header)
                print("-" * len(header))

                for col in columns:
                    print(
                        f"{col['Field']:<25} "
                        f"{col['Type']:<20} "
                        f"{col['Null']:<6} "
                        f"{col['Key']:<5} "
                        f"{str(col['Default'] or ''):<12} "
                        f"{col['Extra']}"
                    )

    except Exception as e:
        print(f"❌ Error while fetching schemas: {e}")

    return schemas


def get_create_statements(db: DBConnect) -> dict:
    """
    Fetch the full CREATE TABLE statement for every table.

    Returns a dict mapping table_name -> CREATE TABLE SQL string.
    Useful for a complete DDL snapshot of the database.
    """
    if not db.validate_connection():
        print("❌ No active connection – cannot fetch CREATE statements.")
        return {}

    create_stmts = {}

    try:
        with db.connection.cursor() as cursor:
            cursor.execute("SHOW TABLES;")
            table_names = [list(row.values())[0] for row in cursor.fetchall()]

            for table in table_names:
                cursor.execute(f"SHOW CREATE TABLE `{table}`;")
                row = cursor.fetchone()
                # Row contains 'Table' and 'Create Table' keys
                create_stmts[table] = row.get("Create Table", "")

                print(f"\n{'=' * 60}")
                print(f"📄 CREATE TABLE for: {table}")
                print(f"{'=' * 60}")
                print(row.get("Create Table", ""))

    except Exception as e:
        print(f"❌ Error while fetching CREATE statements: {e}")

    return create_stmts


if __name__ == "__main__":
    db = DBConnect()

    print("\n" + "═" * 60)
    print("  SECTION 1 – Column-level schemas (DESCRIBE)")
    print("═" * 60)
    schemas = get_all_schemas(db)

    print("\n\n" + "═" * 60)
    print("  SECTION 2 – Full DDL snapshots (SHOW CREATE TABLE)")
    print("═" * 60)
    create_stmts = get_create_statements(db)

    print("\n✅ Done. All schemas retrieved successfully.")

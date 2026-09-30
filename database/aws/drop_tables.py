import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from database.aws.db_connector import DBConnect


def drop_all_tables(db: DBConnect) -> None:
    """
    Forcefully drop ALL tables in the connected database.

    Strategy:
      1. Disable foreign key checks so order doesn't matter.
      2. Fetch all table names dynamically (no hardcoding needed).
      3. DROP each table.
      4. Re-enable foreign key checks.
    """
    if not db.validate_connection():
        print("❌ No active connection – cannot drop tables.")
        return

    try:
        with db.connection.cursor() as cursor:

            # Step 1 – turn off FK enforcement
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
            print("🔓 Foreign key checks disabled.\n")

            # Step 2 – get all table names
            cursor.execute("SHOW TABLES;")
            rows = cursor.fetchall()
            table_names = [list(row.values())[0] for row in rows]

            if not table_names:
                print("ℹ️  No tables found in the database. Nothing to drop.")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
                return

            print(f"🗑  Dropping {len(table_names)} table(s)...\n")

            # Step 3 – drop each table
            for table in table_names:
                cursor.execute(f"DROP TABLE IF EXISTS `{table}`;")
                print(f"  ✅ Dropped: {table}")

            # Step 4 – re-enable FK enforcement
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
            print("\n🔒 Foreign key checks re-enabled.")

        db.connection.commit()
        print("\n🎉 All tables have been dropped successfully.")

    except Exception as e:
        db.connection.rollback()
        # Always restore FK checks even on error
        try:
            with db.connection.cursor() as cursor:
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        except Exception:
            pass
        print(f"\n❌ Error while dropping tables: {e}")
        raise


if __name__ == "__main__":
    confirm = input(
        "⚠️  WARNING: This will permanently delete ALL tables and data.\n"
        "Type 'yes' to confirm: "
    )

    if confirm.strip().lower() != "yes":
        print("❌ Aborted. No tables were dropped.")
        sys.exit(0)

    db = DBConnect()
    drop_all_tables(db)

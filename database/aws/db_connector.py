import os
from dotenv import load_dotenv
import pymysql


class DBConnect:
    def __init__(self):
        load_dotenv()

        self.db_host = os.getenv('DB_HOST')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')

        try:
            self.connection = pymysql.connect(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name,
                port=3306,
                cursorclass=pymysql.cursors.DictCursor
            )
            print("✅ Database connection established successfully!")

        except pymysql.MySQLError as e:
            print("❌ Database connection failed:")
            print(e)
            self.connection = None

    def validate_connection(self):
        if self.connection and self.connection.open:
            print("✅ Connection is active.")
            return True
        else:
            print("❌ Connection is not active.")
            return False

    def get_all_tables(self):
        if not self.validate_connection():
            return None

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SHOW TABLES;")
                tables = cursor.fetchall()

                print("📂 Tables in database:")
                for table in tables:
                    print(table)

                return tables

        except pymysql.MySQLError as e:
            print("❌ Failed to fetch tables:")
            print(e)
            return None



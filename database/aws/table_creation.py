import os
import pymysql
from dotenv import load_dotenv
from database.aws.db_connector import DBConnect

def ensure_database_exists() -> None:
    """
    Connect to the MySQL server WITHOUT selecting a database,
    then CREATE DATABASE IF NOT EXISTS for the target DB.
    This must run before DBConnect is instantiated.
    """
    load_dotenv()
    host     = os.getenv('DB_HOST')
    db_name  = os.getenv('DB_NAME')
    user     = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')

    try:
        # Connect without specifying a database
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            port=3306
        )
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
            print(f"✅ Database '{db_name}' is ready (created or already exists).")
        conn.close()
    except pymysql.MySQLError as e:
        print(f"❌ Failed to create database: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# DDL statements ordered so that parent tables are created
# before any child table that references them via a FK.
# AUTOINCREMENT (SQLite) → AUTO_INCREMENT (MySQL)
# ─────────────────────────────────────────────────────────────

CREATE_TABLES = [

    # ── Independent / parent tables ──────────────────────────

    """
    CREATE TABLE IF NOT EXISTS Doctors (
        doctor_id   INT          PRIMARY KEY AUTO_INCREMENT,
        doctor_code VARCHAR(10)  UNIQUE NOT NULL,
        f_name      VARCHAR(30)  NOT NULL,
        l_name      VARCHAR(30)  NOT NULL,
        country_code VARCHAR(5),
        phone_number VARCHAR(20),
        official_email_id VARCHAR(50) UNIQUE NOT NULL
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS Nurses (
        nurse_id    INT          PRIMARY KEY AUTO_INCREMENT,
        nurse_code  VARCHAR(10)  UNIQUE NOT NULL,
        f_name      VARCHAR(30)  NOT NULL,
        l_name      VARCHAR(30)  NOT NULL,
        country_code VARCHAR(5),
        phone_number VARCHAR(20),
        official_email_id VARCHAR(50) UNIQUE NOT NULL
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS Departments (
        department_id    INT         PRIMARY KEY AUTO_INCREMENT,
        department_code  VARCHAR(10) UNIQUE NOT NULL,
        department_name  VARCHAR(50) NOT NULL,
        specialisation   TEXT        NOT NULL,
        department_email_id VARCHAR(50) UNIQUE,
        UNIQUE (department_name, specialisation(255))
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS Patients (
        patient_id   INT          PRIMARY KEY AUTO_INCREMENT,
        patient_code VARCHAR(10)  UNIQUE NOT NULL,
        f_name       VARCHAR(40)  NOT NULL,
        l_name       VARCHAR(40)  NOT NULL,
        country_code VARCHAR(5),
        phone_number VARCHAR(20),
        email_id     VARCHAR(50)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS MedicalProductCatalog (
        product_id   INT  PRIMARY KEY AUTO_INCREMENT,
        product_code VARCHAR(10) UNIQUE NOT NULL,
        product_name TEXT        NOT NULL,
        manufacturer TEXT        NOT NULL,
        description  TEXT
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS MedicalEquipmentCatalog (
        equipment_id   INT  PRIMARY KEY AUTO_INCREMENT,
        equipment_code VARCHAR(10) UNIQUE NOT NULL,
        equipment_name TEXT        NOT NULL,
        description    TEXT
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS OperationRooms (
        room_id       INT  PRIMARY KEY AUTO_INCREMENT,
        room_code     VARCHAR(10) UNIQUE NOT NULL,
        room_location TEXT        NOT NULL,
        floor         VARCHAR(20),
        description   TEXT
    ) ENGINE=InnoDB;
    """,

    # ── Tables that depend on the parent tables above ─────────

    """
    CREATE TABLE IF NOT EXISTS Appointments (
        appointment_id   INT      PRIMARY KEY AUTO_INCREMENT,
        patient_id       INT      NOT NULL,
        doctor_id        INT      NOT NULL,
        appointment_time DATETIME NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES Patients(patient_id),
        FOREIGN KEY (doctor_id)  REFERENCES Doctors(doctor_id)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS OperationScheduling (
        operation_id   INT         PRIMARY KEY AUTO_INCREMENT,
        operation_code VARCHAR(10) UNIQUE NOT NULL,
        operation_date DATE        NOT NULL,
        start_time     DATETIME    NOT NULL,
        end_time       DATETIME    NOT NULL,
        or_room_id     INT         NOT NULL,
        department_id  INT         NOT NULL,
        appointment_id INT,
        status         VARCHAR(20) NOT NULL,
        FOREIGN KEY (or_room_id)     REFERENCES OperationRooms(room_id),
        FOREIGN KEY (department_id)  REFERENCES Departments(department_id),
        FOREIGN KEY (appointment_id) REFERENCES Appointments(appointment_id),
        CHECK (end_time > start_time)
    ) ENGINE=InnoDB;
    """,

    # ── M:N junction tables ───────────────────────────────────

    """
    CREATE TABLE IF NOT EXISTS DoctorDepartment (
        doctor_id     INT  NOT NULL,
        department_id INT  NOT NULL,
        role          TEXT NOT NULL,
        PRIMARY KEY (doctor_id, department_id),
        FOREIGN KEY (doctor_id)     REFERENCES Doctors(doctor_id),
        FOREIGN KEY (department_id) REFERENCES Departments(department_id)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS NurseDepartment (
        nurse_id      INT  NOT NULL,
        department_id INT  NOT NULL,
        role          TEXT NOT NULL,
        PRIMARY KEY (nurse_id, department_id),
        FOREIGN KEY (nurse_id)      REFERENCES Nurses(nurse_id),
        FOREIGN KEY (department_id) REFERENCES Departments(department_id)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS DoctorScheduledOperations (
        operation_id INT  NOT NULL,
        doctor_id    INT  NOT NULL,
        doctor_role  TEXT NOT NULL,
        PRIMARY KEY (operation_id, doctor_id),
        FOREIGN KEY (operation_id) REFERENCES OperationScheduling(operation_id),
        FOREIGN KEY (doctor_id)    REFERENCES Doctors(doctor_id)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS NurseScheduledOperations (
        operation_id INT  NOT NULL,
        nurse_id     INT  NOT NULL,
        nurse_role   TEXT NOT NULL,
        PRIMARY KEY (operation_id, nurse_id),
        FOREIGN KEY (operation_id) REFERENCES OperationScheduling(operation_id),
        FOREIGN KEY (nurse_id)     REFERENCES Nurses(nurse_id)
    ) ENGINE=InnoDB;
    """,

    # ── Shift tables ──────────────────────────────────────────

    """
    CREATE TABLE IF NOT EXISTS DoctorShifts (
        shift_id   INT  PRIMARY KEY AUTO_INCREMENT,
        doctor_id  INT  NOT NULL,
        shift_date DATE NOT NULL,
        start_time TIME NOT NULL,
        end_time   TIME NOT NULL,
        FOREIGN KEY (doctor_id) REFERENCES Doctors(doctor_id),
        CHECK (end_time > start_time)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS NurseShifts (
        shift_id   INT  PRIMARY KEY AUTO_INCREMENT,
        nurse_id   INT  NOT NULL,
        shift_date DATE NOT NULL,
        start_time TIME NOT NULL,
        end_time   TIME NOT NULL,
        FOREIGN KEY (nurse_id) REFERENCES Nurses(nurse_id),
        CHECK (end_time > start_time)
    ) ENGINE=InnoDB;
    """,

    # ── Inventory tables ──────────────────────────────────────

    """
    CREATE TABLE IF NOT EXISTS ProductInventory (
        inventory_id INT         PRIMARY KEY AUTO_INCREMENT,
        product_id   INT         NOT NULL,
        lot_number   VARCHAR(15) NOT NULL,
        expiry_date  DATE        NOT NULL,
        quantity     INT         NOT NULL CHECK (quantity >= 0),
        FOREIGN KEY (product_id) REFERENCES MedicalProductCatalog(product_id),
        UNIQUE (product_id, lot_number)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS EquipmentInventory (
        equipment_inv_id INT PRIMARY KEY AUTO_INCREMENT,
        equipment_id     INT NOT NULL,
        quantity         INT NOT NULL CHECK (quantity >= 0),
        FOREIGN KEY (equipment_id) REFERENCES MedicalEquipmentCatalog(equipment_id)
    ) ENGINE=InnoDB;
    """,

    # ── Booking tables (depend on inventory + operations) ─────

    """
    CREATE TABLE IF NOT EXISTS OperationProductBooking (
        booking_id        INT PRIMARY KEY AUTO_INCREMENT,
        operation_id      INT NOT NULL,
        inventory_id      INT NOT NULL,
        quantity_reserved INT NOT NULL CHECK (quantity_reserved > 0),
        FOREIGN KEY (operation_id)  REFERENCES OperationScheduling(operation_id),
        FOREIGN KEY (inventory_id)  REFERENCES ProductInventory(inventory_id)
    ) ENGINE=InnoDB;
    """,

    """
    CREATE TABLE IF NOT EXISTS OperationEquipmentBooking (
        booking_id        INT PRIMARY KEY AUTO_INCREMENT,
        operation_id      INT NOT NULL,
        equipment_inv_id  INT NOT NULL,
        quantity_reserved INT NOT NULL CHECK (quantity_reserved > 0),
        FOREIGN KEY (operation_id)       REFERENCES OperationScheduling(operation_id),
        FOREIGN KEY (equipment_inv_id)   REFERENCES EquipmentInventory(equipment_inv_id)
    ) ENGINE=InnoDB;
    """,
]


def create_all_tables(db: DBConnect) -> None:
    """Execute every DDL statement in CREATE_TABLES using the given DBConnect instance."""

    if not db.validate_connection():
        print("❌ Cannot create tables – no active database connection.")
        return

    with db.connection.cursor() as cursor:
        for ddl in CREATE_TABLES:
            # Extract table name for readable logging
            first_line = [l.strip() for l in ddl.strip().splitlines() if l.strip()][0]
            table_name = first_line.split("EXISTS")[-1].strip().split()[0] if "EXISTS" in first_line else "?"

            try:
                cursor.execute(ddl)
                print(f"✅ Table created (or already exists): {table_name}")
            except pymysql.MySQLError as e:
                print(f"❌ Failed to create table {table_name}: {e}")

    db.connection.commit()
    print("\n🎉 Table creation complete.")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    ensure_database_exists()   # Step 1: create DB on server if it doesn't exist
    db = DBConnect()           # Step 2: connect to that DB
    create_all_tables(db)      # Step 3: create all tables


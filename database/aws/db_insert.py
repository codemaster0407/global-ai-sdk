"""
db_insert.py
------------
Insert helpers for every table in the Hospital ER Management schema.

Design decisions:
- Every function accepts a `DBConnect` instance and the column values as
  keyword arguments so call-sites are self-documenting.
- All FK references are validated *before* the INSERT.  If a required FK is
  missing the function raises `ForeignKeyError` with a clear message stating
  which value was not found in which table/column.
- Optional FKs (e.g. appointment_id in OperationScheduling) are skipped when
  None is passed.
- Functions return the `lastrowid` of the new row so callers can chain
  inserts without needing a second query.
"""

import pymysql
from utils.database_utils.db_connector import DBConnect


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class ForeignKeyError(ValueError):
    """Raised when a required foreign key value does not exist in its parent table."""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_fk(cursor, table: str, pk_col: str, pk_val) -> None:
    """
    Assert that `pk_val` exists in `table`.`pk_col`.
    Raises ForeignKeyError with a human-readable message if it does not.
    """
    cursor.execute(
        f"SELECT 1 FROM `{table}` WHERE `{pk_col}` = %s LIMIT 1;",
        (pk_val,)
    )
    if cursor.fetchone() is None:
        raise ForeignKeyError(
            f"Foreign key violation: value '{pk_val}' not found in "
            f"`{table}`.`{pk_col}`."
        )


def _require_connection(db: DBConnect) -> None:
    if not db.validate_connection():
        raise ConnectionError("No active database connection.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Doctors
# ─────────────────────────────────────────────────────────────────────────────

def insert_doctor(
    db: DBConnect,
    *,
    doctor_code: str,
    f_name: str,
    l_name: str,
    official_email_id: str,
    country_code: str = None,
    phone_number: str = None,
    gender: str = None,
) -> int:
    """
    Insert a new Doctor row.

    Returns:
        doctor_id (int) of the newly created row.
    """
    _require_connection(db)
    sql = """
        INSERT INTO Doctors
            (doctor_code, f_name, l_name, country_code, phone_number, official_email_id, gender)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (doctor_code, f_name, l_name, country_code, phone_number, official_email_id, gender))
        db.connection.commit()
        print(f"✅ Doctor '{f_name} {l_name}' inserted (doctor_code={doctor_code}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert Doctor: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. Nurses
# ─────────────────────────────────────────────────────────────────────────────

def insert_nurse(
    db: DBConnect,
    *,
    nurse_code: str,
    f_name: str,
    l_name: str,
    official_email_id: str,
    country_code: str = None,
    phone_number: str = None,
    gender : str = None, 
) -> int:
    """
    Insert a new Nurse row.

    Returns:
        nurse_id (int) of the newly created row.
    """
    _require_connection(db)
    sql = """
        INSERT INTO Nurses
            (nurse_code, f_name, l_name, country_code, phone_number, official_email_id, gender)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (nurse_code, f_name, l_name, country_code, phone_number, official_email_id, gender))
        db.connection.commit()
        print(f"✅ Nurse '{f_name} {l_name}' inserted (nurse_code={nurse_code}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert Nurse: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. Departments
# ─────────────────────────────────────────────────────────────────────────────

def insert_department(
    db: DBConnect,
    *,
    department_code: str,
    department_name: str,
    specialisation: str,
    department_email_id: str = None,
) -> int:
    """
    Insert a new Department row.

    Returns:
        department_id (int) of the newly created row.
    """
    _require_connection(db)
    sql = """
        INSERT INTO Departments
            (department_code, department_name, specialisation, department_email_id)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (department_code, department_name, specialisation, department_email_id))
        db.connection.commit()
        print(f"✅ Department '{department_name}' inserted (department_code={department_code}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert Department: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 4. Patients
# ─────────────────────────────────────────────────────────────────────────────

def insert_patient(
    db: DBConnect,
    *,
    patient_code: str,
    f_name: str,
    l_name: str,
    country_code: str = None,
    phone_number: str = None,
    email_id: str = None,
    gender : str = None
) -> int:
    """
    Insert a new Patient row.

    Returns:
        patient_id (int) of the newly created row.
    """
    _require_connection(db)
    sql = """
        INSERT INTO Patients
            (patient_code, f_name, l_name, country_code, phone_number, email_id, gender)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (patient_code, f_name, l_name, country_code, phone_number, email_id, gender))
        db.connection.commit()
        print(f"✅ Patient '{f_name} {l_name}' inserted (patient_code={patient_code}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert Patient: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 5. MedicalProductCatalog
# ─────────────────────────────────────────────────────────────────────────────

def insert_medical_product(
    db: DBConnect,
    *,
    product_code: str,
    product_name: str,
    manufacturer: str,
    description: str = None,
) -> int:
    """
    Insert a new Medical Product into the catalog.

    Returns:
        product_id (int) of the newly created row.
    """
    _require_connection(db)
    sql = """
        INSERT INTO MedicalProductCatalog
            (product_code, product_name, manufacturer, description)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (product_code, product_name, manufacturer, description))
        db.connection.commit()
        print(f"✅ Medical product '{product_name}' inserted (product_code={product_code}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert MedicalProductCatalog: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 6. MedicalEquipmentCatalog
# ─────────────────────────────────────────────────────────────────────────────

def insert_medical_equipment(
    db: DBConnect,
    *,
    equipment_code: str,
    equipment_name: str,
    description: str = None,
) -> int:
    """
    Insert a new Medical Equipment entry into the catalog.

    Returns:
        equipment_id (int) of the newly created row.
    """
    _require_connection(db)
    sql = """
        INSERT INTO MedicalEquipmentCatalog
            (equipment_code, equipment_name, description)
        VALUES (%s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (equipment_code, equipment_name, description))
        db.connection.commit()
        print(f"✅ Medical equipment '{equipment_name}' inserted (equipment_code={equipment_code}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert MedicalEquipmentCatalog: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 7. OperationRooms
# ─────────────────────────────────────────────────────────────────────────────

def insert_operation_room(
    db: DBConnect,
    room_code: str,
    room_location: str,
    description: str = None,
    floor: str = None
) -> int:
    """
    Insert a new Operation Room.

    Returns:
        room_id (int) of the newly created row.
    """
    _require_connection(db)
    
    sql = "INSERT INTO OperationRooms (room_code, room_location, description, floor) VALUES (%s, %s, %s, %s);"
    
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (room_code, room_location, description, floor))
        db.connection.commit()
        print(f"✅ Operation Room '{room_code}' inserted (location: {room_location}, floor: {floor}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert OperationRoom: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 8. Appointments
#    FKs: patient_id → Patients, doctor_id → Doctors
# ─────────────────────────────────────────────────────────────────────────────

def insert_appointment(
    db: DBConnect,
    *,
    patient_id: int,
    doctor_id: int,
    appointment_time: str,   # ISO-8601: 'YYYY-MM-DD HH:MM:SS'
    status : str
) -> int:
    """
    Insert a new Appointment.

    Validates:
        - patient_id exists in Patients
        - doctor_id  exists in Doctors

    Returns:
        appointment_id (int) of the newly created row.
    """
    _require_connection(db)
    with db.connection.cursor() as cursor:
        _check_fk(cursor, "Patients", "patient_id", patient_id)
        _check_fk(cursor, "Doctors",  "doctor_id",  doctor_id)

    sql = """
        INSERT INTO Appointments
            (patient_id, doctor_id, appointment_time, status)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (patient_id, doctor_id, appointment_time, status))
        db.connection.commit()
        print(f"✅ Appointment inserted (patient_id={patient_id}, doctor_id={doctor_id}, time={appointment_time}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert Appointment: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 9. OperationScheduling
#    FKs: or_room_id, department_id, appointment_id (optional)
# ─────────────────────────────────────────────────────────────────────────────

def insert_operation_scheduling(
    db: DBConnect,
    *,
    operation_code: str,
    operation_date: str,     # 'YYYY-MM-DD'
    start_time: str,         # 'YYYY-MM-DD HH:MM:SS'
    end_time: str,           # 'YYYY-MM-DD HH:MM:SS'
    or_room_id: int,
    department_id: int,
    appointment_id: int = None,
    status: str = "Scheduled"
) -> int:
    """
    Insert a new Operation Scheduling entry.

    Validates:
        - or_room_id    exists in OperationRooms
        - department_id exists in Departments
        - appointment_id exists in Appointments (only when provided)

    Returns:
        operation_id (int) of the newly created row.
    """
    _require_connection(db)

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "OperationRooms",   "room_id",        or_room_id)
        _check_fk(cursor, "Departments",      "department_id",  department_id)
        if appointment_id is not None:
            _check_fk(cursor, "Appointments", "appointment_id", appointment_id)

    sql = """
        INSERT INTO OperationScheduling
            (operation_code, operation_date, start_time, end_time,
             or_room_id, department_id, appointment_id, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (
                operation_code, operation_date, start_time, end_time,
                or_room_id, department_id, appointment_id, status
            ))
        db.connection.commit()
        print(f"✅ Operation '{operation_code}' scheduled on {operation_date}.")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert OperationScheduling: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 10. DoctorDepartment  (M:N junction)
#     FKs: doctor_id → Doctors, department_id → Departments
# ─────────────────────────────────────────────────────────────────────────────

def insert_doctor_department(
    db: DBConnect,
    *,
    doctor_id: int,
    department_id: int,
    role: str,
) -> None:
    """
    Assign a Doctor to a Department with a given role.

    Validates:
        - doctor_id     exists in Doctors
        - department_id exists in Departments
    """
    _require_connection(db)
    with db.connection.cursor() as cursor:
        _check_fk(cursor, "Doctors",     "doctor_id",     doctor_id)
        _check_fk(cursor, "Departments", "department_id", department_id)

    sql = """
        INSERT INTO DoctorDepartment (doctor_id, department_id, role)
        VALUES (%s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (doctor_id, department_id, role))
        db.connection.commit()
        print(f"✅ Doctor (id={doctor_id}) assigned to Department (id={department_id}) as '{role}'.")
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert DoctorDepartment: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 11. NurseDepartment  (M:N junction)
#     FKs: nurse_id → Nurses, department_id → Departments
# ─────────────────────────────────────────────────────────────────────────────

def insert_nurse_department(
    db: DBConnect,
    *,
    nurse_id: int,
    department_id: int,
    role: str,
) -> None:
    """
    Assign a Nurse to a Department with a given role.

    Validates:
        - nurse_id      exists in Nurses
        - department_id exists in Departments
    """
    _require_connection(db)
    with db.connection.cursor() as cursor:
        _check_fk(cursor, "Nurses",      "nurse_id",      nurse_id)
        _check_fk(cursor, "Departments", "department_id", department_id)

    sql = """
        INSERT INTO NurseDepartment (nurse_id, department_id, role)
        VALUES (%s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (nurse_id, department_id, role))
        db.connection.commit()
        print(f"✅ Nurse (id={nurse_id}) assigned to Department (id={department_id}) as '{role}'.")
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert NurseDepartment: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 12. DoctorScheduledOperations  (M:N junction)
#     FKs: operation_id → OperationScheduling, doctor_id → Doctors
# ─────────────────────────────────────────────────────────────────────────────

def insert_doctor_scheduled_operation(
    db: DBConnect,
    *,
    operation_id: int,
    doctor_id: int,
    doctor_role: str,
) -> None:
    """
    Assign a Doctor to a scheduled Operation with a specific role.

    Validates:
        - operation_id exists in OperationScheduling
        - doctor_id    exists in Doctors
    """
    _require_connection(db)
    with db.connection.cursor() as cursor:
        _check_fk(cursor, "OperationScheduling", "operation_id", operation_id)
        _check_fk(cursor, "Doctors",             "doctor_id",    doctor_id)

    sql = """
        INSERT INTO DoctorScheduledOperations (operation_id, doctor_id, doctor_role)
        VALUES (%s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (operation_id, doctor_id, doctor_role))
        db.connection.commit()
        print(f"✅ Doctor (id={doctor_id}) assigned to Operation (id={operation_id}) as '{doctor_role}'.")
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert DoctorScheduledOperations: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 13. NurseScheduledOperations  (M:N junction)
#     FKs: operation_id → OperationScheduling, nurse_id → Nurses
# ─────────────────────────────────────────────────────────────────────────────

def insert_nurse_scheduled_operation(
    db: DBConnect,
    *,
    operation_id: int,
    nurse_id: int,
    nurse_role: str,
) -> None:
    """
    Assign a Nurse to a scheduled Operation with a specific role.

    Validates:
        - operation_id exists in OperationScheduling
        - nurse_id     exists in Nurses
    """
    _require_connection(db)
    with db.connection.cursor() as cursor:
        _check_fk(cursor, "OperationScheduling", "operation_id", operation_id)
        _check_fk(cursor, "Nurses",              "nurse_id",     nurse_id)

    sql = """
        INSERT INTO NurseScheduledOperations (operation_id, nurse_id, nurse_role)
        VALUES (%s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (operation_id, nurse_id, nurse_role))
        db.connection.commit()
        print(f"✅ Nurse (id={nurse_id}) assigned to Operation (id={operation_id}) as '{nurse_role}'.")
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert NurseScheduledOperations: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 14. DoctorShifts
#     FKs: doctor_id → Doctors
# ─────────────────────────────────────────────────────────────────────────────

def insert_doctor_shift(
    db: DBConnect,
    *,
    doctor_id: int,
    shift_date: str,         # 'YYYY-MM-DD'
    start_time: str,         # 'HH:MM:SS'
    end_time: str,           # 'HH:MM:SS'
) -> int:
    """
    Insert a Doctor Shift record.

    Validates:
        - doctor_id exists in Doctors
        - end_time  is after start_time (client-side guard before DB CHECK)

    Returns:
        shift_id (int) of the newly created row.
    """
    _require_connection(db)
    if end_time <= start_time:
        raise ValueError(f"end_time '{end_time}' must be after start_time '{start_time}'.")

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "Doctors", "doctor_id", doctor_id)

    sql = """
        INSERT INTO DoctorShifts (doctor_id, shift_date, start_time, end_time)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (doctor_id, shift_date, start_time, end_time))
        db.connection.commit()
        print(f"✅ Doctor shift inserted (doctor_id={doctor_id}, date={shift_date}, {start_time}–{end_time}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert DoctorShift: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 15. NurseShifts
#     FKs: nurse_id → Nurses
# ─────────────────────────────────────────────────────────────────────────────

def insert_nurse_shift(
    db: DBConnect,
    *,
    nurse_id: int,
    shift_date: str,         # 'YYYY-MM-DD'
    start_time: str,         # 'HH:MM:SS'
    end_time: str,           # 'HH:MM:SS'
) -> int:
    """
    Insert a Nurse Shift record.

    Validates:
        - nurse_id  exists in Nurses
        - end_time  is after start_time (client-side guard before DB CHECK)

    Returns:
        shift_id (int) of the newly created row.
    """
    _require_connection(db)
    if end_time <= start_time:
        raise ValueError(f"end_time '{end_time}' must be after start_time '{start_time}'.")

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "Nurses", "nurse_id", nurse_id)

    sql = """
        INSERT INTO NurseShifts (nurse_id, shift_date, start_time, end_time)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (nurse_id, shift_date, start_time, end_time))
        db.connection.commit()
        print(f"✅ Nurse shift inserted (nurse_id={nurse_id}, date={shift_date}, {start_time}–{end_time}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert NurseShift: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 16. ProductInventory  (lot-based)
#     FKs: product_id → MedicalProductCatalog
# ─────────────────────────────────────────────────────────────────────────────

def insert_product_inventory(
    db: DBConnect,
    *,
    product_id: int,
    lot_number: str,
    expiry_date: str,        # 'YYYY-MM-DD'
    quantity: int,
) -> int:
    """
    Insert a lot-based Product Inventory entry.

    Validates:
        - product_id exists in MedicalProductCatalog
        - quantity   >= 0

    Returns:
        inventory_id (int) of the newly created row.
    """
    _require_connection(db)
    if quantity < 0:
        raise ValueError(f"quantity must be >= 0, got {quantity}.")

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "MedicalProductCatalog", "product_id", product_id)

    sql = """
        INSERT INTO ProductInventory
            (product_id, lot_number, expiry_date, quantity)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (product_id, lot_number, expiry_date, quantity))
        db.connection.commit()
        print(f"✅ Product inventory lot '{lot_number}' inserted (product_id={product_id}, qty={quantity}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert ProductInventory: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 17. EquipmentInventory
#     FKs: equipment_id → MedicalEquipmentCatalog
# ─────────────────────────────────────────────────────────────────────────────

def insert_equipment_inventory(
    db: DBConnect,
    *,
    equipment_id: int,
    quantity: int,
) -> int:
    """
    Insert an Equipment Inventory entry.

    Validates:
        - equipment_id exists in MedicalEquipmentCatalog
        - quantity     >= 0

    Returns:
        equipment_inv_id (int) of the newly created row.
    """
    _require_connection(db)
    if quantity < 0:
        raise ValueError(f"quantity must be >= 0, got {quantity}.")

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "MedicalEquipmentCatalog", "equipment_id", equipment_id)

    sql = """
        INSERT INTO EquipmentInventory (equipment_id, quantity)
        VALUES (%s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (equipment_id, quantity))
        db.connection.commit()
        print(f"✅ Equipment inventory inserted (equipment_id={equipment_id}, qty={quantity}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert EquipmentInventory: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 18. OperationProductBooking
#     FKs: operation_id → OperationScheduling, inventory_id → ProductInventory
# ─────────────────────────────────────────────────────────────────────────────

def insert_operation_product_booking(
    db: DBConnect,
    *,
    operation_id: int,
    inventory_id: int,
    quantity_reserved: int,
    quantity_allocated: int = 0,
) -> int:
    """
    Reserve (and optionally allocate) a product lot for a scheduled operation.

    Validates:
        - operation_id  exists in OperationScheduling
        - inventory_id  exists in ProductInventory
        - quantity_reserved > 0
        - quantity_reserved does not exceed available stock in ProductInventory
        - quantity_allocated >= 0 and <= quantity_reserved

    Returns:
        booking_id (int) of the newly created row.
    """
    _require_connection(db)
    if quantity_reserved <= 0:
        raise ValueError(f"quantity_reserved must be > 0, got {quantity_reserved}.")
    if quantity_allocated < 0:
        raise ValueError(f"quantity_allocated must be >= 0, got {quantity_allocated}.")
    if quantity_allocated > quantity_reserved:
        raise ValueError(
            f"quantity_allocated ({quantity_allocated}) cannot exceed "
            f"quantity_reserved ({quantity_reserved})."
        )

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "OperationScheduling", "operation_id", operation_id)
        _check_fk(cursor, "ProductInventory",    "inventory_id", inventory_id)

        # Check available stock
        cursor.execute(
            "SELECT quantity FROM ProductInventory WHERE inventory_id = %s;",
            (inventory_id,)
        )
        row = cursor.fetchone()
        available = row["quantity"] if row else 0
        if quantity_reserved > available:
            raise ValueError(
                f"Cannot reserve {quantity_reserved} units from inventory_id={inventory_id}: "
                f"only {available} units in stock."
            )

    sql = """
        INSERT INTO OperationProductBooking
            (operation_id, inventory_id, quantity_reserved, quantity_allocated)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (operation_id, inventory_id, quantity_reserved, quantity_allocated))
        db.connection.commit()
        print(f"✅ Product booking inserted (operation_id={operation_id}, inventory_id={inventory_id}, qty_reserved={quantity_reserved}, qty_allocated={quantity_allocated}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert OperationProductBooking: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 19. OperationEquipmentBooking
#     FKs: operation_id → OperationScheduling, equipment_inv_id → EquipmentInventory
# ─────────────────────────────────────────────────────────────────────────────

def insert_operation_equipment_booking(
    db: DBConnect,
    *,
    operation_id: int,
    equipment_inv_id: int,
    quantity_reserved: int,
    quantity_allocated: int = 0,
) -> int:
    """
    Reserve (and optionally allocate) equipment units for a scheduled operation.

    Validates:
        - operation_id     exists in OperationScheduling
        - equipment_inv_id exists in EquipmentInventory
        - quantity_reserved > 0
        - quantity_reserved does not exceed available stock in EquipmentInventory
        - quantity_allocated >= 0 and <= quantity_reserved

    Returns:
        booking_id (int) of the newly created row.
    """
    _require_connection(db)
    if quantity_reserved <= 0:
        raise ValueError(f"quantity_reserved must be > 0, got {quantity_reserved}.")
    if quantity_allocated < 0:
        raise ValueError(f"quantity_allocated must be >= 0, got {quantity_allocated}.")
    if quantity_allocated > quantity_reserved:
        raise ValueError(
            f"quantity_allocated ({quantity_allocated}) cannot exceed "
            f"quantity_reserved ({quantity_reserved})."
        )

    with db.connection.cursor() as cursor:
        _check_fk(cursor, "OperationScheduling", "operation_id",     operation_id)
        _check_fk(cursor, "EquipmentInventory",  "equipment_inv_id", equipment_inv_id)

        # Check available stock
        cursor.execute(
            "SELECT quantity FROM EquipmentInventory WHERE equipment_inv_id = %s;",
            (equipment_inv_id,)
        )
        row = cursor.fetchone()
        available = row["quantity"] if row else 0
        if quantity_reserved > available:
            raise ValueError(
                f"Cannot reserve {quantity_reserved} units from equipment_inv_id={equipment_inv_id}: "
                f"only {available} units in stock."
            )

    sql = """
        INSERT INTO OperationEquipmentBooking
            (operation_id, equipment_inv_id, quantity_reserved, quantity_allocated)
        VALUES (%s, %s, %s, %s);
    """
    try:
        with db.connection.cursor() as cursor:
            cursor.execute(sql, (operation_id, equipment_inv_id, quantity_reserved, quantity_allocated))
        db.connection.commit()
        print(f"✅ Equipment booking inserted (operation_id={operation_id}, equipment_inv_id={equipment_inv_id}, qty_reserved={quantity_reserved}, qty_allocated={quantity_allocated}).")
        return cursor.lastrowid
    except pymysql.MySQLError as e:
        db.connection.rollback()
        print(f"❌ Failed to insert OperationEquipmentBooking: {e}")
        raise

"""
test_inserts.py
---------------
Inserts one dummy row into every table (in FK-safe order) to verify
that db_insert.py and the live database are both working correctly.

Run from the project root:
    python test_inserts.py
"""

from database.aws.db_connector import DBConnect
from database.aws.db_insert import (
    insert_doctor,
    insert_nurse,
    insert_department,
    insert_patient,
    insert_medical_product,
    insert_medical_equipment,
    insert_operation_room,
    insert_appointment,
    insert_operation_scheduling,
    insert_doctor_department,
    insert_nurse_department,
    insert_doctor_scheduled_operation,
    insert_nurse_scheduled_operation,
    insert_doctor_shift,
    insert_nurse_shift,
    insert_product_inventory,
    insert_equipment_inventory,
    insert_operation_product_booking,
    insert_operation_equipment_booking,
    ForeignKeyError,
)

db = DBConnect()

print("\n" + "═" * 60)
print("  HOSPITAL ER — DUMMY DATA INSERT TEST")
print("═" * 60 + "\n")

# ── 1. Doctors ────────────────────────────────────────────────
doctor_id = insert_doctor(
    db,
    doctor_code="D001",
    f_name="Alice",
    l_name="Walker",
    official_email_id="alice.walker@hospital.com",
    country_code="+44",
    phone_number="07700900001",
)

# ── 2. Nurses ─────────────────────────────────────────────────
nurse_id = insert_nurse(
    db,
    nurse_code="N001",
    f_name="Bob",
    l_name="Green",
    official_email_id="bob.green@hospital.com",
    country_code="+44",
    phone_number="07700900002",
)

# ── 3. Departments ────────────────────────────────────────────
dept_id = insert_department(
    db,
    department_code="DEP01",
    department_name="Emergency",
    specialisation="Trauma & Critical Care",
    department_email_id="emergency@hospital.com",
)

# ── 4. Patients ───────────────────────────────────────────────
patient_id = insert_patient(
    db,
    patient_code="PT001",
    f_name="Charlie",
    l_name="Brown",
    country_code="+44",
    phone_number="07700900003",
    email_id="charlie.brown@email.com",
)

# ── 5. MedicalProductCatalog ──────────────────────────────────
product_id = insert_medical_product(
    db,
    product_code="P001",
    product_name="Paracetamol 500mg",
    manufacturer="PharmaCo Ltd",
    description="Standard analgesic and antipyretic tablet.",
)

# ── 6. MedicalEquipmentCatalog ────────────────────────────────
equipment_id = insert_medical_equipment(
    db,
    equipment_code="EQ001",
    equipment_name="Defibrillator",
    description="Automated External Defibrillator (AED) for cardiac arrest.",
)

# ── 7. OperationRooms ─────────────────────────────────────────
room_id = insert_operation_room(
    db,
    room={
        "room_code": "OR01",
        "room_location": "Block A, Floor 2",
        "floor": "Floor 2",
        "description": "Primary cardiac operation theatre."
    }
)

# ── 8. Appointments ───────────────────────────────────────────
appointment_id = insert_appointment(
    db,
    patient_id=patient_id,
    doctor_id=doctor_id,
    appointment_time="2026-03-01 09:00:00",
)

# ── 9. OperationScheduling ────────────────────────────────────
operation_id = insert_operation_scheduling(
    db,
    operation_code="OP001",
    operation_date="2026-03-10",
    start_time="2026-03-10 08:00:00",
    end_time="2026-03-10 11:00:00",
    patient_id=patient_id,
    or_room_id=room_id,
    department_id=dept_id,
    appointment_id=appointment_id,
)

# ── 10. DoctorDepartment (M:N) ────────────────────────────────
insert_doctor_department(
    db,
    doctor_id=doctor_id,
    department_id=dept_id,
    role="Lead Surgeon",
)

# ── 11. NurseDepartment (M:N) ─────────────────────────────────
insert_nurse_department(
    db,
    nurse_id=nurse_id,
    department_id=dept_id,
    role="Scrub Nurse",
)

# ── 12. DoctorScheduledOperations (M:N) ──────────────────────
insert_doctor_scheduled_operation(
    db,
    operation_id=operation_id,
    doctor_id=doctor_id,
    doctor_role="Primary Surgeon",
)

# ── 13. NurseScheduledOperations (M:N) ───────────────────────
insert_nurse_scheduled_operation(
    db,
    operation_id=operation_id,
    nurse_id=nurse_id,
    nurse_role="Anaesthesia Nurse",
)

# ── 14. DoctorShifts ──────────────────────────────────────────
insert_doctor_shift(
    db,
    doctor_id=doctor_id,
    shift_date="2026-03-10",
    start_time="07:00:00",
    end_time="15:00:00",
)

# ── 15. NurseShifts ───────────────────────────────────────────
insert_nurse_shift(
    db,
    nurse_id=nurse_id,
    shift_date="2026-03-10",
    start_time="07:00:00",
    end_time="15:00:00",
)

# ── 16. ProductInventory ──────────────────────────────────────
inventory_id = insert_product_inventory(
    db,
    product_id=product_id,
    lot_number="LOT-2026-001",
    expiry_date="2028-01-31",
    quantity=500,
)

# ── 17. EquipmentInventory ────────────────────────────────────
equipment_inv_id = insert_equipment_inventory(
    db,
    equipment_id=equipment_id,
    quantity=3,
)

# ── 18. OperationProductBooking ───────────────────────────────
insert_operation_product_booking(
    db,
    operation_id=operation_id,
    inventory_id=inventory_id,
    quantity_reserved=20,
)

# ── 19. OperationEquipmentBooking ─────────────────────────────
insert_operation_equipment_booking(
    db,
    operation_id=operation_id,
    equipment_inv_id=equipment_inv_id,
    quantity_reserved=1,
)

print("\n" + "═" * 60)
print("  ALL 19 TABLES POPULATED SUCCESSFULLY ✅")
print("═" * 60 + "\n")

# ── FK validation demo ────────────────────────────────────────
print("── FK Validation Demo ──────────────────────────────────")
try:
    insert_appointment(
        db,
        patient_id=99999,   # does not exist
        doctor_id=doctor_id,
        appointment_time="2026-04-01 10:00:00",
    )
except ForeignKeyError as e:
    print(f"✅ ForeignKeyError caught as expected:\n   {e}")

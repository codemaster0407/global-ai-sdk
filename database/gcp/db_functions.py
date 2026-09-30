'''
Boilerplate for working with the Cloud SQL database: inspect, create/alter/drop tables,
columns, foreign keys and indexes, and insert/select/update/delete rows.

Every function takes an optional ``engine``; when omitted, one shared engine from
``connect_with_connector()`` is created on first use and reused (one pool per process).

Schema changes go through Alembic's operations API, which emits the right DDL per
dialect (e.g. SQL Server's ``sp_rename`` and dropping a column's default constraint
before the column). Row values are always sent as bound parameters.

    from sqlalchemy import Column, Integer, String
    from database.gcp import db_functions as db

    db.create_table("demo_users", Column("id", Integer, primary_key=True), Column("name", String(100)))
    db.insert_rows("demo_users", [{"name": "Asha"}, {"name": "Ravi"}])
    db.update_rows("demo_users", {"name": "Asha K"}, where={"id": 1})
    print(db.select_rows("demo_users", order_by="id"))
'''

from functools import lru_cache
from typing import Any, Iterable, Sequence

import sqlalchemy as sqla
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import Engine

from database.gcp.cloud_sql_connector import connect_with_connector

# SQL Server caps one statement at 2100 parameters and a VALUES list at 1000 rows
MAX_PARAMS_PER_STATEMENT = 2000
MAX_ROWS_PER_STATEMENT = 1000


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_engine() -> Engine:
    '''
        Shared engine, created on first use. Reuse it; each new engine opens a new connector and pool.
    '''
    return connect_with_connector()


def _engine(engine: Engine | None) -> Engine:
    return engine or get_engine()


def _run_ddl(engine: Engine | None, operation) -> None:
    '''
        Run ``operation(op)`` with an Alembic ``Operations`` object inside one transaction.
    '''
    with _engine(engine).begin() as conn:
        operation(Operations(MigrationContext.configure(conn)))


# ---------------------------------------------------------------------------
# Inspecting
# ---------------------------------------------------------------------------
def list_tables(schema: str | None = None, engine: Engine | None = None) -> list[str]:
    '''
        Table names in ``schema`` (default schema, ``dbo`` on SQL Server, when omitted).
    '''
    return sqla.inspect(_engine(engine)).get_table_names(schema=schema)


def table_exists(table_name: str, schema: str | None = None, engine: Engine | None = None) -> bool:
    return sqla.inspect(_engine(engine)).has_table(table_name, schema=schema)


def describe_table(table_name: str, schema: str | None = None, engine: Engine | None = None) -> dict:
    '''
        Columns, primary key, foreign keys, unique constraints and indexes of a table.
    '''
    inspector = sqla.inspect(_engine(engine))
    indexes = inspector.get_indexes(table_name, schema=schema)
    try:
        unique_constraints = inspector.get_unique_constraints(table_name, schema=schema)
    except NotImplementedError:
        # SQL Server's dialect reports unique constraints as unique indexes instead
        unique_constraints = [
            {"name": i["name"], "column_names": i["column_names"]} for i in indexes if i.get("unique")
        ]
    return {
        "columns": [
            {"name": c["name"], "type": str(c["type"]), "nullable": c["nullable"], "default": c.get("default")}
            for c in inspector.get_columns(table_name, schema=schema)
        ],
        "primary_key": inspector.get_pk_constraint(table_name, schema=schema),
        "foreign_keys": inspector.get_foreign_keys(table_name, schema=schema),
        "unique_constraints": unique_constraints,
        "indexes": indexes,
    }


def get_table(table_name: str, schema: str | None = None, engine: Engine | None = None) -> sqla.Table:
    '''
        Reflect an existing table so it can be used in SQLAlchemy queries.
    '''
    return sqla.Table(table_name, sqla.MetaData(), schema=schema, autoload_with=_engine(engine))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def create_table(
    table_name: str,
    *columns: sqla.Column | sqla.Constraint | sqla.Index,
    schema: str | None = None,
    if_not_exists: bool = True,
    engine: Engine | None = None,
) -> sqla.Table:
    '''
        Create a table from SQLAlchemy ``Column`` / constraint objects, e.g.
        ``create_table("demo", Column("id", Integer, primary_key=True), Column("name", String(100)))``.
    '''
    table = sqla.Table(table_name, sqla.MetaData(), *columns, schema=schema)
    table.create(_engine(engine), checkfirst=if_not_exists)
    return table


def drop_table(table_name: str, schema: str | None = None, if_exists: bool = True, engine: Engine | None = None) -> None:
    '''
        Drop a table and all its data. Fails if another table's foreign key still references it.
    '''
    table = sqla.Table(table_name, sqla.MetaData(), schema=schema)
    table.drop(_engine(engine), checkfirst=if_exists)


def rename_table(old_name: str, new_name: str, schema: str | None = None, engine: Engine | None = None) -> None:
    _run_ddl(engine, lambda op: op.rename_table(old_name, new_name, schema=schema))


def truncate_table(table_name: str, schema: str | None = None, engine: Engine | None = None) -> None:
    '''
        Remove all rows quickly. SQL Server refuses TRUNCATE on a table referenced by a foreign key;
        use ``delete_rows(..., allow_all=True)`` for those.
    '''
    engine = _engine(engine)
    name = engine.dialect.identifier_preparer.format_table(sqla.Table(table_name, sqla.MetaData(), schema=schema))
    with engine.begin() as conn:
        conn.execute(sqla.text(f"TRUNCATE TABLE {name}"))


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
def add_column(table_name: str, column: sqla.Column, schema: str | None = None, engine: Engine | None = None) -> None:
    '''
        e.g. ``add_column("demo", Column("email", String(255), nullable=True))``.
        A NOT NULL column on a table with rows needs a ``server_default``.
    '''
    _run_ddl(engine, lambda op: op.add_column(table_name, column, schema=schema))


def drop_column(table_name: str, column_name: str, schema: str | None = None, engine: Engine | None = None) -> None:
    '''
        Drop a column. On SQL Server, its default, check and foreign key constraints are dropped first.
    '''
    _run_ddl(engine, lambda op: op.drop_column(
        table_name, column_name, schema=schema,
        mssql_drop_default=True, mssql_drop_check=True, mssql_drop_foreign_key=True,
    ))


def rename_column(table_name: str, old_name: str, new_name: str, schema: str | None = None, engine: Engine | None = None) -> None:
    _run_ddl(engine, lambda op: op.alter_column(table_name, old_name, new_column_name=new_name, schema=schema))


def alter_column(
    table_name: str,
    column_name: str,
    type_: sqla.types.TypeEngine | None = None,
    nullable: bool | None = None,
    server_default: str | None | bool = False,
    schema: str | None = None,
    engine: Engine | None = None,
) -> None:
    '''
        Change a column's type, nullability and/or default. Only the arguments you pass are changed.
        ``server_default=None`` removes the default; the default ``False`` leaves it alone.
        The column's current definition is read first: SQL Server's ALTER COLUMN restates the
        whole column, so without it a type change would silently make a NOT NULL column nullable.
    '''
    columns = sqla.inspect(_engine(engine)).get_columns(table_name, schema=schema)
    existing = next((c for c in columns if c["name"] == column_name), None)
    if existing is None:
        raise KeyError(f"Column {column_name!r} not in table {table_name!r}")
    _run_ddl(engine, lambda op: op.alter_column(
        table_name, column_name, type_=type_, nullable=nullable, server_default=server_default,
        existing_type=existing["type"], existing_nullable=existing["nullable"],
        existing_server_default=existing.get("default"), schema=schema,
    ))


# ---------------------------------------------------------------------------
# Keys, constraints and indexes
# ---------------------------------------------------------------------------
def add_foreign_key(
    constraint_name: str,
    source_table: str,
    referent_table: str,
    local_cols: Sequence[str],
    remote_cols: Sequence[str],
    ondelete: str | None = None,
    onupdate: str | None = None,
    schema: str | None = None,
    engine: Engine | None = None,
) -> None:
    '''
        e.g. ``add_foreign_key("fk_orders_user", "Orders", "Users", ["user_id"], ["id"], ondelete="CASCADE")``.
        ``ondelete`` / ``onupdate``: "CASCADE", "SET NULL", "NO ACTION" (SQL Server has no "RESTRICT").
    '''
    _run_ddl(engine, lambda op: op.create_foreign_key(
        constraint_name, source_table, referent_table, list(local_cols), list(remote_cols),
        ondelete=ondelete, onupdate=onupdate, source_schema=schema, referent_schema=schema,
    ))


def drop_foreign_key(constraint_name: str, table_name: str, schema: str | None = None, engine: Engine | None = None) -> None:
    drop_constraint(constraint_name, table_name, type_="foreignkey", schema=schema, engine=engine)


def add_unique_constraint(constraint_name: str, table_name: str, columns: Sequence[str], schema: str | None = None, engine: Engine | None = None) -> None:
    _run_ddl(engine, lambda op: op.create_unique_constraint(constraint_name, table_name, list(columns), schema=schema))


def add_primary_key(constraint_name: str, table_name: str, columns: Sequence[str], schema: str | None = None, engine: Engine | None = None) -> None:
    '''
        Add a primary key to a table that has none. The columns must be NOT NULL.
    '''
    _run_ddl(engine, lambda op: op.create_primary_key(constraint_name, table_name, list(columns), schema=schema))


def drop_constraint(
    constraint_name: str,
    table_name: str,
    type_: str | None = None,
    schema: str | None = None,
    engine: Engine | None = None,
) -> None:
    '''
        Drop any named constraint. ``type_``: "foreignkey", "primary", "unique" or "check"
        (needed on MySQL; optional on SQL Server). Names are listed by ``describe_table``.
    '''
    _run_ddl(engine, lambda op: op.drop_constraint(constraint_name, table_name, type_=type_, schema=schema))


def create_index(index_name: str, table_name: str, columns: Sequence[str], unique: bool = False, schema: str | None = None, engine: Engine | None = None) -> None:
    _run_ddl(engine, lambda op: op.create_index(index_name, table_name, list(columns), unique=unique, schema=schema))


def drop_index(index_name: str, table_name: str, schema: str | None = None, engine: Engine | None = None) -> None:
    _run_ddl(engine, lambda op: op.drop_index(index_name, table_name=table_name, schema=schema))


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def _where_clause(table: sqla.Table, where: dict[str, Any] | None):
    '''
        ``{"col": value}`` -> col = value; a list/tuple/set -> col IN (...); None -> col IS NULL.
    '''
    conditions = []
    for name, value in (where or {}).items():
        if name not in table.c:
            raise KeyError(f"Column {name!r} not in table {table.name!r}")
        column = table.c[name]
        if value is None:
            conditions.append(column.is_(None))
        elif isinstance(value, (list, tuple, set)):
            conditions.append(column.in_(list(value)))
        else:
            conditions.append(column == value)
    return sqla.and_(*conditions) if conditions else None


def _chunks(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def insert_rows(table_name: str, rows: Sequence[dict[str, Any]], schema: str | None = None, engine: Engine | None = None) -> int:
    '''
        Insert many rows in one transaction, as multi-row INSERTs sized to SQL Server's limits.
        Every row must have the same keys. Returns the number of rows inserted.
    '''
    rows = list(rows)
    if not rows:
        return 0
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("All rows must have the same keys")

    engine = _engine(engine)
    table = get_table(table_name, schema=schema, engine=engine)
    chunk_size = max(1, min(MAX_ROWS_PER_STATEMENT, MAX_PARAMS_PER_STATEMENT // len(keys)))
    with engine.begin() as conn:
        for chunk in _chunks(rows, chunk_size):
            conn.execute(table.insert().values(chunk))
    return len(rows)


def select_rows(
    table_name: str,
    where: dict[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
    limit: int | None = None,
    schema: str | None = None,
    engine: Engine | None = None,
) -> list[dict]:
    '''
        Rows as dicts. ``order_by`` accepts "col" or "-col" (descending), or a list of them.
    '''
    engine = _engine(engine)
    table = get_table(table_name, schema=schema, engine=engine)
    query = sqla.select(*(table.c[c] for c in columns)) if columns else sqla.select(table)

    clause = _where_clause(table, where)
    if clause is not None:
        query = query.where(clause)
    for name in ([order_by] if isinstance(order_by, str) else order_by or []):
        query = query.order_by(table.c[name[1:]].desc() if name.startswith("-") else table.c[name])
    if limit is not None:
        query = query.limit(limit)

    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(query)]


def count_rows(table_name: str, where: dict[str, Any] | None = None, schema: str | None = None, engine: Engine | None = None) -> int:
    engine = _engine(engine)
    table = get_table(table_name, schema=schema, engine=engine)
    query = sqla.select(sqla.func.count()).select_from(table)
    clause = _where_clause(table, where)
    if clause is not None:
        query = query.where(clause)
    with engine.connect() as conn:
        return conn.execute(query).scalar_one()


def update_rows(
    table_name: str,
    values: dict[str, Any],
    where: dict[str, Any] | None,
    allow_all: bool = False,
    schema: str | None = None,
    engine: Engine | None = None,
) -> int:
    '''
        Set ``values`` on rows matching ``where``. Returns the number of rows changed.
        An empty ``where`` updates every row, so it requires ``allow_all=True``.
    '''
    engine = _engine(engine)
    table = get_table(table_name, schema=schema, engine=engine)
    clause = _where_clause(table, where)
    if clause is None and not allow_all:
        raise ValueError("update_rows without `where` changes every row; pass allow_all=True to confirm")
    query = table.update().values(**values)
    if clause is not None:
        query = query.where(clause)
    with engine.begin() as conn:
        return conn.execute(query).rowcount


def delete_rows(
    table_name: str,
    where: dict[str, Any] | None,
    allow_all: bool = False,
    schema: str | None = None,
    engine: Engine | None = None,
) -> int:
    '''
        Delete rows matching ``where``. Returns the number of rows deleted.
        An empty ``where`` deletes every row, so it requires ``allow_all=True``.
    '''
    engine = _engine(engine)
    table = get_table(table_name, schema=schema, engine=engine)
    clause = _where_clause(table, where)
    if clause is None and not allow_all:
        raise ValueError("delete_rows without `where` deletes every row; pass allow_all=True to confirm")
    query = table.delete()
    if clause is not None:
        query = query.where(clause)
    with engine.begin() as conn:
        return conn.execute(query).rowcount


def run_sql(sql: str, params: dict[str, Any] | None = None, engine: Engine | None = None) -> list[dict]:
    '''
        Run raw SQL in a transaction, with values bound as ``:name`` parameters, e.g.
        ``run_sql("SELECT * FROM Users WHERE id = :id", {"id": 1})``. Never format values into ``sql``.
        Returns rows for queries that produce them, else an empty list.
    '''
    with _engine(engine).begin() as conn:
        result = conn.execute(sqla.text(sql), params or {})
        return [dict(row._mapping) for row in result] if result.returns_rows else []

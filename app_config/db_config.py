from dataclasses import dataclass, field
from functools import cached_property, partial
from io import StringIO

import pandas as pd
import sqlalchemy as sa

PRETTY_LINE = "====================================================="


@dataclass
class DBConfig:
    engine_expr: str  # it is expected that this expression will define a `engine` variable which will be of type `sa.Engine`
    table_stats_queries: list[str] = field(default_factory=list)

    @cached_property
    def sa_engine(self) -> sa.Engine:
        ret = {}
        exec(self.engine_expr, {}, ret)
        return ret["engine"]

    @cached_property
    def sa_metadata(self) -> sa.MetaData:
        metadata = sa.MetaData()
        metadata.reflect(bind=self.sa_engine)
        return metadata

    @cached_property
    def schema(self) -> str:
        buff = StringIO()
        print_buff = partial(print, file=buff)
        inspector = sa.inspect(self.sa_engine)

        # Get all table names
        table_names = inspector.get_table_names()
        # Iterate through each table to retrieve detailed schema information
        for table_name in table_names:
            print_buff(PRETTY_LINE)
            print_buff(f"SCHEMA DETAILS FOR TABLE: {table_name}")
            print_buff(PRETTY_LINE)

            # 1. Get Column Information (Name, Type, Nullability)
            columns = inspector.get_columns(table_name)
            print_buff(f"\n--- Columns ({len(columns)}) ---")
            for col in columns:
                print_buff(
                    f"  Name: {col['name']:<20} Type: {str(col['type']):<15} Nullable: {col['nullable']}"
                )

            # 2. Get Primary Key Information
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_columns = pk_constraint.get("constrained_columns", [])
            print_buff(f"\n--- Primary Key ({len(pk_columns)}) ---")
            if pk_columns:
                print_buff(f"  Columns: {', '.join(pk_columns)}")
            else:
                print_buff("  No primary key found.")

            # 3. Get Index Information
            indexes = inspector.get_indexes(table_name)
            print_buff(f"\n--- Indexes ({len(indexes)}) ---")
            if indexes:
                for index in indexes:
                    index_col_names = ", ".join(index["column_names"])  # type: ignore
                    print_buff(
                        f"  Name: {index['name']:<25} Columns: {index_col_names:<25} Unique: {index['unique']}"
                    )
            else:
                print_buff("  No indexes found.")

            # 4. Get Foreign Key Relationships
            foreign_keys = inspector.get_foreign_keys(table_name)
            print_buff(f"\n--- Foreign Keys ({len(foreign_keys)}) ---")
            if foreign_keys:
                for fk in foreign_keys:
                    print_buff(f"  Constraint Name: {fk['name']}")
                    print_buff(f"    Constrains Column(s): {fk['constrained_columns']}")
                    print_buff(f"    Refers To Table:      {fk['referred_table']}")
                    print_buff(f"    Refers To Column(s):  {fk['referred_columns']}")
                    print_buff(
                        f"    On Delete/Update:     {fk.get('ondelete')}/{fk.get('onupdate')}"
                    )
            else:
                print_buff("  No foreign keys found.")

            print_buff()

        buff.seek(0)
        return buff.read()

    @cached_property
    def table_stats(self) -> str:
        buff = StringIO()
        print_buff = partial(print, file=buff)
        with self.sa_engine.connect() as conn:
            for table_stat_query in self.table_stats_queries:
                print_buff(PRETTY_LINE)
                print_buff(table_stat_query)
                print_buff(PRETTY_LINE)
                df = pd.read_sql(table_stat_query, con=conn)
                df.to_markdown(buf=buff, index=False)
                print_buff("\n\n")
        buff.seek(0)
        return buff.read()

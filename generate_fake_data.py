import logging
import os
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
from faker import Faker
from faker.providers.company.en_PH import Provider as Company_en_PH_Provider

SQLITE3_FILE = "./data.db"
SQLITE3_SCHEMA_FILE = "./schema.sql"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(name)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(
        self,
        *,
        num_products: int,
        num_regions: int,
        num_customers: int,
        num_sales: int,
        sales_start_date: date,
        sales_end_date: date,
        num_transactions: int,
        max_units_per_transaction: int,
    ):
        self.num_products = num_products
        self.num_regions = num_regions
        self.num_customers = num_customers
        self.num_sales = num_sales
        self.sales_start_date = sales_start_date
        self.sales_end_date = sales_end_date
        self.num_transactions = num_transactions
        self.max_units_per_transaction = max_units_per_transaction

        self.faker = Faker()
        self.faker.add_provider(Company_en_PH_Provider)

    def generate_products_data(self) -> pd.DataFrame:
        products = []
        for _ in range(self.num_products):
            product_id = len(products) + 1
            category = self.faker.random_company_product()
            product_name = self.faker.company() + " - " + category
            unit_price = round(self.faker.random.uniform(10, 1000), 2)
            products.append([product_id, product_name, unit_price, category])
        return pd.DataFrame(
            products, columns=["product_id", "product_name", "unit_price", "category"]
        )

    def generate_region_data(self) -> pd.DataFrame:
        regions = []
        for _ in range(self.num_regions):
            region_id = len(regions) + 1
            region_name = self.faker.city() + " Region"
            regions.append([region_id, region_name])
        return pd.DataFrame(regions, columns=["region_id", "region_name"])

    def generate_customers_data(self) -> pd.DataFrame:
        customers = []
        for _ in range(self.num_customers):
            customer_id = len(customers) + 1
            customer_name = self.faker.name()
            region_id = self.faker.random_int(min=1, max=self.num_regions)
            customers.append([customer_id, customer_name, region_id])
        return pd.DataFrame(
            customers, columns=["customer_id", "customer_name", "region_id"]
        )

    def generate_sales_data(self) -> pd.DataFrame:
        sales = []
        for _ in range(self.num_sales):
            sale_id = len(sales) + 1
            customer_id = self.faker.random_int(min=1, max=self.num_customers)
            region_id = self.faker.random_int(min=1, max=self.num_regions)
            sale_date = self.faker.date_between(
                self.sales_start_date, self.sales_end_date
            )
            sales.append(
                [
                    sale_id,
                    customer_id,
                    region_id,
                    sale_date,
                ]
            )
        return pd.DataFrame(
            sales,
            columns=[
                "sale_id",
                "customer_id",
                "region_id",
                "sale_date",
            ],
        )

    def generate_transactions_data(self) -> pd.DataFrame:
        transactions = []
        for _ in range(self.num_transactions):
            transaction_id = len(transactions) + 1
            sale_id = self.faker.random_int(min=1, max=self.num_sales)
            product_id = self.faker.random_int(min=1, max=self.num_products)
            units = self.faker.random_int(min=1, max=self.max_units_per_transaction)
            transactions.append(
                [
                    transaction_id,
                    sale_id,
                    product_id,
                    units,
                ]
            )
        return pd.DataFrame(
            transactions,
            columns=[
                "transaction_id",
                "sale_id",
                "product_id",
                "units",
            ],
        )


def main():
    data_generator = DataGenerator(
        num_products=1000,
        num_regions=10,
        num_customers=1000,
        num_sales=10_000,
        sales_start_date=date(year=2023, month=1, day=1),
        sales_end_date=date(year=2025, month=9, day=30),
        num_transactions=100_000,
        max_units_per_transaction=10,
    )
    logger.info("Generating Random Data...")
    data_generator.faker.seed_instance(seed=6969)
    products_df = data_generator.generate_products_data()
    regions_df = data_generator.generate_region_data()
    customers_df = data_generator.generate_customers_data()
    sales_df = data_generator.generate_sales_data()
    transactions_df = data_generator.generate_transactions_data()
    logger.info("Random Data Generated")

    logger.info(
        f"Loading to SqLite3 ({SQLITE3_FILE})...\nNote that the existing data will be replaced."
    )
    if os.path.exists(SQLITE3_FILE):
        logger.info(
            f"Sqlite file: {SQLITE3_FILE} already existing, removing it before proceeding..."
        )
        os.remove(SQLITE3_FILE)
        logger.info(f"Sqlite file: {SQLITE3_FILE} removed!")
    logger.info(
        f"Creating the file: {SQLITE3_FILE}, and creating schema using: {SQLITE3_SCHEMA_FILE}..."
    )
    with sqlite3.connect(SQLITE3_FILE) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        schema_ddl = Path(SQLITE3_SCHEMA_FILE).read_text()
        logger.info(f"Creating Schema using: {SQLITE3_SCHEMA_FILE} file...")
        conn.executescript(schema_ddl)
        logger.info(f"Schema created using: {SQLITE3_SCHEMA_FILE} file!")
    logger.info("Loading data...")
    with sqlite3.connect(SQLITE3_FILE) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        products_df.to_sql(
            "products",
            if_exists="append",
            con=conn,
            index=False,
        )
        regions_df.to_sql(
            "regions",
            if_exists="append",
            con=conn,
            index=False,
        )
        customers_df.to_sql(
            "customers",
            if_exists="append",
            con=conn,
            index=False,
        )
        sales_df.to_sql(
            "sales",
            if_exists="append",
            con=conn,
            index=False,
        )
        transactions_df.to_sql(
            "transactions",
            if_exists="append",
            con=conn,
            index=False,
        )
        logger.info("Done")


if __name__ == "__main__":
    main()

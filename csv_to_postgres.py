#!/usr/bin/env python3
"""
Utility to load evaluation CSVs (image_eval_embed.py or stock_photo_evaluator.py)
into a PostgreSQL table.
"""

import argparse
import csv
import logging
import os
import re
from typing import List, Tuple

import psycopg2
from psycopg2 import sql


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_column_name(name: str) -> str:
    name = name.strip().lower().replace(' ', '_')
    name = re.sub(r'[^a-z0-9_]', '_', name)
    if not name or name[0].isdigit():
        name = f"col_{name}"
    return name


def create_table_if_needed(conn, table: str, columns: List[str]) -> None:
    with conn.cursor() as cur:
        cols = [
            sql.SQL("{} TEXT").format(sql.Identifier(col))
            for col in columns
        ]
        query = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "id SERIAL PRIMARY KEY, "
            "{}"
            ");"
        ).format(
            sql.Identifier(table),
            sql.SQL(', ').join(cols)
        )
        cur.execute(query)
    conn.commit()


def insert_rows(conn, table: str, columns: List[str], rows: List[List[str]]) -> None:
    placeholders = sql.SQL(', ').join(sql.Placeholder() * len(columns))
    insert_stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table),
        sql.SQL(', ').join(map(sql.Identifier, columns)),
        placeholders
    )

    with conn.cursor() as cur:
        cur.executemany(insert_stmt, rows)
    conn.commit()


def load_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        columns = [normalize_column_name(col) for col in header]
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        return columns, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load evaluation CSV into PostgreSQL")
    parser.add_argument(
        'csv_path',
        help="Path to CSV produced by image_eval_embed.py or stock_photo_evaluator.py"
    )
    parser.add_argument(
        '--db-url',
        default=os.environ.get("DATABASE_URL", "postgres://postgres@localhost:5432/photos"),
        help="PostgreSQL connection URL (default: DATABASE_URL env var or postgres://postgres@localhost:5432/photos)"
    )
    parser.add_argument(
        '--table',
        default="image_evaluation_results",
        help="Target table name (default: %(default)s)"
    )
    parser.add_argument(
        '--truncate',
        action='store_true',
        help="Truncate the target table before inserting"
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        logger.error("CSV file does not exist: %s", args.csv_path)
        return

    columns, rows = load_csv(args.csv_path)

    if not columns:
        logger.error("CSV header could not be parsed")
        return

    if not rows:
        logger.warning("CSV contains no data rows; nothing to insert")
        return

    logger.info("Connecting to %s", args.db_url)
    try:
        with psycopg2.connect(args.db_url) as conn:
            if args.truncate:
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(args.table)))
                conn.commit()
                logger.info("Truncated table %s", args.table)

            create_table_if_needed(conn, args.table, columns)
            insert_rows(conn, args.table, columns, rows)

            logger.info("Inserted %d rows into %s.%s", len(rows), conn.get_dsn_parameters().get("dbname"), args.table)
    except psycopg2.OperationalError as e:
        logger.error("Failed to connect to database: %s", e)
        return


if __name__ == "__main__":
    main()

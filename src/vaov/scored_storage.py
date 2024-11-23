import sqlite3
from typing import List, Tuple, Any, Dict
import os


class ScoredStorage:
    def __init__(self, db_path: os.PathLike, num_params: int, max_rows_per_key: int):
        self.db_path = db_path
        self.num_params = num_params
        self.max_rows_per_key = max_rows_per_key

        # Ensure the database connection is optimized
        self.connection = sqlite3.connect(self.db_path, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging
        self.connection.execute("PRAGMA synchronous=NORMAL;")  # Reduce durability for better performance
        self.connection.execute("PRAGMA mmap_size=30000000000;")  # Use memory-mapped I/O
        self.connection.execute("PRAGMA cache_size=-16000;")  # Use ~16MB of cache
        self.connection.execute("PRAGMA page_size=65536;")  # Increase page size to 64KB

        # Dynamically create table columns based on num_params
        param_columns = ", ".join([f"param{i} REAL NOT NULL" for i in range(1, num_params + 1)])
        param_index = ", ".join([f"param{i}" for i in range(1, num_params + 1)])

        with self.connection:
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scored_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_name INTEGER NOT NULL,
                    {param_columns},
                    score REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(key_name, {param_index})
                )
                """
            )
            self.connection.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS enforce_top_k_per_key
                AFTER INSERT ON scored_data
                BEGIN
                    DELETE FROM scored_data
                    WHERE id IN (
                        SELECT id
                        FROM (
                            SELECT id, ROW_NUMBER() OVER (
                                PARTITION BY key_name
                                ORDER BY score DESC, created_at DESC
                            ) AS rank
                            FROM scored_data
                            WHERE key_name = NEW.key_name
                        )
                        WHERE rank > {self.max_rows_per_key}
                    );
                END;
                """
            )

    def insert_many(self, entries: List[Tuple[int, Tuple[Any, ...], float]]):
        """
        Insert multiple entries into the database at once.

        Args:
            entries: A list of tuples where each tuple contains (key, params, score).

        Raises:
            ValueError: If any tuple has an incorrect number of parameters.
        """
        if not entries:
            return  # No entries to insert

        # Validate that all entries have the correct number of parameters
        for _, params, _ in entries:
            if len(params) != self.num_params:
                raise ValueError("Incorrect number of parameters provided.")

        # Prepare values for batch insertion
        values = [
            (key, *params, score)
            for key, params, score in entries
        ]

        with self.connection:
            cursor = self.connection.cursor()
            try:
                # Use executemany for batch insertion
                cursor.executemany(
                    f"""
                    INSERT INTO scored_data (key_name, {", ".join([f"param{i}" for i in range(1, self.num_params + 1)])}, score)
                    VALUES ({", ".join(["?"] * (self.num_params + 2))})
                    """,
                    values,
                )
            except sqlite3.IntegrityError:
                # Handle duplicates silently by ignoring them
                self.connection.rollback()

    def get_rows(self, key: int) -> List[Tuple[Tuple[Any, ...], float]]:
        cursor = self.connection.cursor()
        rows = cursor.execute(
            f"""
            SELECT {", ".join([f"param{i}" for i in range(1, self.num_params + 1)])}, score
            FROM scored_data
            WHERE key_name = ?
            ORDER BY score DESC, created_at DESC
            """,
            (key,),
        ).fetchall()

        return [(tuple(row[:-1]), row[-1]) for row in rows]

    def key_counts(self) -> Dict[int, int]:
        cursor = self.connection.cursor()
        counts = cursor.execute(
            """
            SELECT key_name, COUNT(*)
            FROM scored_data
            GROUP BY key_name
            """
        ).fetchall()

        return {k: count for k, count in counts}

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()

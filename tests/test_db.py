import unittest
import os
import sqlite3

from fasthtml.components import S
from src.vaov.scored_storage import ScoredStorage
import random

class TestScoredStorage(unittest.TestCase):
    DB_PATH = 'test_scored_storage.db'

    def setUp(self):
        # Remove the database file if it exists before each test
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def tearDown(self):
        # Clean up by removing the database file after each test
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def test_insert_many(self):
        storage = ScoredStorage(self.DB_PATH, num_params=2, max_rows_per_key=5)
        entries = [
            (1, (10, 20), 100.0),
            (1, (15, 25), 95.0),
            (1, (20, 30), 90.0)
        ]

        # Perform the batch insertion
        storage.insert_many(entries)

        # Verify the content in the database
        rows = storage.get_rows(1)
        self.assertEqual(rows, [
            ((10, 20), 100.0),
            ((15, 25), 95.0),
            ((20, 30), 90.0)
        ])

    def test_insert_many_over_max_rows(self):
        storage = ScoredStorage(self.DB_PATH, num_params=2, max_rows_per_key=3)
        entries = [
            (1, (10, 20), 100.0),
            (1, (15, 25), 90.0),
            (1, (20, 30), 80.0),
            (1, (25, 35), 70.0)  # This should not be inserted due to max rows
        ]

        # Perform the batch insertion
        storage.insert_many(entries)

        # Verify only top 3 rows remain
        rows = storage.get_rows(1)
        self.assertEqual(rows, [
            ((10, 20), 100.0),
            ((15, 25), 90.0),
            ((20, 30), 80.0)
        ])

    def test_insert_many_over_max_rows_randomized(self):
        random.seed(5)
        for _ in range(128):
            max_rows = random.randint(1, 100)
            storage = ScoredStorage(self.DB_PATH, num_params=2, max_rows_per_key=max_rows)
            num_entries = random.randint(5, 200)
            entries = [
                (1, (random.randint(0, 100), random.randint(0, 100)), random.randint(0, 100) * 1.0)
                for _ in range(num_entries)
            ]

            # Perform the batch insertion
            storage.insert_many(entries)

            # Verify only top 3 rows remain
            rows = storage.get_rows(1)
            expected = [x[1:] for x in sorted(entries, key=lambda x: x[-1], reverse=True)[:max_rows]]
            rows = sorted(rows, key=lambda x: x[-1], reverse=True)
            self.assertEqual(rows, expected)
            del storage

if __name__ == '__main__':
    unittest.main()

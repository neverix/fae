from typing import List, Tuple, Any, Dict
import numpy as np
import os


class ScoredStorage:
    def __init__(
        self,
        db_path: os.PathLike, num_params: int, max_rows_per_key: int,
        mode: str = "w+"
    ):
        self.db_path = db_path
        self.num_params = num_params
        self.max_rows_per_key = max_rows_per_key
        self.mode = mode

        self.db = np.memmap(
            db_path,
            dtype=np.uint32,
            mode=mode,
            shape=(1, self.max_rows_per_key + 1, self.entry_len)
        )
        self.db[:] = 0

    @property
    def entry_len(self):
        return 1 + self.num_params

    @property
    def record_len(self):
        return 1 + self.entry_len * self.max_rows_per_key

    def to_st(self, i):
        return 1 + (i - 1) * self.entry_len

    def fr_st(self, i):
        return 1 + (i - 1) // self.entry_len

    def insert_many(self, entries: List[Tuple[int, Tuple[Any, ...], float]]):
        """
        Insert multiple entries into the database at once.

        Args:
            entries: A list of tuples where each tuple contains (key, params, score).

        Raises:
            ValueError: If any tuple has an incorrect number of parameters.
        """
        if "w" not in self.mode:
            raise ValueError("Database is not open for writing.")
        for key, params, score in entries:
            if key >= self.db.shape[0]:
                self.db.flush()
                new_size = (key + 1,) + self.db.shape[1:]
                del self.db
                self.db = np.memmap(
                    self.db_path,
                    dtype=np.uint32,
                    mode=self.mode,
                    shape=new_size
                )
            num_entries = self.db[key, 0, 0]
            if num_entries < self.max_rows_per_key:
                start_index = 1 + num_entries
                self.db[key, start_index, 0] = np.array(score, dtype=np.float32).view(np.uint32)
                self.db[key, start_index, 1:] = np.array(params, dtype=np.uint32)
                self.db[key, 0, 0] += 1
            else:
                if score < self.db[key, 1, 0].view(np.float32):
                    continue
                self.db[key, 1], self.db[key, -1] = self.db[key, -1], self.db[key, 1]
                self.db[key, -1, 0] = np.array(score, dtype=np.float32).view(np.uint32)
                self.db[key, -1, 1:] = np.array(params, dtype=np.uint32)
                current_idx = 1
                # add 1 for s9ze
                # subtract one for 0-indexing
                # subtract one for invalid entry (old min)
                last_valid_entry = (((self.max_rows_per_key + 1) - 1) - 1)
                while current_idx * 2 <= last_valid_entry:
                    left_child_idx = current_idx * 2
                    right_child_idx = left_child_idx + 1
                    current_score = self.db[key, current_idx, 0].view(np.float32)
                    left_score = self.db[key, left_child_idx, 0].view(np.float32)
                    if right_child_idx > last_valid_entry:
                        if current_score > left_score:
                            self.db[key, current_idx], self.db[key, left_child_idx] = self.db[key, left_child_idx], self.db[key, current_idx]
                        break
                    right_score = self.db[key, right_child_idx, 0].view(np.float32)
                    if current_score < left_score and current_score < right_score:
                        break
                    if left_score < right_score:
                        self.db[key, current_idx], self.db[key, left_child_idx] = self.db[key, left_child_idx], self.db[key, current_idx]
                        current_idx = left_child_idx
                    else:
                        self.db[key, current_idx], self.db[key, right_child_idx] = self.db[key, right_child_idx], self.db[key, current_idx]
                        current_idx = right_child_idx
                self.db[key, -1, 0] = np.array(score, dtype=np.float32).view(np.uint32)
                self.db[key, -1, 1:] = np.array(params, dtype=np.uint32)
                current_idx = self.max_rows_per_key
                while current_idx // 2 > 0:
                    parent_idx = current_idx // 2
                    if self.db[key, current_idx, 0] < self.db[key, parent_idx, 0]:
                        break
                    self.db[key, current_idx], self.db[key, parent_idx] = self.db[key, parent_idx], self.db[key, current_idx]
                    current_idx = parent_idx
        self.db.flush()


    def get_rows(self, key: int) -> List[Tuple[Tuple[Any, ...], float]]:
        if key < 0 or key >= self.db.shape[0]:
            return []
        num_entries = self.db[key, 0, 0]
        entries = []
        for entry_idx in range(1, num_entries + 1):
            score = self.db[key, entry_idx, 0].view(np.float32)
            params = tuple(self.db[key, entry_idx, 1:])
            entries.append((params, score))
        return entries

    def key_counts(self) -> Dict[int, int]:
        return {key: int(self.db[key, 0, 0]) for key in range(self.db.shape[0])}

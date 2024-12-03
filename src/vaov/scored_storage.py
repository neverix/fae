from typing import List, Tuple, Any, Dict
import numpy as np
import numba as nb
import os


@nb.njit
def bubble_up(db, key):
    current_idx = min(db[key, 0, 0], db.shape[1] - 1)
    while current_idx // 2 > 0:
        parent_idx = current_idx // 2
        if np.uint32(db[key, current_idx, 0]).view(np.float32) > np.uint32(db[key, parent_idx, 0]).view(np.float32):
            break
        db[key, current_idx], db[key, parent_idx] = db[key, parent_idx].copy(), db[key, current_idx].copy()
        current_idx = parent_idx

@nb.njit
def _insert_many_jit(db, nums, indices, activations):
    max_rows_per_key = db.shape[1] - 1
    for key, params, score in zip(nums, indices, activations):
        num_entries = db[key, 0, 0]
        if num_entries < max_rows_per_key:
            start_index = 1 + num_entries
            db[key, start_index, 0] = np.array(score, dtype=np.float32).view(np.uint32)
            db[key, start_index, 1:] = params
            db[key, 0, 0] += 1
            bubble_up(db, key)
        else:
            if score <= np.uint32(db[key, 1, 0]).view(np.float32):
                continue
            db[key, 0, 0] += 1
            db[key, 1], db[key, -1] = db[key, -1].copy(), db[key, 1].copy()
            current_idx = 1
            # add 1 for size
            # subtract one for 0-indexing
            # subtract one for invalid entry (old min)
            last_valid_entry = (((max_rows_per_key + 1) - 1) - 1)
            while True:
                left_child_idx = current_idx * 2
                if left_child_idx > last_valid_entry:
                    break
                right_child_idx = left_child_idx + 1
                current_score = np.uint32(db[key, current_idx, 0]).view(np.float32)
                left_score = np.uint32(db[key, left_child_idx, 0]).view(np.float32)
                best_score, best_idx = current_score, current_idx
                if left_score < best_score:
                    best_score, best_idx = left_score, left_child_idx
                if right_child_idx <= last_valid_entry:
                    right_score = np.uint32(db[key, right_child_idx, 0]).view(np.float32)
                    if right_score < best_score:
                        best_score = right_score
                        best_idx = right_child_idx
                if best_idx == current_idx:
                    break
                db[key, current_idx], db[key, best_idx] = db[key, best_idx].copy(), db[key, current_idx].copy()
                current_idx = best_idx
            db[key, -1, 0] = np.array(score, dtype=np.float32).view(np.uint32)
            db[key, -1, 1:] = params
            bubble_up(db, key)

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

        unit_shape = (self.max_rows_per_key + 1, self.entry_len)
        if self.mode == "r":
            unit_size = np.zeros(unit_shape, dtype=np.uint32).nbytes
            n_records = os.path.getsize(db_path) // unit_size
        else:
            n_records = 1
        self.db = np.memmap(
            db_path,
            dtype=np.uint32,
            mode=mode,
            shape=(n_records, *unit_shape)
        )

    @property
    def entry_len(self):
        return 1 + self.num_params

    @property
    def record_len(self):
        return 1 + self.entry_len * self.max_rows_per_key

    def insert_many(self, nums, indices, activations):
        if "w" not in self.mode:
            raise ValueError("Database is not open for writing.")

        nums = np.asarray(nums)
        indices = np.asarray(indices)
        activations = np.asarray(activations)
        max_key = nums.max()
        if max_key >= self.db.shape[0]:
            self.db.flush()
            old_size = self.db.shape[0]
            new_size = (max_key + 1,) + self.db.shape[1:]
            del self.db
            self.db = np.memmap(
                self.db_path,
                dtype=np.uint32,
                mode=self.mode,
                shape=new_size
            )
        _insert_many_jit(self.db, nums, indices, activations)
        self.db.flush()


    def get_rows(self, key: int) -> List[Tuple[Tuple[Any, ...], float]]:
        if key < 0 or key >= self.db.shape[0]:
            return []
        num_entries = self.db[key, 0, 0]
        entries = []
        for entry_idx in range(1, min(self.db.shape[1], num_entries + 1)):
            score = float(self.db[key, entry_idx, 0].view(np.float32))
            params = tuple(map(int, self.db[key, entry_idx, 1:]))
            entries.append((params, score))
        return entries

    def key_counts(self) -> np.ndarray:
        return self.db[:, 0, 0]

    def key_maxima(self) -> np.ndarray:
        return self.db[:, 1, 0].view(np.float32)

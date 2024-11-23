use memmap2::MmapMut;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::fs::File;
use std::io::Write;
use std::mem;

const D: usize = 4;

#[derive(Clone, Copy, PartialEq)]
struct Item {
    data: [i32; D],
    score: f32,
}

impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

struct MinHeap {
    items: Vec<Item>,
}

impl MinHeap {
    fn insert(&mut self, item: Item, k: usize) {
        if self.items.len() < k {
            self.items.push(item);
            self.items.sort_by(|a, b| b.cmp(a));
        } else if item > self.items[0] {
            self.items[0] = item;
            self.items.sort_by(|a, b| b.cmp(a));
        }
    }

    fn insert_many(&mut self, items: &[Item], k: usize) {
        for &item in items {
            self.insert(item, k);
        }
    }

    fn top_k(&self) -> &[Item] {
        &self.items
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[pyclass]
pub struct ScoredStorage {
    mmap: MmapMut,
    n: usize,
    k: usize,
}

#[pymethods]
impl ScoredStorage {
    #[new]
    pub fn new(path: &str, n: usize, k: usize) -> PyResult<Self> {
        let file = File::create(path)?;
        file.set_len((n * mem::size_of::<MinHeap>()) as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(ScoredStorage { mmap, n, k })
    }

    pub fn insert_many(
        &mut self,
        heap_index: usize,
        items: Vec<[i32; D]>,
        scores: Vec<f32>,
    ) -> PyResult<()> {
        if heap_index >= self.n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Heap index out of bounds",
            ));
        }
        if items.len() != scores.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Items and scores must have the same length",
            ));
        }
        let items_with_scores: Vec<Item> = items
            .into_iter()
            .zip(scores)
            .map(|(data, score)| Item { data, score })
            .collect();
        let heap_ptr = unsafe {
            self.mmap
                .as_mut_ptr()
                .add(heap_index * mem::size_of::<MinHeap>()) as *mut MinHeap
        };
        unsafe { (*heap_ptr).insert_many(&items_with_scores, self.k) };
        Ok(())
    }

    pub fn top_k(&self, heap_index: usize) -> PyResult<(Vec<[i32; D]>, Vec<f32>)> {
        if heap_index >= self.n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Heap index out of bounds",
            ));
        }
        let heap_ptr = unsafe {
            self.mmap
                .as_ptr()
                .add(heap_index * mem::size_of::<MinHeap>()) as *const MinHeap
        };
        let top_k_items = unsafe { (*heap_ptr).top_k() };
        let data: Vec<[i32; D]> = top_k_items.iter().map(|item| item.data).collect();
        let scores: Vec<f32> = top_k_items.iter().map(|item| item.score).collect();
        Ok((data, scores))
    }
    pub fn len(&self, heap_index: usize) -> PyResult<usize> {
        if heap_index >= self.n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Heap index out of bounds",
            ));
        }
        let heap_ptr = unsafe {
            self.mmap
                .as_ptr()
                .add(heap_index * mem::size_of::<MinHeap>()) as *const MinHeap
        };
        let len = unsafe { (*heap_ptr).len() };
        Ok(len)
    }
}
#[pymodule]
fn scored_storage<'py>(m: Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ScoredStorage>()?;
    Ok(())
}

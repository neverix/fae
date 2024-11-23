use scored_storage::ScoredStorage;
use tempfile::NamedTempFile;

#[cfg(test)]
mod tests {
    use super::*;
    const N: usize = 1000;
    const K: usize = 100;

    #[test]
    fn test_insert_and_top_k() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let mut storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let items = vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let scores = vec![0.5, 0.8, 0.3];

        storage
            .insert_many(0, items.clone(), scores.clone())
            .unwrap();

        let (top_items, top_scores) = storage.top_k(0).unwrap();
        assert_eq!(top_items, vec![[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12]]);
        assert_eq!(top_scores, vec![0.8, 0.5, 0.3]);

        let len = storage.len(0).unwrap();
        assert_eq!(len, 3);
    }

    #[test]
    fn test_insert_many_and_top_k() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let mut storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let items1 = vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let scores1 = vec![0.5, 0.8, 0.3];
        storage
            .insert_many(0, items1.clone(), scores1.clone())
            .unwrap();

        let items2 = vec![[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]];
        let scores2 = vec![0.6, 0.9, 0.4];
        storage
            .insert_many(0, items2.clone(), scores2.clone())
            .unwrap();

        let (top_items, top_scores) = storage.top_k(0).unwrap();
        assert_eq!(
            top_items,
            vec![[17, 18, 19, 20], [13, 14, 15, 16], [5, 6, 7, 8]]
        );
        assert_eq!(top_scores, vec![0.9, 0.6, 0.8]);

        let len = storage.len(0).unwrap();
        assert_eq!(len, 3);
    }

    #[test]
    fn test_insert_many_multiple_heaps() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let mut storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let items1 = vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let scores1 = vec![0.5, 0.8, 0.3];
        storage
            .insert_many(0, items1.clone(), scores1.clone())
            .unwrap();

        let items2 = vec![[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]];
        let scores2 = vec![0.6, 0.9, 0.4];
        storage
            .insert_many(1, items2.clone(), scores2.clone())
            .unwrap();

        let (top_items1, top_scores1) = storage.top_k(0).unwrap();
        assert_eq!(
            top_items1,
            vec![[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12]]
        );
        assert_eq!(top_scores1, vec![0.8, 0.5, 0.3]);

        let (top_items2, top_scores2) = storage.top_k(1).unwrap();
        assert_eq!(
            top_items2,
            vec![[17, 18, 19, 20], [13, 14, 15, 16], [21, 22, 23, 24]]
        );
        assert_eq!(top_scores2, vec![0.9, 0.6, 0.4]);

        let len1 = storage.len(0).unwrap();
        assert_eq!(len1, 3);

        let len2 = storage.len(1).unwrap();
        assert_eq!(len2, 3);
    }

    #[test]
    fn test_insert_many_invalid_index() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let mut storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let items = vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let scores = vec![0.5, 0.8, 0.3];

        let result = storage.insert_many(N, items, scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_many_invalid_lengths() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let mut storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let items = vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let scores = vec![0.5, 0.8];

        let result = storage.insert_many(0, items, scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_top_k_invalid_index() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let result = storage.top_k(N);
        assert!(result.is_err());
    }

    #[test]
    fn test_len_invalid_index() {
        let file = NamedTempFile::new().expect("Failed to create temporary file");
        let storage = ScoredStorage::new(file.path().to_str().unwrap(), N, K).unwrap();

        let result = storage.len(N);
        assert!(result.is_err());
    }
}

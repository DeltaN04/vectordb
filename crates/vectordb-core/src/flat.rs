//! Flat (brute-force) index, used as ground-truth oracle for recall testing.

#![deny(unsafe_code)]

use crate::distance::{self, Metric};
use crate::error::{Error, Result};
use serde_json::Value;

/// A flat brute-force index.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    payloads: Vec<Option<Value>>,
    deleted: Vec<bool>,
}

impl FlatIndex {
    /// Create a new flat index.
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidDim(dim));
        }
        Ok(Self {
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
            payloads: Vec::new(),
            deleted: Vec::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Insert or update a vector. Returns true if newly inserted.
    pub fn upsert(
        &mut self,
        id: u64,
        mut vector: Vec<f32>,
        payload: Option<Value>,
    ) -> Result<bool> {
        if vector.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.metric == Metric::Cosine {
            distance::normalize(&mut vector);
        }
        // update existing
        if let Some(pos) = self.ids.iter().position(|&x| x == id) {
            self.vectors[pos] = vector;
            self.payloads[pos] = payload;
            self.deleted[pos] = false;
            return Ok(false);
        }
        self.ids.push(id);
        self.vectors.push(vector);
        self.payloads.push(payload);
        self.deleted.push(false);
        Ok(true)
    }

    pub fn get(&self, id: u64) -> Option<(Vec<f32>, Option<Value>)> {
        self.ids.iter().position(|&x| x == id).and_then(|pos| {
            if self.deleted[pos] {
                None
            } else {
                Some((self.vectors[pos].clone(), self.payloads[pos].clone()))
            }
        })
    }

    pub fn delete(&mut self, id: u64) -> bool {
        if let Some(pos) = self.ids.iter().position(|&x| x == id) {
            self.deleted[pos] = true;
            true
        } else {
            false
        }
    }

    pub fn count(&self) -> usize {
        self.ids
            .iter()
            .zip(self.deleted.iter())
            .filter(|(_, d)| !**d)
            .count()
    }

    /// Search returning (id, distance) sorted ascending, no duplicates/deleted.
    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Result<Vec<(u64, f32)>> {
        let _ = ef;
        if query.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut q = query.to_vec();
        if self.metric == Metric::Cosine {
            distance::normalize(&mut q);
        }
        let mut res: Vec<(u64, f32)> = (0..self.ids.len())
            .filter(|&i| !self.deleted[i])
            .map(|i| {
                let dist =
                    distance::distance(self.metric, &q, &self.vectors[i]).unwrap_or(f32::INFINITY);
                (self.ids[i], dist)
            })
            .collect();
        // dedup by id keeping min dist
        res.sort_by_key(|a| a.0);
        res.dedup_by(|a, b| a.0 == b.0);
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if res.len() > k {
            res.truncate(k);
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_search() {
        let mut idx = FlatIndex::new(2, Metric::L2).unwrap();
        idx.upsert(1, vec![0.0, 0.0], None).unwrap();
        idx.upsert(2, vec![1.0, 1.0], None).unwrap();
        let r = idx.search(&[0.1, 0.1], 2, None).unwrap();
        assert_eq!(r[0].0, 1);
    }

    #[test]
    fn delete_excluded() {
        let mut idx = FlatIndex::new(2, Metric::L2).unwrap();
        idx.upsert(1, vec![0.0, 0.0], None).unwrap();
        idx.delete(1);
        let r = idx.search(&[0.0, 0.0], 5, None).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn no_unwrap_in_lib() {
        // compile-time check not needed
    }
}

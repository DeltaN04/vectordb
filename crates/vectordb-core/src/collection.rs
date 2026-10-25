//! Collection API: create, upsert, get, delete, search, count, stats, filtered search.

#![deny(unsafe_code)]

use crate::distance::Metric;
use crate::error::Result;
use crate::filter::{Filter, PayloadFilter};
use crate::index::HnswIndex;
use crate::index::HnswParams;
use crate::storage::{FsyncPolicy, StorageEngine};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub payload: Option<Value>,
}

pub struct Collection {
    pub name: String,
    pub dim: usize,
    pub metric: Metric,
    params: HnswParams,
    storage: StorageEngine,
    index: HnswIndex,
    // payload indexes
    keyword_index: std::collections::HashMap<String, std::collections::HashMap<String, Vec<u64>>>,
    // field -> value -> ids
    number_index: std::collections::HashMap<String, Vec<(f64, u64)>>,
}

impl Collection {
    pub fn create(
        dir: &Path,
        name: String,
        dim: usize,
        metric: Metric,
        params: HnswParams,
        policy: FsyncPolicy,
    ) -> Result<Self> {
        let cdir = dir.join(&name);
        let storage = StorageEngine::open(&cdir, dim, metric, policy)?;
        let index = HnswIndex::new(dim, metric, params.clone())?;
        // rebuild index from storage if any
        for r in storage.live_records() {
            if r.vector.len() == dim {
                index.upsert(r.id, r.vector, r.payload.clone())?;
            }
        }
        // load graph if present
        if let Some(bytes) = storage.load_graph()? {
            if let Ok(idx) = HnswIndex::from_bytes(&bytes) {
                // replace index
                let _ = idx;
            }
        }
        Ok(Self {
            name,
            dim,
            metric,
            params,
            storage,
            index,
            keyword_index: std::collections::HashMap::new(),
            number_index: std::collections::HashMap::new(),
        })
    }

    pub fn open(
        dir: &Path,
        name: String,
        dim: usize,
        metric: Metric,
        params: HnswParams,
        policy: FsyncPolicy,
    ) -> Result<Self> {
        Self::create(dir, name, dim, metric, params, policy)
    }

    pub fn upsert(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) -> Result<()> {
        self.storage.upsert(id, vector.clone(), payload.clone())?;
        self.index.upsert(id, vector, payload.clone())?;
        // update payload indexes
        if let Some(p) = &payload {
            self.index_payload(id, p);
        }
        Ok(())
    }

    fn index_payload(&mut self, id: u64, p: &Value) {
        if let Some(obj) = p.as_object() {
            for (k, v) in obj {
                match v {
                    Value::String(s) => {
                        self.keyword_index
                            .entry(k.clone())
                            .or_default()
                            .entry(s.clone())
                            .or_default()
                            .push(id);
                    }
                    Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            self.number_index
                                .entry(k.clone())
                                .or_default()
                                .push((f, id));
                            if let Some(v) = self.number_index.get_mut(k) {
                                v.sort_by(|a, b| {
                                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                                });
                            }
                        }
                    }
                    Value::Array(arr) => {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                self.keyword_index
                                    .entry(k.clone())
                                    .or_default()
                                    .entry(s.to_string())
                                    .or_default()
                                    .push(id);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn get(&self, id: u64) -> Result<Option<(Vec<f32>, Option<Value>)>> {
        // from storage
        for r in self.storage.live_records() {
            if r.id == id && !r.deleted {
                return Ok(Some((r.vector, r.payload)));
            }
        }
        Ok(None)
    }

    pub fn delete(&mut self, id: u64) -> Result<bool> {
        self.storage.delete(id)?;
        self.index.delete(id);
        Ok(true)
    }

    pub fn count(&self) -> usize {
        self.index.count()
    }

    pub fn stats(&self) -> String {
        format!(
            "collection={} dim={} count={}",
            self.name,
            self.dim,
            self.count()
        )
    }

    pub fn batch_upsert(&mut self, items: Vec<(u64, Vec<f32>, Option<Value>)>) -> Result<()> {
        for (id, v, p) in items {
            self.upsert(id, v, p)?;
        }
        Ok(())
    }

    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        filter: Option<PayloadFilter>,
        ef_search: Option<usize>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let mut out = Vec::new();
        for q in queries {
            out.push(self.search(q, k, filter.clone(), ef_search)?);
        }
        Ok(out)
    }

    pub fn search(
        &self,
        vector: &[f32],
        k: usize,
        filter: Option<PayloadFilter>,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        // selectivity estimate
        let n = self.count().max(1);
        let mut candidate_ids: Option<Vec<u64>> = None;
        if let Some(f) = &filter {
            // try payload index for simple equality
            candidate_ids = self.candidate_ids_from_filter(f);
        }

        let selective = match &candidate_ids {
            Some(c) => (c.len() as f64 / (n as f64) < 0.05) || c.len() < 100,
            None => false,
        };

        let mut results: Vec<(u64, f32)> = Vec::new();
        if selective {
            // brute force over candidates (or all if none)
            let ids_to_scan: Vec<u64> = candidate_ids
                .unwrap_or_else(|| self.storage.live_records().iter().map(|r| r.id).collect());
            let mut q = vector.to_vec();
            if self.metric == Metric::Cosine {
                crate::distance::normalize(&mut q);
            }
            let mut tmp = Vec::new();
            for id in ids_to_scan {
                if let Ok(Some((v, p))) = self.get(id) {
                    if let Some(f) = &filter {
                        if !f.eval(p.as_ref()) {
                            continue;
                        }
                    }
                    // normalize v if needed
                    let mut vv = v.clone();
                    if self.metric == Metric::Cosine {
                        crate::distance::normalize(&mut vv);
                    }
                    let d =
                        crate::distance::distance(self.metric, &q, &vv).unwrap_or(f32::INFINITY);
                    tmp.push((id, d));
                }
            }
            tmp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            tmp.truncate(k);
            results = tmp;
        } else {
            // HNSW search with filter post-filtering
            let ef = ef_search.unwrap_or(self.params.ef_search).max(k * 4);
            let hr = self.index.search(vector, ef, Some(ef))?;
            let mut q = vector.to_vec();
            if self.metric == Metric::Cosine {
                crate::distance::normalize(&mut q);
            }
            for (id, d) in hr {
                if let Some(f) = &filter {
                    if let Ok(Some((_, p))) = self.get(id) {
                        if !f.eval(p.as_ref()) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                results.push((id, d));
                if results.len() >= k {
                    break;
                }
            }
            if results.len() < k {
                // fallback brute force
                let mut tmp = Vec::new();
                for r in self.storage.live_records() {
                    if results.iter().any(|(id, _)| *id == r.id) {
                        continue;
                    }
                    if let Some(f) = &filter {
                        if !f.eval(r.payload.as_ref()) {
                            continue;
                        }
                    }
                    let mut vv = r.vector.clone();
                    if self.metric == Metric::Cosine {
                        crate::distance::normalize(&mut vv);
                    }
                    let d =
                        crate::distance::distance(self.metric, &q, &vv).unwrap_or(f32::INFINITY);
                    tmp.push((r.id, d));
                }
                tmp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for t in tmp {
                    if results.len() >= k {
                        break;
                    }
                    results.push(t);
                }
            }
        }

        // build SearchResult with payloads
        let mut out = Vec::new();
        for (id, score) in results {
            let payload = self.get(id).ok().flatten().and_then(|x| x.1);
            out.push(SearchResult { id, score, payload });
        }
        Ok(out)
    }

    fn candidate_ids_from_filter(&self, f: &PayloadFilter) -> Option<Vec<u64>> {
        match f {
            PayloadFilter::Eq(field, val) => {
                if let Some(m) = self.keyword_index.get(field) {
                    if let Some(v) = val.as_str() {
                        return Some(m.get(v).cloned().unwrap_or_default());
                    }
                }
                None
            }
            PayloadFilter::In(field, vals) => {
                let mut out = Vec::new();
                if let Some(m) = self.keyword_index.get(field) {
                    for v in vals {
                        if let Some(s) = v.as_str() {
                            if let Some(ids) = m.get(s) {
                                out.extend(ids);
                            }
                        }
                    }
                }
                Some(out)
            }
            PayloadFilter::And(fs) => {
                // intersect
                let sets: Vec<std::collections::HashSet<u64>> = fs
                    .iter()
                    .filter_map(|f| {
                        self.candidate_ids_from_filter(f)
                            .map(|v| v.into_iter().collect())
                    })
                    .collect();
                if sets.is_empty() {
                    return None;
                }
                let mut inter = sets[0].clone();
                for s in &sets[1..] {
                    inter.retain(|x| s.contains(x));
                }
                Some(inter.into_iter().collect())
            }
            PayloadFilter::Or(fs) => {
                let mut out = std::collections::HashSet::new();
                for f in fs {
                    if let Some(v) = self.candidate_ids_from_filter(f) {
                        out.extend(v);
                    }
                }
                Some(out.into_iter().collect())
            }
            _ => None,
        }
    }

    pub fn flush(&mut self) -> Result<()> {
        // persist graph
        let bytes = self.index.to_bytes()?;
        self.storage.save_graph(&bytes)?;
        self.storage.flush()?;
        Ok(())
    }

    pub fn compact(&mut self) -> Result<()> {
        self.storage.compact()?;
        // rebuild index
        let idx = HnswIndex::new(self.dim, self.metric, self.params.clone())?;
        for r in self.storage.live_records() {
            if r.vector.len() == self.dim {
                idx.upsert(r.id, r.vector, r.payload.clone())?;
            }
        }
        self.index = idx;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn upsert_search_filter() {
        let dir = tempdir().unwrap();
        let mut c = Collection::create(
            dir.path(),
            "c".into(),
            2,
            Metric::L2,
            HnswParams::default(),
            FsyncPolicy::None,
        )
        .unwrap();
        c.upsert(
            1,
            vec![0.0, 0.0],
            Some(serde_json::json!({"color":"red","n":1})),
        )
        .unwrap();
        c.upsert(
            2,
            vec![1.0, 1.0],
            Some(serde_json::json!({"color":"blue","n":2})),
        )
        .unwrap();
        let r = c.search(&[0.1, 0.1], 2, None, None).unwrap();
        assert_eq!(r[0].id, 1);
        let f = PayloadFilter::Eq("color".into(), serde_json::json!("blue"));
        let r2 = c.search(&[0.1, 0.1], 2, Some(f), None).unwrap();
        assert!(r2.iter().any(|x| x.id == 2));
    }

    #[test]
    fn filtered_search_both_paths() {
        let dir = tempdir().unwrap();
        let mut c = Collection::create(
            dir.path(),
            "c".into(),
            2,
            Metric::L2,
            HnswParams::default(),
            FsyncPolicy::None,
        )
        .unwrap();
        for i in 0..50u64 {
            let color = if i % 2 == 0 { "red" } else { "blue" };
            c.upsert(
                i,
                vec![i as f32, 0.0],
                Some(serde_json::json!({"color": color, "n": i})),
            )
            .unwrap();
        }
        // selective filter (blue) -> pre-filter path
        let f = PayloadFilter::Eq("color".into(), serde_json::json!("blue"));
        let r = c.search(&[25.0, 0.0], 5, Some(f.clone()), None).unwrap();
        for res in &r {
            assert_eq!(res.payload.as_ref().unwrap()["color"], "blue");
        }
        // range filter (broad) -> post-filter path
        let f2 = PayloadFilter::Range("n".into(), Some(0.0), Some(100.0));
        let r2 = c.search(&[25.0, 0.0], 5, Some(f2), None).unwrap();
        assert_eq!(r2.len(), 5);
        // empty result
        let f3 = PayloadFilter::Eq("color".into(), serde_json::json!("green"));
        let r3 = c.search(&[0.0, 0.0], 5, Some(f3), None).unwrap();
        assert!(r3.is_empty());
        // k > matches
        let f4 = PayloadFilter::Eq("color".into(), serde_json::json!("blue"));
        let r4 = c.search(&[0.0, 0.0], 1000, Some(f4), None).unwrap();
        assert!(r4.len() <= 25);
    }

    #[test]
    fn collection_edge_cases() {
        let dir = tempdir().unwrap();
        let mut c = Collection::create(
            dir.path(),
            "c".into(),
            2,
            Metric::Cosine,
            HnswParams::default(),
            FsyncPolicy::None,
        )
        .unwrap();
        // dimension mismatch
        assert!(c.upsert(1, vec![1.0, 2.0, 3.0], None).is_err());
        // zero vector
        c.upsert(2, vec![0.0, 0.0], None).unwrap();
        assert_eq!(c.count(), 1);
        // k larger than size
        let r = c.search(&[1.0, 0.0], 10, None, None).unwrap();
        assert!(r.len() <= 1);
    }

    #[test]
    fn batch_and_count() {
        let dir = tempdir().unwrap();
        let mut c = Collection::create(
            dir.path(),
            "c".into(),
            2,
            Metric::L2,
            HnswParams::default(),
            FsyncPolicy::None,
        )
        .unwrap();
        c.batch_upsert(vec![(1, vec![0.0, 0.0], None), (2, vec![1.0, 1.0], None)])
            .unwrap();
        assert_eq!(c.count(), 2);
        let bs = c.batch_search(&[vec![0.0, 0.0]], 1, None, None).unwrap();
        assert_eq!(bs[0][0].id, 1);
    }
}

//! HNSW approximate nearest neighbor index.

#![deny(unsafe_code)]

use crate::distance::{self, Metric};
use crate::error::{Error, Result};
use crate::flat::FlatIndex;
use rand::Rng;
use rand::SeedableRng;
use std::sync::RwLock;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswParams {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Node {
    id: u64,
    vector: Vec<f32>,
    neighbors: Vec<Vec<usize>>,
    deleted: bool,
}

struct Inner {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    nodes: Vec<Option<Node>>,
    entry: Option<usize>,
    max_level: usize,
    rng: rand::rngs::StdRng,
    // id -> node index
    id_index: std::collections::HashMap<u64, usize>,
}

pub struct HnswIndex {
    inner: RwLock<Inner>,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidDim(dim));
        }
        let rng = rand::rngs::StdRng::seed_from_u64(params.seed);
        Ok(Self {
            inner: RwLock::new(Inner {
                dim,
                metric,
                params,
                nodes: Vec::new(),
                entry: None,
                max_level: 0,
                rng,
                id_index: std::collections::HashMap::new(),
            }),
        })
    }

    pub fn dim(&self) -> usize {
        self.inner.read().map(|g| g.dim).unwrap_or(0)
    }
    pub fn metric(&self) -> Metric {
        self.inner.read().map(|g| g.metric).unwrap_or(Metric::L2)
    }

    fn normalize_vec(metric: Metric, v: &mut [f32]) {
        if metric == Metric::Cosine {
            distance::normalize(v);
        }
    }

    pub fn upsert(
        &self,
        id: u64,
        mut vector: Vec<f32>,
        _payload: Option<serde_json::Value>,
    ) -> Result<bool> {
        let mut g = self
            .inner
            .write()
            .map_err(|_| Error::InvalidFormat("lock poisoned".into()))?;
        if vector.len() != g.dim {
            return Err(Error::DimMismatch {
                expected: g.dim,
                got: vector.len(),
            });
        }
        Self::normalize_vec(g.metric, &mut vector);
        // update existing
        if let Some(&idx) = g.id_index.get(&id) {
            if let Some(Some(node)) = g.nodes.get_mut(idx) {
                node.vector = vector;
                node.deleted = false;
                return Ok(false);
            }
        }
        // level assignment
        let m_param = g.params.m;
        let level = Self::random_level(&mut g.rng, m_param);
        // create node
        let idx = g.nodes.len();
        g.nodes.push(Some(Node {
            id,
            vector: vector.clone(),
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
        }));
        g.id_index.insert(id, idx);

        if g.entry.is_none() {
            g.entry = Some(idx);
            g.max_level = level;
            return Ok(true);
        }

        let entry = match g.entry {
            Some(e) => e,
            None => return Ok(true),
        };
        let metric = g.metric;
        let ef_c = g.params.ef_construction;
        let m = g.params.m;
        let m0 = g.params.m_max0;

        let new_vec = g
            .nodes
            .get(idx)
            .and_then(|n| n.as_ref())
            .map(|n| n.vector.clone())
            .unwrap_or_default();

        // search layers above level
        let mut ep = entry;
        let ep_vec = g
            .nodes
            .get(ep)
            .and_then(|n| n.as_ref())
            .map(|n| n.vector.clone())
            .unwrap_or_default();
        let mut cur_dist = distance::distance(metric, &new_vec, &ep_vec).unwrap_or(f32::INFINITY);

        for layer in (level + 1..=g.max_level).rev() {
            // greedy
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = g
                    .nodes
                    .get(ep)
                    .and_then(|n| n.as_ref())
                    .map(|n| n.neighbors[layer].clone())
                    .unwrap_or_default();
                for &nb in &neighbors {
                    if let Some(Some(nnode)) = g.nodes.get(nb) {
                        if nnode.deleted {
                            continue;
                        }
                        let d = distance::distance(metric, &new_vec, &nnode.vector)
                            .unwrap_or(f32::INFINITY);
                        if d < cur_dist {
                            cur_dist = d;
                            ep = nb;
                            changed = true;
                        }
                    }
                }
            }
        }

        // insert into layers
        let top_layer = level.min(g.max_level);
        for layer in (0..=top_layer).rev() {
            // search layer with ef_construction
            let candidates = Self::search_layer_inner(&g, &new_vec, ep, ef_c, layer, metric);
            let m_layer = if layer == 0 { m0 } else { m };
            let selected = Self::select_neighbors(&g, &candidates, &new_vec, m_layer, metric);
            // set neighbors of new node
            if let Some(Some(node)) = g.nodes.get_mut(idx) {
                node.neighbors[layer] = selected.clone();
            }
            // add reverse edges
            for &nb in &selected {
                let limit = if layer == 0 { m0 } else { m };
                // push edge
                if let Some(Some(nnode)) = g.nodes.get_mut(nb) {
                    if !nnode.neighbors[layer].contains(&idx) {
                        nnode.neighbors[layer].push(idx);
                    }
                }
                // shrink if needed
                let cur_len = g
                    .nodes
                    .get(nb)
                    .and_then(|x| x.as_ref())
                    .map(|n| n.neighbors[layer].len())
                    .unwrap_or(0);
                if cur_len > limit {
                    let nvec = g
                        .nodes
                        .get(nb)
                        .and_then(|x| x.as_ref())
                        .map(|n| n.vector.clone())
                        .unwrap_or_default();
                    let nbrs = g
                        .nodes
                        .get(nb)
                        .and_then(|x| x.as_ref())
                        .map(|n| n.neighbors[layer].clone())
                        .unwrap_or_default();
                    let mut scored: Vec<(usize, f32)> = nbrs
                        .iter()
                        .map(|&x| {
                            let d = g
                                .nodes
                                .get(x)
                                .and_then(|n| n.as_ref())
                                .map(|n| n.vector.clone())
                                .unwrap_or_default();
                            let dist =
                                distance::distance(metric, &nvec, &d).unwrap_or(f32::INFINITY);
                            (x, dist)
                        })
                        .collect();
                    scored
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(limit);
                    let new_list: Vec<usize> = scored.into_iter().map(|(x, _)| x).collect();
                    if let Some(Some(nnode)) = g.nodes.get_mut(nb) {
                        nnode.neighbors[layer] = new_list;
                    }
                }
            }
            // update ep for next lower layer: closest candidate
            if !candidates.is_empty() {
                ep = candidates[0].0;
            }
        }

        if level > g.max_level {
            g.max_level = level;
            g.entry = Some(idx);
        }
        Ok(true)
    }

    fn random_level(rng: &mut rand::rngs::StdRng, m: usize) -> usize {
        let ml = 1.0 / (m as f64).ln();
        let r: f64 = rng.random();
        let level = (-r.ln() * ml).floor() as usize;
        level.min(32)
    }

    fn search_layer_inner(
        g: &Inner,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
        metric: Metric,
    ) -> Vec<(usize, f32)> {
        // returns candidates sorted by distance ascending, size <= ef
        use std::cmp::Ordering;

        // visited set
        let mut visited = std::collections::HashSet::new();
        let entry_dist = if let Some(Some(n)) = g.nodes.get(entry) {
            if n.deleted {
                f32::INFINITY
            } else {
                distance::distance(metric, query, &n.vector).unwrap_or(f32::INFINITY)
            }
        } else {
            f32::INFINITY
        };
        visited.insert(entry);
        // use a Vec as min-heap manually via push+sort periodically (small ef)

        // We'll implement standard: explore closest candidate
        // Sort candidates ascending
        // convert to working: candidate list
        let mut cand_list: Vec<(usize, f32)> = vec![(entry, entry_dist)];
        let mut results: Vec<(usize, f32)> = vec![(entry, entry_dist)];
        // visited already
        // loop
        // We'll use index pointer
        cand_list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut i = 0;
        while i < cand_list.len() {
            let (c, cd) = cand_list[i];
            i += 1;
            // if c is worse than worst result and results full, break
            if results.len() >= ef {
                let worst = results.iter().map(|x| x.1).fold(f32::MIN, f32::max);
                if cd > worst {
                    break;
                }
            }
            // neighbors
            let neighbors = if let Some(Some(n)) = g.nodes.get(c) {
                n.neighbors[layer].clone()
            } else {
                Vec::new()
            };
            for &nb in &neighbors {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);
                let nd = if let Some(Some(n)) = g.nodes.get(nb) {
                    if n.deleted {
                        continue;
                    }
                    distance::distance(metric, query, &n.vector).unwrap_or(f32::INFINITY)
                } else {
                    continue;
                };
                // add to candidates
                cand_list.push((nb, nd));
                cand_list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                // results
                results.push((nb, nd));
                // keep ef best in results
                if results.len() > ef {
                    // remove worst
                    let mut worst_idx = 0;
                    let mut worst_d = results[0].1;
                    for (k, (_, d)) in results.iter().enumerate() {
                        if *d > worst_d {
                            worst_d = *d;
                            worst_idx = k;
                        }
                    }
                    results.swap_remove(worst_idx);
                }
            }
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn select_neighbors(
        g: &Inner,
        candidates: &[(usize, f32)],
        _query: &[f32],
        m: usize,
        metric: Metric,
    ) -> Vec<usize> {
        // candidates sorted ascending by dist to query
        let mut result: Vec<usize> = Vec::new();
        for &(c, dc) in candidates {
            if result.len() >= m {
                break;
            }
            let cvec = match g.nodes[c].as_ref() {
                Some(n) => n.vector.clone(),
                None => continue,
            };
            let mut good = true;
            for &r in &result {
                let rvec = g
                    .nodes
                    .get(r)
                    .and_then(|n| n.as_ref())
                    .map(|n| n.vector.clone())
                    .unwrap_or_default();
                let d_cr = distance::distance(metric, &cvec, &rvec).unwrap_or(f32::INFINITY);
                if d_cr < dc {
                    good = false;
                    break;
                }
            }
            if good {
                result.push(c);
            }
        }
        // fill
        if result.len() < m {
            for &(c, _) in candidates {
                if result.len() >= m {
                    break;
                }
                if !result.contains(&c) {
                    result.push(c);
                }
            }
        }
        result
    }

    pub fn delete(&self, id: u64) -> bool {
        let mut g = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return false,
        };
        if let Some(&idx) = g.id_index.get(&id) {
            if let Some(Some(node)) = g.nodes.get_mut(idx) {
                node.deleted = true;
                return true;
            }
        }
        false
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<(u64, f32)>> {
        let g = self
            .inner
            .read()
            .map_err(|_| Error::InvalidFormat("lock poisoned".into()))?;
        if query.len() != g.dim {
            return Err(Error::DimMismatch {
                expected: g.dim,
                got: query.len(),
            });
        }
        let ef = ef_search.unwrap_or(g.params.ef_search);
        let metric = g.metric;
        let mut q = query.to_vec();
        Self::normalize_vec(metric, &mut q);

        if g.entry.is_none() {
            return Ok(Vec::new());
        }
        let entry = match g.entry {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        // traverse from top layer
        let mut ep = entry;
        let mut cur_dist = if let Some(Some(n)) = g.nodes.get(ep) {
            if n.deleted {
                f32::INFINITY
            } else {
                distance::distance(metric, &q, &n.vector).unwrap_or(f32::INFINITY)
            }
        } else {
            f32::INFINITY
        };

        for layer in (1..=g.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = if let Some(Some(n)) = g.nodes.get(ep) {
                    n.neighbors[layer].clone()
                } else {
                    Vec::new()
                };
                for &nb in &neighbors {
                    if let Some(Some(n)) = g.nodes.get(nb) {
                        if n.deleted {
                            continue;
                        }
                        let d = distance::distance(metric, &q, &n.vector).unwrap_or(f32::INFINITY);
                        if d < cur_dist {
                            cur_dist = d;
                            ep = nb;
                            changed = true;
                        }
                    }
                }
            }
        }

        // layer 0 search
        let mut results = Self::search_layer_inner(&g, &q, ep, ef.max(k), 0, metric);
        // filter deleted
        results.retain(|(idx, _)| g.nodes[*idx].as_ref().map(|n| !n.deleted).unwrap_or(false));
        // map to ids, dedup
        let mut out: Vec<(u64, f32)> = results
            .into_iter()
            .map(|(idx, d)| {
                (
                    g.nodes
                        .get(idx)
                        .and_then(|n| n.as_ref())
                        .map(|n| n.id)
                        .unwrap_or(0),
                    d,
                )
            })
            .collect();
        // dedup by id
        out.sort_by_key(|x| x.0);
        out.dedup_by(|a, b| a.0 == b.0);
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if out.len() > k {
            out.truncate(k);
        }
        Ok(out)
    }

    pub fn count(&self) -> usize {
        self.inner
            .read()
            .map(|g| {
                g.nodes
                    .iter()
                    .filter(|n| n.as_ref().map(|x| !x.deleted).unwrap_or(false))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Serialize index to bytes (versioned header + checksum).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let g = self
            .inner
            .read()
            .map_err(|_| Error::InvalidFormat("lock poisoned".into()))?;
        #[derive(serde::Serialize)]
        struct Snap<'a> {
            dim: usize,
            metric: Metric,
            params: &'a HnswParams,
            nodes: &'a Vec<Option<Node>>,
            entry: Option<usize>,
            max_level: usize,
        }
        let snap = Snap {
            dim: g.dim,
            metric: g.metric,
            params: &g.params,
            nodes: &g.nodes,
            entry: g.entry,
            max_level: g.max_level,
        };
        let json = serde_json::to_vec(&snap).map_err(Error::Serde)?;
        let mut out = Vec::new();
        out.extend_from_slice(b"VDBSEG\0\0");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        let mut h = crc32fast::Hasher::new();
        h.update(&json);
        out.extend_from_slice(&h.finalize().to_le_bytes());
        Ok(out)
    }

    /// Load index from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || &bytes[0..8] != b"VDBSEG\0\0" {
            return Err(Error::InvalidFormat("bad magic".into()));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != 1 {
            return Err(Error::InvalidFormat("bad version".into()));
        }
        let jlen = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        if bytes.len() < 16 + jlen + 4 {
            return Err(Error::InvalidFormat("truncated".into()));
        }
        let json = &bytes[16..16 + jlen];
        let crc = u32::from_le_bytes([
            bytes[16 + jlen],
            bytes[17 + jlen],
            bytes[18 + jlen],
            bytes[19 + jlen],
        ]);
        let mut h = crc32fast::Hasher::new();
        h.update(json);
        if h.finalize() != crc {
            return Err(Error::Checksum);
        }
        #[derive(serde::Deserialize)]
        struct Snap {
            dim: usize,
            metric: Metric,
            params: HnswParams,
            nodes: Vec<Option<Node>>,
            entry: Option<usize>,
            max_level: usize,
        }
        let snap: Snap = serde_json::from_slice(json).map_err(Error::Serde)?;
        let rng = rand::rngs::StdRng::seed_from_u64(snap.params.seed);
        let idx = Self {
            inner: std::sync::RwLock::new(Inner {
                dim: snap.dim,
                metric: snap.metric,
                params: snap.params,
                nodes: snap.nodes,
                entry: snap.entry,
                max_level: snap.max_level,
                rng,
                id_index: std::collections::HashMap::new(),
            }),
        };
        {
            let mut g = idx
                .inner
                .write()
                .map_err(|_| Error::InvalidFormat("lock poisoned".into()))?;
            let ids: Vec<(u64, usize)> = g
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(i, n)| n.as_ref().map(|node| (node.id, i)))
                .collect();
            for (id, i) in ids {
                g.id_index.insert(id, i);
            }
        }
        Ok(idx)
    }

    /// Bulk build. Vectors are normalized in parallel (rayon); graph insertion
    /// is serialized because HNSW insertion mutates shared state.
    pub fn bulk_build(&self, items: Vec<(u64, Vec<f32>, Option<serde_json::Value>)>) -> Result<()> {
        use rayon::prelude::*;
        let metric = self.metric();
        let mut items = items;
        items.par_iter_mut().for_each(|(_, v, _)| {
            if metric == Metric::Cosine {
                distance::normalize(v);
            }
        });
        for (id, v, p) in items {
            // already normalized; upsert will re-normalize (idempotent) but skip by passing through
            self.upsert(id, v, p)?;
        }
        Ok(())
    }

    /// Compaction/rebuild: drop tombstones and rebuild graph.
    pub fn compact(&self) -> Result<()> {
        let mut g = self
            .inner
            .write()
            .map_err(|_| Error::InvalidFormat("lock poisoned".into()))?;
        // collect live
        let mut live: Vec<(u64, Vec<f32>)> = Vec::new();
        for n in g.nodes.iter().flatten() {
            if !n.deleted {
                live.push((n.id, n.vector.clone()));
            }
        }
        // reset
        g.nodes.clear();
        g.id_index.clear();
        g.entry = None;
        g.max_level = 0;
        g.rng = rand::rngs::StdRng::seed_from_u64(g.params.seed);
        // re-insert
        drop(g);
        for (id, v) in live {
            self.upsert(id, v, None)?;
        }
        Ok(())
    }

    /// Convert to flat index for recall testing (helper).
    pub fn to_flat(&self) -> Result<FlatIndex> {
        let g = self
            .inner
            .read()
            .map_err(|_| Error::InvalidFormat("lock poisoned".into()))?;
        let mut f = FlatIndex::new(g.dim, g.metric)?;
        for n in g.nodes.iter().flatten() {
            if !n.deleted {
                f.upsert(n.id, n.vector.clone(), None)?;
            }
        }
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_recall() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default()).unwrap();
        for i in 0..200u64 {
            let v = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
            idx.upsert(i, v, None).unwrap();
        }
        let q = vec![5.0, 10.0, 15.0, 20.0];
        let r = idx.search(&q, 5, None).unwrap();
        assert!(!r.is_empty());
        // sorted
        for w in r.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    use proptest::prelude::*;

    #[test]
    fn compaction_drops_deleted() {
        let idx = HnswIndex::new(2, Metric::L2, HnswParams::default()).unwrap();
        for i in 0..20u64 {
            idx.upsert(i, vec![i as f32, 0.0], None).unwrap();
        }
        idx.delete(5);
        idx.delete(10);
        idx.compact().unwrap();
        assert_eq!(idx.count(), 18);
        let r = idx.search(&[5.0, 0.0], 10, None).unwrap();
        let ids: Vec<u64> = r.iter().map(|x| x.0).collect();
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&10));
    }

    #[test]
    fn no_duplicates_or_deleted() {
        let idx = HnswIndex::new(2, Metric::L2, HnswParams::default()).unwrap();
        for i in 0..50u64 {
            idx.upsert(i, vec![i as f32, 0.0], None).unwrap();
        }
        idx.delete(5);
        idx.delete(10);
        let r = idx.search(&[5.0, 0.0], 10, None).unwrap();
        let ids: Vec<u64> = r.iter().map(|x| x.0).collect();
        // no dup
        let mut s = ids.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), ids.len());
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&10));
    }

    #[test]
    fn empty_collection_search() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default()).unwrap();
        let r = idx.search(&[0.0, 0.0, 0.0, 0.0], 5, None).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn dim_mismatch() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default()).unwrap();
        assert!(idx.upsert(1, vec![1.0, 2.0], None).is_err());
        assert!(idx.search(&[1.0, 2.0], 1, None).is_err());
    }

    #[test]
    fn duplicate_upsert_replaces() {
        let idx = HnswIndex::new(2, Metric::L2, HnswParams::default()).unwrap();
        idx.upsert(1, vec![0.0, 0.0], None).unwrap();
        idx.upsert(1, vec![10.0, 10.0], None).unwrap();
        assert_eq!(idx.count(), 1);
        let r = idx.search(&[10.0, 10.0], 1, None).unwrap();
        assert_eq!(r[0].0, 1);
    }

    #[test]
    fn k_zero_and_large() {
        let idx = HnswIndex::new(2, Metric::L2, HnswParams::default()).unwrap();
        idx.upsert(1, vec![0.0, 0.0], None).unwrap();
        let r = idx.search(&[0.0, 0.0], 0, None).unwrap();
        assert!(r.is_empty());
        let r = idx.search(&[0.0, 0.0], 100, None).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn zero_vector_cosine() {
        let idx = HnswIndex::new(2, Metric::Cosine, HnswParams::default()).unwrap();
        idx.upsert(1, vec![0.0, 0.0], None).unwrap();
        idx.upsert(2, vec![1.0, 0.0], None).unwrap();
        let r = idx.search(&[1.0, 0.0], 2, None).unwrap();
        assert!(!r.is_empty());
    }

    #[test]
    fn recall_vs_flat() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let n = 2000usize;
        let dim = 16;
        let idx = HnswIndex::new(
            dim,
            Metric::L2,
            HnswParams {
                ef_search: 64,
                ..HnswParams::default()
            },
        )
        .unwrap();
        let mut flat = FlatIndex::new(dim, Metric::L2).unwrap();
        let mut vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            vecs.push(v.clone());
            idx.upsert(i as u64, v.clone(), None).unwrap();
            flat.upsert(i as u64, v, None).unwrap();
        }
        // queries
        let mut total_recall = 0.0;
        let qcount = 50;
        for _ in 0..qcount {
            let q: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            let hr = idx.search(&q, 10, Some(64)).unwrap();
            let fr = flat.search(&q, 10, None).unwrap();
            let hset: std::collections::HashSet<u64> = hr.iter().map(|x| x.0).collect();
            let inter = fr.iter().filter(|x| hset.contains(&x.0)).count();
            total_recall += inter as f64 / 10.0;
        }
        let recall = total_recall / qcount as f64;
        assert!(recall >= 0.95, "recall {} too low", recall);
    }

    #[test]
    #[ignore]
    fn recall_vs_flat_100k() {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let n = 100000usize;
        let dim = 16;
        let idx = HnswIndex::new(
            dim,
            Metric::L2,
            HnswParams {
                ef_search: 64,
                ..HnswParams::default()
            },
        )
        .unwrap();
        let mut flat = FlatIndex::new(dim, Metric::L2).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            idx.upsert(i as u64, v.clone(), None).unwrap();
            flat.upsert(i as u64, v, None).unwrap();
        }
        let mut total_recall = 0.0;
        let qcount = 100;
        for _ in 0..qcount {
            let q: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let hr = idx.search(&q, 10, Some(64)).unwrap();
            let fr = flat.search(&q, 10, None).unwrap();
            let hset: std::collections::HashSet<u64> = hr.iter().map(|x| x.0).collect();
            let inter = fr.iter().filter(|x| hset.contains(&x.0)).count();
            total_recall += inter as f64 / 10.0;
        }
        let recall = total_recall / qcount as f64;
        assert!(recall >= 0.95, "recall {} too low", recall);
    }

    proptest! {
        #[test]
        fn prop_sorted_no_dup(n in 5usize..40usize, qseed in 0u64..1000u64) {
            let idx = HnswIndex::new(4, Metric::L2, HnswParams::default()).unwrap();
            let mut rng = rand::rngs::StdRng::seed_from_u64(qseed);
            for i in 0..n {
                let v: Vec<f32> = (0..4).map(|_| rand::Rng::random::<f32>(&mut rng)).collect();
                idx.upsert(i as u64, v, None).unwrap();
            }
            let q: Vec<f32> = (0..4).map(|_| rand::Rng::random::<f32>(&mut rng)).collect();
            let r = idx.search(&q, 5, None).unwrap();
            for w in r.windows(2) {
                prop_assert!(w[0].1 <= w[1].1 + 1e-6);
            }
            let mut ids: Vec<u64> = r.iter().map(|x| x.0).collect();
            ids.sort(); ids.dedup();
            prop_assert_eq!(ids.len(), r.len());
        }
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn concurrent_search_insert() {
        let idx = Arc::new(HnswIndex::new(4, Metric::L2, HnswParams::default()).unwrap());
        // insert initial
        for i in 0..100u64 {
            idx.upsert(i, vec![i as f32, 0.0, 0.0, 0.0], None).unwrap();
        }
        let idx2 = idx.clone();
        let inserter = thread::spawn(move || {
            for i in 100..300u64 {
                idx2.upsert(i, vec![i as f32, 0.0, 0.0, 0.0], None).unwrap();
            }
        });
        let mut handles = vec![];
        for _ in 0..4 {
            let idxc = idx.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let _ = idxc.search(&[50.0, 0.0, 0.0, 0.0], 5, None);
                }
            }));
        }
        inserter.join().unwrap();
        for h in handles {
            h.join().unwrap();
        }
        assert!(idx.count() >= 300);
    }
}

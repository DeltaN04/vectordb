//! Storage engine: segments, WAL, mmap, payloads, graph persistence.

#![deny(unsafe_code)]

mod payload;
mod segment;
mod wal;
pub use payload::PayloadStore;
pub use segment::Segment;
pub use wal::{FsyncPolicy, Wal, WalOp};

use crate::distance::Metric;
use crate::error::{Error, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Record {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Option<Value>,
    pub deleted: bool,
}

#[derive(Debug)]
pub struct MutableSegment {
    pub records: Vec<Record>,
}

impl MutableSegment {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}
impl Default for MutableSegment {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct StorageEngine {
    pub dir: PathBuf,
    pub dim: usize,
    pub metric: Metric,
    pub policy: FsyncPolicy,
    wal: Wal,
    pub mutable: MutableSegment,
    pub segments: Vec<Segment>,
    pub payloads: PayloadStore,
    graph_path: PathBuf,
}

impl StorageEngine {
    pub fn open(dir: &Path, dim: usize, metric: Metric, policy: FsyncPolicy) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let wal_path = dir.join("wal.log");
        let wal = Wal::open(wal_path.clone(), policy)?;
        let payloads = PayloadStore::open(&dir.join("payloads.redb"))?;
        let graph_path = dir.join("hnsw.graph");

        // load segments
        let mut segments = Vec::new();
        let seg_dir = dir.join("segments");
        std::fs::create_dir_all(&seg_dir)?;
        let mut entries: Vec<_> = std::fs::read_dir(&seg_dir)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("seg") {
                segments.push(Segment::open(&p)?);
            }
        }

        let mut engine = Self {
            dir: dir.to_path_buf(),
            dim,
            metric,
            policy,
            wal,
            mutable: MutableSegment::new(),
            segments,
            payloads,
            graph_path,
        };

        // replay WAL
        let ops = Wal::replay(&wal_path)?;
        for op in ops {
            match op {
                WalOp::Upsert {
                    id,
                    vector,
                    payload,
                } => {
                    if let Some(p) = &payload {
                        let _ = engine.payloads.put(id, p);
                    }
                    // update mutable: remove existing
                    engine.mutable.records.retain(|r| r.id != id);
                    engine.mutable.records.push(Record {
                        id,
                        vector,
                        payload,
                        deleted: false,
                    });
                }
                WalOp::Delete { id } => {
                    let _ = engine.payloads.delete(id);
                    engine.mutable.records.retain(|r| r.id != id);
                    engine.mutable.records.push(Record {
                        id,
                        vector: Vec::new(),
                        payload: None,
                        deleted: true,
                    });
                }
            }
        }
        Ok(engine)
    }

    pub fn upsert(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.wal.append(&WalOp::Upsert {
            id,
            vector: vector.clone(),
            payload: payload.clone(),
        })?;
        if let Some(p) = &payload {
            let _ = self.payloads.put(id, p);
        }
        self.mutable.records.retain(|r| r.id != id);
        self.mutable.records.push(Record {
            id,
            vector,
            payload,
            deleted: false,
        });
        Ok(())
    }

    pub fn delete(&mut self, id: u64) -> Result<bool> {
        self.wal.append(&WalOp::Delete { id })?;
        let _ = self.payloads.delete(id);
        self.mutable.records.retain(|r| r.id != id);
        self.mutable.records.push(Record {
            id,
            vector: Vec::new(),
            payload: None,
            deleted: true,
        });
        Ok(true)
    }

    /// All live records across segments + mutable (mutable overrides by id).
    pub fn all_records(&self) -> Vec<Record> {
        let mut map: std::collections::HashMap<u64, Record> = std::collections::HashMap::new();
        for seg in &self.segments {
            let payloads = seg.payloads();
            for i in 0..seg.count() {
                let id = seg.id(i);
                if seg.deleted(i) {
                    map.insert(
                        id,
                        Record {
                            id,
                            vector: Vec::new(),
                            payload: None,
                            deleted: true,
                        },
                    );
                    continue;
                }
                let v = seg.vector(i).to_vec();
                let p = payloads.get(i).cloned().flatten();
                map.insert(
                    id,
                    Record {
                        id,
                        vector: v,
                        payload: p,
                        deleted: false,
                    },
                );
            }
        }
        for r in &self.mutable.records {
            map.insert(r.id, r.clone());
        }
        map.into_values().collect()
    }

    pub fn live_records(&self) -> Vec<Record> {
        self.all_records()
            .into_iter()
            .filter(|r| !r.deleted)
            .collect()
    }

    /// Flush mutable segment to a new immutable segment and clear mutable + WAL.
    pub fn flush(&mut self) -> Result<()> {
        if self.mutable.records.is_empty() {
            return Ok(());
        }
        let seg_dir = self.dir.join("segments");
        std::fs::create_dir_all(&seg_dir)?;
        let seg_name = format!("seg_{:04}.seg", self.segments.len());
        let seg_path = seg_dir.join(seg_name);
        let recs: Vec<(u64, Vec<f32>, Option<Value>, bool)> = self
            .mutable
            .records
            .iter()
            .map(|r| (r.id, r.vector.clone(), r.payload.clone(), r.deleted))
            .collect();
        Segment::write(&seg_path, self.dim, &recs)?;
        self.segments.push(Segment::open(&seg_path)?);
        self.mutable.records.clear();
        // truncate WAL
        let wal_path = self.dir.join("wal.log");
        // close and recreate
        self.wal = Wal::open(wal_path.clone(), self.policy)?;
        // truncate file
        // reopen with truncate
        let f = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&wal_path)?;
        drop(f);
        // reopen append
        self.wal = Wal::open(wal_path, self.policy)?;
        Ok(())
    }

    pub fn compact(&mut self) -> Result<()> {
        self.flush()?;
        // merge all segments into one: read all live records, rewrite single segment
        let live = self.live_records();
        let seg_dir = self.dir.join("segments");
        // remove existing seg files
        for e in std::fs::read_dir(&seg_dir)?.flatten() {
            if e.path().extension().and_then(|s| s.to_str()) == Some("seg") {
                let _ = std::fs::remove_file(e.path());
            }
        }
        self.segments.clear();
        let recs: Vec<(u64, Vec<f32>, Option<Value>, bool)> = live
            .iter()
            .map(|r| (r.id, r.vector.clone(), r.payload.clone(), false))
            .collect();
        let seg_path = seg_dir.join("seg_0000.seg");
        Segment::write(&seg_path, self.dim, &recs)?;
        self.segments.push(Segment::open(&seg_path)?);
        Ok(())
    }

    pub fn save_graph(&self, data: &[u8]) -> Result<()> {
        std::fs::write(&self.graph_path, data)?;
        Ok(())
    }

    pub fn load_graph(&self) -> Result<Option<Vec<u8>>> {
        match std::fs::read(&self.graph_path) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Io(e)),
        }
    }

    pub fn fsync_policy(&self) -> FsyncPolicy {
        self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn wal_replay() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("c");
        {
            let mut e = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::EveryWrite).unwrap();
            e.upsert(1, vec![1.0, 2.0], None).unwrap();
            e.upsert(2, vec![3.0, 4.0], None).unwrap();
            e.delete(1).unwrap();
        }
        let e = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::EveryWrite).unwrap();
        let recs = e.all_records();
        assert!(recs.iter().any(|r| r.id == 2 && !r.deleted));
    }

    #[test]
    fn flush_and_compact() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("c");
        let mut e = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::None).unwrap();
        e.upsert(1, vec![1.0, 2.0], None).unwrap();
        e.upsert(2, vec![3.0, 4.0], None).unwrap();
        e.flush().unwrap();
        assert_eq!(e.segments.len(), 1);
        e.upsert(3, vec![5.0, 6.0], None).unwrap();
        e.compact().unwrap();
        let live = e.live_records();
        assert_eq!(live.len(), 3);
    }

    #[test]
    fn partial_wal_ignored() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("c");
        let mut e = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::None).unwrap();
        e.upsert(1, vec![1.0, 2.0], None).unwrap();
        drop(e);
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(p.join("wal.log"))
            .unwrap();
        f.write_all(&[0u8; 5]).unwrap();
        drop(f);
        let e2 = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::None).unwrap();
        assert!(e2.all_records().iter().any(|r| r.id == 1));
    }

    #[test]
    fn segment_wrong_magic() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("bad.seg");
        std::fs::write(
            &p,
            b"NOTASEG\0\0\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00",
        )
        .unwrap();
        assert!(Segment::open(&p).is_err());
    }

    #[test]
    fn wal_truncate_recovery() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("c");
        let mut e = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::EveryWrite).unwrap();
        for i in 0..5u64 {
            e.upsert(i, vec![i as f32, 0.0], None).unwrap();
        }
        drop(e);
        let wal_path = p.join("wal.log");
        let data = std::fs::read(&wal_path).unwrap();
        // truncate at every offset; recovery must not panic
        for cut in 0..=data.len() {
            let mut d = data.clone();
            d.truncate(cut);
            std::fs::write(&wal_path, &d).unwrap();
            let _ = StorageEngine::open(&p, 2, Metric::L2, FsyncPolicy::EveryWrite);
        }
        // restore original
        std::fs::write(&wal_path, &data).unwrap();
    }

    #[test]
    fn graph_roundtrip() {
        let idx = crate::index::HnswIndex::new(2, Metric::L2, crate::index::HnswParams::default())
            .unwrap();
        idx.upsert(1, vec![1.0, 2.0], None).unwrap();
        let bytes = idx.to_bytes().unwrap();
        let idx2 = crate::index::HnswIndex::from_bytes(&bytes).unwrap();
        assert_eq!(idx2.count(), 1);
    }
}

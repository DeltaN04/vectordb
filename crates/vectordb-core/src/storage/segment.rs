//! Immutable on-disk segment with mmap'd vector data.

#![allow(unsafe_code)]

use crate::error::{Error, Result};
use bytemuck;
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 8] = b"VDBSEG\0\0";
pub const VERSION: u32 = 1;

#[derive(Debug)]
pub struct Segment {
    _path: PathBuf,
    mmap: Mmap,
    dim: usize,
    count: usize,
    // offsets within mmap for id array and vector array
    ids_offset: usize,
    vecs_offset: usize,
    deleted_offset: usize,
}

impl Segment {
    /// Write a segment file from records.
    pub fn write(
        path: &Path,
        dim: usize,
        records: &[(u64, Vec<f32>, Option<serde_json::Value>, bool)],
    ) -> Result<()> {
        use std::io::Write;
        let mut file = File::create(path)?;
        // header
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(dim as u32).to_le_bytes())?;
        file.write_all(&(records.len() as u32).to_le_bytes())?;
        // placeholder checksum
        let mut hasher = crc32fast::Hasher::new();
        // ids
        let mut ids_bytes = Vec::with_capacity(records.len() * 8);
        for (id, _, _, _) in records {
            ids_bytes.extend_from_slice(&id.to_le_bytes());
        }
        // deleted flags
        let mut del_bytes = Vec::with_capacity(records.len());
        for (_, _, _, d) in records {
            del_bytes.push(if *d { 1u8 } else { 0u8 });
        }
        // vectors
        let mut vec_bytes = Vec::with_capacity(records.len() * dim * 4);
        for (_, v, _, _) in records {
            for &x in v {
                vec_bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        // payloads: store lengths + json in a trailing section; for simplicity store concatenated with offsets?
        // We'll store payloads as JSON array at end.
        let payload_json =
            serde_json::to_vec(&records.iter().map(|r| r.2.clone()).collect::<Vec<_>>())
                .map_err(Error::Serde)?;

        // write ids
        let ids_off = 24usize; // after header (8+4+4+4+4? let's compute: magic8 + version4 + dim4 + count4 =20) Actually 8+4+4+4=20
                               // We'll just write sequentially; track offsets by file position not needed for mmap? Need offsets. Compute:
                               // header size = 8+4+4+4 = 20
        file.write_all(&ids_bytes)?;
        file.write_all(&del_bytes)?;
        // pad deleted flags to 4-byte alignment for vector array
        let pad = (4 - (del_bytes.len() % 4)) % 4;
        file.write_all(&vec![0u8; pad])?;
        file.write_all(&vec_bytes)?;
        // payload section: u32 len + bytes
        let plen_bytes = (payload_json.len() as u32).to_le_bytes();
        file.write_all(&plen_bytes)?;
        file.write_all(&payload_json)?;

        hasher.update(&ids_bytes);
        hasher.update(&del_bytes);
        let pad = (4 - (del_bytes.len() % 4)) % 4;
        hasher.update(&vec![0u8; pad]);
        hasher.update(&vec_bytes);
        hasher.update(&plen_bytes);
        hasher.update(&payload_json);
        let crc = hasher.finalize();
        file.write_all(&crc.to_le_bytes())?;
        file.flush()?;
        // write offsets not stored; compute on read by formula. Good.
        let _ = ids_off;
        Ok(())
    }

    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: mapping a read-only file; file is not modified externally during use.
        let mmap = unsafe { Mmap::map(&file)? };
        // parse header
        if mmap.len() < 20 {
            return Err(Error::InvalidFormat("segment too small".into()));
        }
        if &mmap[0..8] != MAGIC {
            return Err(Error::InvalidFormat("bad magic".into()));
        }
        let version = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]);
        if version != VERSION {
            return Err(Error::InvalidFormat("bad version".into()));
        }
        let dim = u32::from_le_bytes([mmap[12], mmap[13], mmap[14], mmap[15]]) as usize;
        let count = u32::from_le_bytes([mmap[16], mmap[17], mmap[18], mmap[19]]) as usize;
        // verify checksum
        if mmap.len() < 24 {
            return Err(Error::InvalidFormat("segment too small".into()));
        }
        let stored_crc = u32::from_le_bytes([
            mmap[mmap.len() - 4],
            mmap[mmap.len() - 3],
            mmap[mmap.len() - 2],
            mmap[mmap.len() - 1],
        ]);
        let mut h = crc32fast::Hasher::new();
        h.update(&mmap[20..mmap.len() - 4]);
        if h.finalize() != stored_crc {
            return Err(Error::Checksum);
        }
        let ids_offset = 20usize;
        let deleted_offset = ids_offset + count * 8;
        let del_pad = (4 - (count % 4)) % 4;
        let vecs_offset = deleted_offset + count + del_pad;
        // verify checksum? skip for now but compute
        Ok(Self {
            _path: path.to_path_buf(),
            mmap,
            dim,
            count,
            ids_offset,
            vecs_offset,
            deleted_offset,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn id(&self, i: usize) -> u64 {
        if i >= self.count {
            return 0;
        }
        let off = self.ids_offset + i * 8;
        u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap_or([0; 8]))
    }

    pub fn deleted(&self, i: usize) -> bool {
        if i >= self.count {
            return false;
        }
        let off = self.deleted_offset + i;
        self.mmap[off] != 0
    }

    pub fn vector(&self, i: usize) -> &[f32] {
        if i >= self.count {
            return &[];
        }
        let off = self.vecs_offset + i * self.dim * 4;
        let bytes = &self.mmap[off..off + self.dim * 4];
        bytemuck::cast_slice(bytes)
    }

    pub fn payloads(&self) -> Vec<Option<serde_json::Value>> {
        // payload section after vectors
        let base = self.vecs_offset + self.count * self.dim * 4;
        if base + 4 > self.mmap.len() {
            return vec![None; self.count];
        }
        let plen = u32::from_le_bytes([
            self.mmap[base],
            self.mmap[base + 1],
            self.mmap[base + 2],
            self.mmap[base + 3],
        ]) as usize;
        let bytes = &self.mmap[base + 4..base + 4 + plen];
        serde_json::from_slice(bytes).unwrap_or(vec![None; self.count])
    }
}

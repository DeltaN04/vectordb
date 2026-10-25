//! Write-ahead log with CRC32 per record and framing.

#![deny(unsafe_code)]

use crate::error::{Error, Result};
use crc32fast::Hasher;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsyncPolicy {
    EveryWrite,
    None,
}

#[derive(Debug, Clone)]
pub enum WalOp {
    Upsert {
        id: u64,
        vector: Vec<f32>,
        payload: Option<serde_json::Value>,
    },
    Delete {
        id: u64,
    },
}

#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
    file: File,
    policy: FsyncPolicy,
}

impl Wal {
    pub fn open(path: PathBuf, policy: FsyncPolicy) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self { path, file, policy })
    }

    fn encode(op: &WalOp) -> Result<Vec<u8>> {
        let mut buf: Vec<u8> = Vec::new();
        match op {
            WalOp::Upsert {
                id,
                vector,
                payload,
            } => {
                buf.push(1);
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                for &x in vector {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                let pbytes = match payload {
                    Some(v) => serde_json::to_vec(v).map_err(Error::Serde)?,
                    None => Vec::new(),
                };
                buf.extend_from_slice(&(pbytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(&pbytes);
            }
            WalOp::Delete { id } => {
                buf.push(2);
                buf.extend_from_slice(&id.to_le_bytes());
            }
        }
        Ok(buf)
    }

    pub fn append(&mut self, op: &WalOp) -> Result<()> {
        let buf = Self::encode(op)?;
        let mut h = Hasher::new();
        h.update(&buf);
        let crc = h.finalize();
        let len = (buf.len() + 4) as u32;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&buf)?;
        self.file.write_all(&crc.to_le_bytes())?;
        if self.policy == FsyncPolicy::EveryWrite {
            self.file.sync_all()?;
        }
        Ok(())
    }

    /// Read all valid records from the WAL (stops at first corrupt/partial record).
    pub fn replay(path: &std::path::Path) -> Result<Vec<WalOp>> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::Io(e)),
        };
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        let mut ops = Vec::new();
        let mut pos = 0;
        while pos + 4 <= data.len() {
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if len < 5 || pos + len > data.len() {
                break; // partial/corrupt record, stop
            }
            let buf = &data[pos..pos + len - 4];
            let crc_bytes = &data[pos + len - 4..pos + len];
            let crc = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
            let mut h = Hasher::new();
            h.update(buf);
            if h.finalize() != crc {
                break; // corrupt, stop
            }
            // decode
            let mut i = 0;
            if buf.is_empty() {
                break;
            }
            let tag = buf[i];
            i += 1;
            if tag == 1 {
                // upsert
                if i + 8 + 4 > buf.len() {
                    break;
                }
                let id = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap_or([0; 8]));
                i += 8;
                let dim = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap_or([0; 4])) as usize;
                i += 4;
                if i + dim * 4 > buf.len() {
                    break;
                }
                let mut vector = Vec::with_capacity(dim);
                for _ in 0..dim {
                    let x = f32::from_le_bytes(buf[i..i + 4].try_into().unwrap_or([0; 4]));
                    vector.push(x);
                    i += 4;
                }
                if i + 4 > buf.len() {
                    break;
                }
                let plen = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap_or([0; 4])) as usize;
                i += 4;
                if i + plen > buf.len() {
                    break;
                }
                let payload = if plen == 0 {
                    None
                } else {
                    serde_json::from_slice(&buf[i..i + plen]).ok()
                };
                let _ = i;
                ops.push(WalOp::Upsert {
                    id,
                    vector,
                    payload,
                });
            } else if tag == 2 {
                if i + 8 > buf.len() {
                    break;
                }
                let id = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap_or([0; 8]));
                ops.push(WalOp::Delete { id });
            } else {
                break;
            }
            pos += len;
        }
        Ok(ops)
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

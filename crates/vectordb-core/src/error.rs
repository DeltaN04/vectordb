//! Error types for vectordb-core.

use thiserror::Error;

/// Errors produced by the core library.
#[derive(Debug, Error)]
pub enum Error {
    /// Dimension mismatch between two vectors.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    /// Invalid dimension (zero).
    #[error("invalid dimension: {0}")]
    InvalidDim(usize),
    /// Invalid metric name.
    #[error("unknown metric: {0}")]
    UnknownMetric(String),
    /// Vector id not found.
    #[error("not found: {0}")]
    NotFound(String),
    /// Duplicate id.
    #[error("duplicate id: {0}")]
    Duplicate(String),
    /// Storage/IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization error.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// Checksum mismatch.
    #[error("checksum mismatch")]
    Checksum,
    /// Invalid format/version.
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    /// Database error (redb).
    #[error("db error: {0}")]
    Db(String),
    /// Filter error.
    #[error("filter error: {0}")]
    Filter(String),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

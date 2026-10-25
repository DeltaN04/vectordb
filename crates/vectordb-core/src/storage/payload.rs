//! Payload (metadata) storage backed by redb.

#![deny(unsafe_code)]

use crate::error::{Error, Result};
use redb::{Database, TableDefinition};
use std::path::Path;

const TABLE: TableDefinition<&str, &str> = TableDefinition::new("payloads");

#[derive(Debug)]
pub struct PayloadStore {
    db: Database,
}

impl PayloadStore {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path).map_err(|e| Error::Db(e.to_string()))?;
        // ensure table
        let txn = db.begin_write().map_err(|e| Error::Db(e.to_string()))?;
        {
            let _ = txn
                .open_table(TABLE)
                .map_err(|e| Error::Db(e.to_string()))?;
        }
        txn.commit().map_err(|e| Error::Db(e.to_string()))?;
        Ok(Self { db })
    }

    pub fn put(&self, id: u64, value: &serde_json::Value) -> Result<()> {
        let txn = self
            .db
            .begin_write()
            .map_err(|e| Error::Db(e.to_string()))?;
        {
            let mut table = txn
                .open_table(TABLE)
                .map_err(|e| Error::Db(e.to_string()))?;
            let key = id.to_string();
            let v = serde_json::to_string(value).map_err(Error::Serde)?;
            table
                .insert(key.as_str(), v.as_str())
                .map_err(|e| Error::Db(e.to_string()))?;
        }
        txn.commit().map_err(|e| Error::Db(e.to_string()))?;
        Ok(())
    }

    pub fn get(&self, id: u64) -> Result<Option<serde_json::Value>> {
        let txn = self.db.begin_read().map_err(|e| Error::Db(e.to_string()))?;
        let table = txn
            .open_table(TABLE)
            .map_err(|e| Error::Db(e.to_string()))?;
        let key = id.to_string();
        match table
            .get(key.as_str())
            .map_err(|e| Error::Db(e.to_string()))?
        {
            Some(v) => {
                let val: serde_json::Value =
                    serde_json::from_str(v.value()).map_err(Error::Serde)?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    pub fn delete(&self, id: u64) -> Result<()> {
        let txn = self
            .db
            .begin_write()
            .map_err(|e| Error::Db(e.to_string()))?;
        {
            let mut table = txn
                .open_table(TABLE)
                .map_err(|e| Error::Db(e.to_string()))?;
            let key = id.to_string();
            let _ = table
                .remove(key.as_str())
                .map_err(|e| Error::Db(e.to_string()))?;
        }
        txn.commit().map_err(|e| Error::Db(e.to_string()))?;
        Ok(())
    }
}

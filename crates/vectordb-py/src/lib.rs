use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::path::PathBuf;
use vectordb_core::collection::Collection as CoreCollection;
use vectordb_core::distance::Metric;
use vectordb_core::index::HnswParams;
use vectordb_core::storage::FsyncPolicy;

#[pyclass(name = "Collection")]
struct PyCollection {
    inner: CoreCollection,
}

#[pymethods]
impl PyCollection {
    #[staticmethod]
    fn create(path: String, name: String, dim: usize, metric: Option<String>) -> PyResult<Self> {
        let m = metric.as_deref().unwrap_or("l2");
        let metric = Metric::parse(m)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let dir = PathBuf::from(path);
        let c = CoreCollection::create(
            &dir,
            name,
            dim,
            metric,
            HnswParams::default(),
            FsyncPolicy::None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: c })
    }

    fn upsert(&mut self, id: u64, vector: Vec<f32>, payload_json: Option<String>) -> PyResult<()> {
        let p = payload_json.and_then(|s| serde_json::from_str(&s).ok());
        self.inner
            .upsert(id, vector, p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn upsert_numpy(&mut self, id: u64, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let v = vector
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyBufferError, _>(e.to_string()))?
            .to_vec();
        self.inner
            .upsert(id, v, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn search(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        let res = self
            .inner
            .search(&vector, k, None, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(res.into_iter().map(|r| (r.id, r.score)).collect())
    }

    fn count(&self) -> usize {
        self.inner.count()
    }
}

#[pymodule]
fn vectordb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCollection>()?;
    Ok(())
}

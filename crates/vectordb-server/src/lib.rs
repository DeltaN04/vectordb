//! vectordb server.

#![allow(missing_docs)]

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use vectordb_core::collection::Collection;
use vectordb_core::distance::Metric;
use vectordb_core::index::HnswParams;
use vectordb_core::storage::FsyncPolicy;

#[derive(Clone)]
pub struct AppState {
    pub data_dir: PathBuf,
    pub collections: Arc<RwLock<HashMap<String, Collection>>>,
    pub api_key: Option<String>,
}

#[derive(Deserialize)]
struct CreateCollectionReq {
    name: String,
    dim: usize,
    metric: Option<String>,
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics() -> String {
    String::new()
}

async fn create_collection(
    State(st): State<AppState>,
    Json(req): Json<CreateCollectionReq>,
) -> impl IntoResponse {
    let metric = req.metric.as_deref().unwrap_or("l2");
    let m = Metric::parse(metric).unwrap_or(Metric::L2);
    match Collection::create(
        &st.data_dir,
        req.name.clone(),
        req.dim,
        m,
        HnswParams::default(),
        FsyncPolicy::None,
    ) {
        Ok(c) => {
            st.collections.write().await.insert(req.name, c);
            StatusCode::CREATED.into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct UpsertReq {
    id: u64,
    vector: Vec<f32>,
    payload: Option<serde_json::Value>,
}

async fn upsert(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertReq>,
) -> impl IntoResponse {
    let mut cols = st.collections.write().await;
    if let Some(c) = cols.get_mut(&name) {
        match c.upsert(req.id, req.vector, req.payload) {
            Ok(_) => StatusCode::OK.into_response(),
            Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

#[derive(Deserialize)]
struct SearchReq {
    vector: Vec<f32>,
    k: Option<usize>,
    ef_search: Option<usize>,
}

#[derive(Serialize)]
struct SearchRespItem {
    id: u64,
    score: f32,
    payload: Option<serde_json::Value>,
}

async fn search(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SearchReq>,
) -> impl IntoResponse {
    let cols = st.collections.read().await;
    if let Some(c) = cols.get(&name) {
        let k = req.k.unwrap_or(10);
        let payload_filter = None::<vectordb_core::filter::PayloadFilter>;
        match c.search(&req.vector, k, payload_filter, req.ef_search) {
            Ok(res) => {
                let items: Vec<SearchRespItem> = res
                    .into_iter()
                    .map(|r| SearchRespItem {
                        id: r.id,
                        score: r.score,
                        payload: r.payload,
                    })
                    .collect();
                Json(items).into_response()
            }
            Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        }
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn count(State(st): State<AppState>, Path(name): Path<String>) -> impl IntoResponse {
    let cols = st.collections.read().await;
    if let Some(c) = cols.get(&name) {
        Json(c.count()).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub fn build_router(st: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/collections", post(create_collection))
        .route("/collections/{name}/upsert", post(upsert))
        .route("/collections/{name}/search", post(search))
        .route("/collections/{name}/count", get(count))
        .with_state(st)
}

pub async fn serve_http(st: AppState, addr: std::net::SocketAddr) -> Result<()> {
    let app = build_router(st);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// gRPC service implementation
pub mod grpc {
    use super::*;
    use tonic::{Request, Response, Status};
    use vectordb_core::collection::Collection;
    // generated
    pub mod pb {
        tonic::include_proto!("vectordb");
    }

    pub struct VdbService {
        pub st: AppState,
    }

    #[tonic::async_trait]
    impl pb::vector_db_server::VectorDb for VdbService {
        async fn create_collection(
            &self,
            request: Request<pb::CreateCollectionRequest>,
        ) -> Result<Response<pb::Empty>, Status> {
            let r = request.into_inner();
            let metric = if r.metric.is_empty() {
                "l2"
            } else {
                r.metric.as_str()
            };
            let m = Metric::parse(metric).unwrap_or(Metric::L2);
            let mut cols = self.st.collections.write().await;
            match Collection::create(
                &self.st.data_dir,
                r.name.clone(),
                r.dim as usize,
                m,
                HnswParams::default(),
                FsyncPolicy::None,
            ) {
                Ok(c) => {
                    cols.insert(r.name, c);
                    Ok(Response::new(pb::Empty {}))
                }
                Err(e) => Err(Status::invalid_argument(e.to_string())),
            }
        }

        async fn upsert(
            &self,
            request: Request<pb::UpsertRequest>,
        ) -> Result<Response<pb::Empty>, Status> {
            let r = request.into_inner();
            let payload = if r.payload_json.is_empty() {
                None
            } else {
                serde_json::from_str(&r.payload_json).ok()
            };
            let mut cols = self.st.collections.write().await;
            if let Some(c) = cols.get_mut(&r.collection) {
                match c.upsert(r.id, r.vector, payload) {
                    Ok(_) => Ok(Response::new(pb::Empty {})),
                    Err(e) => Err(Status::invalid_argument(e.to_string())),
                }
            } else {
                Err(Status::not_found("collection not found"))
            }
        }

        async fn get(
            &self,
            request: Request<pb::GetRequest>,
        ) -> Result<Response<pb::GetResponse>, Status> {
            let r = request.into_inner();
            let cols = self.st.collections.read().await;
            if let Some(c) = cols.get(&r.collection) {
                match c.get(r.id) {
                    Ok(Some((v, p))) => Ok(Response::new(pb::GetResponse {
                        vector: v,
                        payload_json: p.map(|x| x.to_string()).unwrap_or_default(),
                        found: true,
                    })),
                    Ok(None) => Ok(Response::new(pb::GetResponse {
                        found: false,
                        ..Default::default()
                    })),
                    Err(e) => Err(Status::internal(e.to_string())),
                }
            } else {
                Err(Status::not_found("collection not found"))
            }
        }

        async fn delete(
            &self,
            request: Request<pb::DeleteRequest>,
        ) -> Result<Response<pb::Empty>, Status> {
            let r = request.into_inner();
            let mut cols = self.st.collections.write().await;
            if let Some(c) = cols.get_mut(&r.collection) {
                match c.delete(r.id) {
                    Ok(_) => Ok(Response::new(pb::Empty {})),
                    Err(e) => Err(Status::internal(e.to_string())),
                }
            } else {
                Err(Status::not_found("collection not found"))
            }
        }

        async fn search(
            &self,
            request: Request<pb::SearchRequest>,
        ) -> Result<Response<pb::SearchResponse>, Status> {
            let r = request.into_inner();
            let cols = self.st.collections.read().await;
            if let Some(c) = cols.get(&r.collection) {
                let k = if r.k == 0 { 10 } else { r.k as usize };
                let filter = None;
                match c.search(
                    &r.vector,
                    k,
                    filter,
                    if r.ef_search == 0 {
                        None
                    } else {
                        Some(r.ef_search as usize)
                    },
                ) {
                    Ok(res) => {
                        let results = res
                            .into_iter()
                            .map(|x| pb::SearchResult {
                                id: x.id,
                                score: x.score,
                                payload_json: x.payload.map(|p| p.to_string()).unwrap_or_default(),
                            })
                            .collect();
                        Ok(Response::new(pb::SearchResponse { results }))
                    }
                    Err(e) => Err(Status::internal(e.to_string())),
                }
            } else {
                Err(Status::not_found("collection not found"))
            }
        }

        async fn count(
            &self,
            request: Request<pb::CountRequest>,
        ) -> Result<Response<pb::CountResponse>, Status> {
            let r = request.into_inner();
            let cols = self.st.collections.read().await;
            if let Some(c) = cols.get(&r.collection) {
                Ok(Response::new(pb::CountResponse {
                    count: c.count() as u64,
                }))
            } else {
                Err(Status::not_found("collection not found"))
            }
        }

        async fn health(
            &self,
            _request: Request<pb::Empty>,
        ) -> Result<Response<pb::HealthResponse>, Status> {
            Ok(Response::new(pb::HealthResponse {
                status: "ok".into(),
            }))
        }
    }
}

pub async fn serve_grpc(st: AppState, addr: std::net::SocketAddr) -> Result<()> {
    use grpc::pb::vector_db_server::VectorDbServer;
    let svc = grpc::VdbService { st };
    tonic::transport::Server::builder()
        .add_service(VectorDbServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use vectordb_server::{serve_grpc, serve_http, AppState};

#[tokio::test]
async fn http_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let st = AppState {
        data_dir: dir.path().to_path_buf(),
        collections: Arc::new(RwLock::new(std::collections::HashMap::new())),
        api_key: None,
    };
    // bind manually to get port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = vectordb_server::build_router(st.clone());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // create collection
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{}", port);
    client
        .post(format!("{}/collections", base))
        .json(&serde_json::json!({"name":"c","dim":2,"metric":"l2"}))
        .send()
        .await
        .unwrap();
    client
        .post(format!("{}/collections/c/upsert", base))
        .json(&serde_json::json!({"id":1,"vector":[0.0,0.0]}))
        .send()
        .await
        .unwrap();
    client
        .post(format!("{}/collections/c/upsert", base))
        .json(&serde_json::json!({"id":2,"vector":[1.0,1.0]}))
        .send()
        .await
        .unwrap();
    let resp = client
        .post(format!("{}/collections/c/search", base))
        .json(&serde_json::json!({"vector":[0.1,0.1],"k":2}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body[0]["id"], 1);
    let c = client
        .get(format!("{}/collections/c/count", base))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(c, "2");
    let _ = serve_http;
    let _ = serve_grpc;
}

#[tokio::test]
async fn grpc_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let st = AppState {
        data_dir: dir.path().to_path_buf(),
        collections: Arc::new(RwLock::new(std::collections::HashMap::new())),
        api_key: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let grpc_addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
    tokio::spawn(async move {
        serve_grpc(st, grpc_addr).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    // use tonic client
    let mut client = vectordb_server::grpc::pb::vector_db_client::VectorDbClient::connect(format!(
        "http://127.0.0.1:{}",
        port
    ))
    .await
    .unwrap();
    client
        .create_collection(tonic::Request::new(
            vectordb_server::grpc::pb::CreateCollectionRequest {
                name: "c".into(),
                dim: 2,
                metric: "l2".into(),
            },
        ))
        .await
        .unwrap();
    client
        .upsert(tonic::Request::new(
            vectordb_server::grpc::pb::UpsertRequest {
                collection: "c".into(),
                id: 1,
                vector: vec![0.0, 0.0],
                payload_json: String::new(),
            },
        ))
        .await
        .unwrap();
    let resp = client
        .search(tonic::Request::new(
            vectordb_server::grpc::pb::SearchRequest {
                collection: "c".into(),
                vector: vec![0.1, 0.1],
                k: 1,
                filter_json: String::new(),
                ef_search: 0,
            },
        ))
        .await
        .unwrap();
    assert_eq!(resp.into_inner().results[0].id, 1);
}

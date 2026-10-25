FROM rust:1.95 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p vectordb-server
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/vectordb-server /usr/local/bin/vectordb-server
EXPOSE 8080 50051
CMD ["vectordb-server"]

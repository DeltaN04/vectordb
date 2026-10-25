# AGENTS.md

## Project rules

- Rust stable, edition 2021, Cargo workspace.
- No `unwrap()` or `expect()` in library code. Use `thiserror` for library errors, `anyhow` only in binaries.
- `unsafe` only inside the SIMD distance module, each block with a `// SAFETY:` comment, and a safe scalar fallback behind runtime CPU feature detection.
- Every public item has rustdoc. Every module has unit tests. Core invariants have `proptest` property tests.
- Use `tracing` for logs, never `println!`.
- Keep dependencies minimal and justified in DEPENDENCIES.md.
- Correctness before speed: measure with `criterion` before optimizing, record results.
- If a requirement is ambiguous, choose simplest reasonable option, record in DECISIONS.md.

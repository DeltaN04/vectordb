# FORMAT.md

## Segment file (.seg)

Header (20 bytes): magic "VDBSEG\0\0" (8), version u32 (4), dim u32 (4), count u32 (4).
Then ids: count * u64.
Then deleted flags: count bytes, padded to 4-byte alignment.
Then vectors: count * dim * f32 (little-endian).
Then payload section: u32 length + JSON array of payloads.
Then CRC32 u32 over ids+deleted+padding+vectors+payloads.

## WAL

Records: u32 len, payload bytes, crc32 u32. Payload: tag + fields.

## HNSW graph

Magic "VDBSEG\0\0", version u32, u32 json length, JSON, crc32.
EOF
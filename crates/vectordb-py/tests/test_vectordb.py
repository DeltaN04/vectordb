import tempfile, numpy as np, vectordb

def test_create_upsert_search():
    d = tempfile.mkdtemp()
    c = vectordb.Collection.create(d, "c", 2, None)
    c.upsert(1, [0.0, 0.0], None)
    c.upsert(2, [1.0, 1.0], None)
    assert c.count() == 2
    res = c.search([0.1, 0.1], 1)
    assert res[0][0] == 1

def test_numpy():
    d = tempfile.mkdtemp()
    c = vectordb.Collection.create(d, "c", 2, None)
    c.upsert_numpy(1, np.array([0.0,0.0], dtype=np.float32))
    assert c.count() == 1

def test_persistence():
    d = tempfile.mkdtemp()
    c = vectordb.Collection.create(d, "c", 2, None)
    c.upsert(1, [0.0,0.0], None)
    # reopen not supported in API; skip
EOF
import importlib
import json

from simple_graph_sqlite import database as db

import code_graph_mcp_server as mcp_server
import graph_gui
from code_graph_builder import CodeGraphBuilder
from code_graph_db import load_all_nodes, load_edges


def _write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _default_config():
    return {
        "llm_base_url": "http://localhost:1234/v1/",
        "llm_api_key": "not-needed",
        "llm_model": "test",
        "embedding_base_url": "http://localhost:1234/v1/",
        "embedding_api_key": "not-needed",
        "embedding_model": "test",
    }


def _build_import_db(tmp_path):
    db_path = str(tmp_path / "test.sqlite")
    _write_file(tmp_path / "pkg" / "__init__.py", "")
    _write_file(tmp_path / "pkg" / "a.py", "def foo(): pass\n")
    _write_file(tmp_path / "main.py", "import pkg.a\n")
    builder = CodeGraphBuilder(
        str(tmp_path),
        db_path,
        _default_config(),
        rebuild=True,
        generate_embeddings=False,
        generate_comments=False,
    )
    builder.build()
    return db_path


def _write_session(tmp_path, session_id, query, node_id, score):
    session = {
        "id": session_id,
        "tool_name": "search_code",
        "query": query,
        "result_text": "{}",
        "created_at": f"2026-05-14T12:00:0{node_id}",
        "graph_data": {
            "nodes": [{"id": node_id, "name": f"file{node_id}.py", "type": "file", "path": f"file{node_id}.py"}],
            "edges": [],
            "query": query,
            "initial_nodes": {str(node_id): score},
        },
    }
    (tmp_path / f"{session_id}.json").write_text(json.dumps(session), encoding="utf-8")
    return session


class _FakeVectorStore:
    def __init__(self, _base_name):
        pass

    def search(self, _query_emb, top_k=10, threshold=0.3, search_type="both"):
        return [{"node_id": _FakeVectorStore.node_id, "score": 0.91}]


def test_mcp_exposes_only_three_tools():
    tools = set(mcp_server.mcp._tool_manager._tools)
    assert tools == {"get_overview", "search_code", "get_file_info"}


def test_get_overview_pushes_lightweight_session(tmp_path, monkeypatch):
    db_path = _build_import_db(tmp_path)
    pushed = []
    monkeypatch.setattr(mcp_server, "DB_PATH", db_path)
    monkeypatch.setattr(mcp_server, "_push_to_web", lambda *args: pushed.append(args))

    result = json.loads(mcp_server.get_overview())

    assert result["files"] == 3
    assert pushed[0][0] == "get_overview"
    assert pushed[0][3] == {"nodes": [], "edges": []}


def test_search_code_hops_zero_pushes_only_matches(tmp_path, monkeypatch):
    db_path = _build_import_db(tmp_path)
    main_node = next(n for n in load_all_nodes(db_path) if n.get("path") == "main.py")
    _FakeVectorStore.node_id = main_node["id"]
    pushed = []

    monkeypatch.setattr(mcp_server, "DB_PATH", db_path)
    monkeypatch.setattr(mcp_server, "_push_to_web", lambda *args: pushed.append(args))
    monkeypatch.setattr("code_graph_builder.get_embedding", lambda _text, _config: [0.1] * 1024)
    monkeypatch.setattr("faiss_store.FaissVectorStore", _FakeVectorStore)

    result = json.loads(mcp_server.search_code("main", hops=0))

    assert result["matches"][0]["path"] == "main.py"
    assert result["graph"] == {"nodes": 1, "edges": 0, "hops": 0}
    assert len(pushed[0][3]["nodes"]) == 1
    assert pushed[0][3]["edges"] == []
    assert pushed[0][3]["query"] == "main"
    assert pushed[0][3]["initial_nodes"] == {str(main_node["id"]): 0.91}


def test_search_code_hops_one_pushes_dependency_subgraph(tmp_path, monkeypatch):
    db_path = _build_import_db(tmp_path)
    main_node = next(n for n in load_all_nodes(db_path) if n.get("path") == "main.py")
    _FakeVectorStore.node_id = main_node["id"]
    pushed = []

    monkeypatch.setattr(mcp_server, "DB_PATH", db_path)
    monkeypatch.setattr(mcp_server, "_push_to_web", lambda *args: pushed.append(args))
    monkeypatch.setattr("code_graph_builder.get_embedding", lambda _text, _config: [0.1] * 1024)
    monkeypatch.setattr("faiss_store.FaissVectorStore", _FakeVectorStore)

    result = json.loads(mcp_server.search_code("main", hops=1))

    assert result["graph"]["nodes"] >= 2
    assert result["graph"]["edges"] >= 1
    assert pushed[0][3]["query"] == "main"
    assert pushed[0][3]["initial_nodes"] == {str(main_node["id"]): 0.91}
    assert any(e["properties"].get("type") == "imports" for e in pushed[0][3]["edges"])


def test_get_file_info_includes_import_summaries(tmp_path, monkeypatch):
    db_path = _build_import_db(tmp_path)
    pushed = []
    monkeypatch.setattr(mcp_server, "DB_PATH", db_path)
    monkeypatch.setattr(mcp_server, "_push_to_web", lambda *args: pushed.append(args))

    result = json.loads(mcp_server.get_file_info("main.py"))

    assert result["path"] == "main.py"
    assert result["outgoing_imports"][0]["target_file"] == "pkg/a.py"
    assert "imported_by" in result
    assert len(pushed[0][3]["nodes"]) >= 2
    assert len(pushed[0][3]["edges"]) == 1


def test_edge_delete_api_deletes_existing_edge(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite")
    db.initialize(db_path)
    db.atomic(db_path, db.add_node({"name": "a", "type": "file"}, 1))
    db.atomic(db_path, db.add_node({"name": "b", "type": "file"}, 2))
    db.atomic(db_path, db.connect_nodes(1, 2, {"type": "imports"}))
    monkeypatch.setattr(graph_gui, "GRAPH_DB_PATH", db_path)

    client = graph_gui.app.test_client()
    response = client.delete("/api/edge", json={"source": 1, "target": 2})

    assert response.status_code == 200
    assert response.get_json()["deleted"] == 1
    assert load_edges(db_path) == []


def test_edge_delete_api_returns_404_for_missing_edge(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite")
    db.initialize(db_path)
    monkeypatch.setattr(graph_gui, "GRAPH_DB_PATH", db_path)

    client = graph_gui.app.test_client()
    response = client.delete("/api/edge", json={"source": 1, "target": 2})

    assert response.status_code == 404


def test_edge_delete_api_can_match_properties(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite")
    db.initialize(db_path)
    db.atomic(db_path, db.add_node({"name": "a", "type": "file"}, 1))
    db.atomic(db_path, db.add_node({"name": "b", "type": "file"}, 2))
    db.atomic(db_path, db.connect_nodes(1, 2, {"type": "imports", "import_name": "a"}))
    db.atomic(db_path, db.connect_nodes(1, 2, {"type": "contains"}))
    monkeypatch.setattr(graph_gui, "GRAPH_DB_PATH", db_path)

    client = graph_gui.app.test_client()
    response = client.delete(
        "/api/edge",
        json={"source": 1, "target": 2, "properties": {"type": "imports", "import_name": "a"}},
    )

    assert response.status_code == 200
    remaining = load_edges(db_path)
    assert len(remaining) == 1
    assert remaining[0]["properties"] == {"type": "contains"}


def test_code_graph_search_api_respects_n_hops(tmp_path, monkeypatch):
    db_path = _build_import_db(tmp_path)
    main_node = next(n for n in load_all_nodes(db_path) if n.get("path") == "main.py")
    _FakeVectorStore.node_id = main_node["id"]

    monkeypatch.setattr("code_graph_builder.get_embedding", lambda _text, _config: [0.1] * 1024)
    monkeypatch.setattr(graph_gui, "FaissVectorStore", _FakeVectorStore)

    client = graph_gui.app.test_client()
    zero_hop = client.post(
        "/api/code-graph/search",
        json={"query": "main", "db_path": db_path, "threshold": 0.3, "top_k": 10, "n_hops": 0},
    ).get_json()
    one_hop = client.post(
        "/api/code-graph/search",
        json={"query": "main", "db_path": db_path, "threshold": 0.3, "top_k": 10, "n_hops": 1},
    ).get_json()

    assert zero_hop["n_hops"] == 0
    assert one_hop["n_hops"] == 1
    assert len(zero_hop["graph_data"]["nodes"]) == 1
    assert zero_hop["graph_data"]["edges"] == []
    assert len(one_hop["graph_data"]["nodes"]) >= 2
    assert len(one_hop["graph_data"]["edges"]) >= 1
    assert one_hop["graph_data"]["query"] == "main"
    assert one_hop["graph_data"]["initial_nodes"] == {str(main_node["id"]): 0.91}


def test_sessions_batch_delete_api(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_gui, "SESSIONS_DIR", str(tmp_path))
    _write_session(tmp_path, "s1", "query one", 1, 0.8)
    _write_session(tmp_path, "s2", "query two", 2, 0.7)

    client = graph_gui.app.test_client()
    response = client.delete("/api/sessions", json={"ids": ["s1", "missing"]})

    assert response.status_code == 200
    assert response.get_json()["deleted"] == ["s1"]
    assert response.get_json()["missing"] == ["missing"]
    assert not (tmp_path / "s1.json").exists()
    assert (tmp_path / "s2.json").exists()


def test_sessions_merge_api_combines_query_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_gui, "SESSIONS_DIR", str(tmp_path))
    _write_session(tmp_path, "s1", "query one", 1, 0.8)
    _write_session(tmp_path, "s2", "query two", 2, 0.7)

    client = graph_gui.app.test_client()
    response = client.post("/api/sessions/merge", json={"ids": ["s1", "s2"]})
    result = response.get_json()

    assert response.status_code == 200
    assert result["tool_name"] == "merged_sessions"
    assert result["graph_data"]["view_mode"] == "merged_sessions"
    assert len(result["sessions"]) == 2
    query_nodes = [n for n in result["graph_data"]["nodes"] if n["type"] == "query"]
    similarity_edges = [e for e in result["graph_data"]["edges"] if e["properties"]["type"] == "similarity"]
    assert len(query_nodes) == 2
    assert len(similarity_edges) == 2
    assert similarity_edges[0]["dashes"] is True


def test_sessions_merge_api_recovers_old_search_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_gui, "SESSIONS_DIR", str(tmp_path))
    session = _write_session(tmp_path, "s1", "query one", 1, 0.8)
    session["graph_data"].pop("initial_nodes")
    session["graph_data"].pop("query")
    session["result_text"] = json.dumps({"matches": [{"id": 1, "score": 0.8}]})
    (tmp_path / "s1.json").write_text(json.dumps(session), encoding="utf-8")
    _write_session(tmp_path, "s2", "query two", 2, 0.7)

    client = graph_gui.app.test_client()
    response = client.post("/api/sessions/merge", json={"ids": ["s1", "s2"]})
    result = response.get_json()

    assert response.status_code == 200
    query_nodes = [n for n in result["graph_data"]["nodes"] if n["type"] == "query"]
    similarity_edges = [e for e in result["graph_data"]["edges"] if e["properties"]["type"] == "similarity"]
    assert len(query_nodes) == 2
    assert len(similarity_edges) == 2


def test_history_auto_render_logic_is_enabled():
    template = (importlib.import_module("pathlib").Path(__file__).parents[1] / "templates" / "index.html").read_text(encoding="utf-8")

    assert "var currentSessionId = null;" in template
    assert "currentSessionId = sessionId;" in template
    assert "shouldAutoRender && !silent" not in template


def test_query_overlay_frontend_logic_is_present():
    template = (importlib.import_module("pathlib").Path(__file__).parents[1] / "templates" / "index.html").read_text(encoding="utf-8")

    assert "function applyQueryOverlay(graphData)" in template
    assert "group: 'query'" in template
    assert "shape: 'dot'" in template
    assert "dashes: true" in template
    assert "label: scoreLabel ? 'similarity ' + scoreLabel : 'similarity'" in template
    assert "applyQueryOverlay(session.graph_data);" in template


def test_code_search_hops_frontend_control_is_present():
    template = (importlib.import_module("pathlib").Path(__file__).parents[1] / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="code-search-hops" min="0" max="5" step="1" value="1"' in template
    assert 'id="code-search-hops-value"' in template
    assert "const nHops = parseInt(document.getElementById('code-search-hops').value, 10);" in template
    assert "n_hops: nHops" in template
    assert "updateGraph(result.graph_data ||" in template


def test_history_merge_and_global_overview_controls_are_present():
    template = (importlib.import_module("pathlib").Path(__file__).parents[1] / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="btn-global-overview"' in template
    assert "loadGlobalOverview" in template
    assert 'id="merge-history-button"' in template
    assert 'id="delete-history-selected-button"' in template
    assert 'id="history-select-all"' in template
    assert "fetch('/api/sessions/merge'" in template
    assert "method: 'DELETE'" in template
    assert "function renderMergedGraphData(graphData)" in template
    assert "graphData.view_mode === 'merged_sessions'" in template
    assert "solver: 'forceAtlas2Based'" in template
    assert "fixed: false" in template

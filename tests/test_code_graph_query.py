"""
测试代码图谱查询引擎
验证 cypher_match_nodes、embedding_search 不因 schema 问题报错
"""

import os
import json
import pytest

from code_graph_builder import CodeGraphBuilder
from code_graph_query import CodeGraphQueryEngine
from code_graph_db import load_all_nodes, load_edges


def _write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _default_config():
    return {
        "llm_base_url": "http://localhost:1234/v1/",
        "llm_api_key": "not-needed",
        "llm_model": "test",
        "embedding_base_url": "http://localhost:1234/v1/",
        "embedding_api_key": "not-needed",
        "embedding_model": "test",
    }


def _build_small_db(tmp_path):
    """构建一个小型测试 DB"""
    db_path = str(tmp_path / "test.sqlite")
    config = _default_config()

    _write_file(str(tmp_path / "pkg" / "__init__.py"), "")
    _write_file(str(tmp_path / "pkg" / "a.py"), "def foo(): pass\n")
    _write_file(str(tmp_path / "main.py"), "import pkg.a\n")

    builder = CodeGraphBuilder(str(tmp_path), db_path, config,
                               rebuild=True, generate_embeddings=False, generate_comments=False)
    builder.build()
    return db_path


class TestCypherMatchNodes:
    def test_cypher_match_nodes_no_schema_error(self, tmp_path):
        """cypher_match_nodes 不报 OperationalError: no such column: properties"""
        db_path = _build_small_db(tmp_path)
        engine = CodeGraphQueryEngine(db_path)

        nodes = engine.cypher_match_nodes(limit=10)
        assert len(nodes) > 0
        # 确保返回的节点有 name、type、id 等字段
        for n in nodes:
            assert "id" in n
            assert "name" in n
            assert "type" in n

    def test_cypher_match_nodes_with_label(self, tmp_path):
        """按 label 过滤节点"""
        db_path = _build_small_db(tmp_path)
        engine = CodeGraphQueryEngine(db_path)

        file_nodes = engine.cypher_match_nodes(label="file")
        assert all(n["type"] == "file" for n in file_nodes)

        folder_nodes = engine.cypher_match_nodes(label="folder")
        assert all(n["type"] == "folder" for n in folder_nodes)


class TestCypherMatchEdges:
    def test_cypher_match_edges(self, tmp_path):
        """cypher_match_edges 正常工作"""
        db_path = _build_small_db(tmp_path)
        engine = CodeGraphQueryEngine(db_path)

        edges = engine.cypher_match_edges()
        assert len(edges) > 0

    def test_cypher_match_edges_by_type(self, tmp_path):
        """按类型过滤边"""
        db_path = _build_small_db(tmp_path)
        engine = CodeGraphQueryEngine(db_path)

        import_edges = engine.cypher_match_edges(edge_type="imports")
        assert all(e["type"] == "imports" for e in import_edges)


class TestCypherMatchPath:
    def test_cypher_match_path(self, tmp_path):
        """cypher_match_path 不报 schema 错误"""
        db_path = _build_small_db(tmp_path)
        engine = CodeGraphQueryEngine(db_path)

        result = engine.cypher_match_path("main.py", hops=1)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0


class TestEmbeddingSearchGraceful:
    def test_embedding_search_graceful_when_no_faiss(self, tmp_path):
        """FAISS 为空时 embedding_search 不因 schema 报错（会因无法连接 embedding 服务返回空列表）"""
        db_path = _build_small_db(tmp_path)
        engine = CodeGraphQueryEngine(db_path)

        # 没有 embedding 服务，get_embedding 会失败，应返回空列表
        results = engine.embedding_search("test", top_k=5)
        assert isinstance(results, list)

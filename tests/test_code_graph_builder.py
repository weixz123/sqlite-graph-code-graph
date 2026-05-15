"""
测试代码图谱构建器
验证 rebuild、incremental build、import 解析的正确性
"""

import os
import json
import shutil
import tempfile
import pytest

from code_graph_builder import (
    CodeGraphBuilder, scan_python_files, parse_imports,
    build_module_index, resolve_import, get_folders,
)
from code_graph_db import load_all_nodes, load_edges


@pytest.fixture
def tmp_project(tmp_path):
    """创建临时项目目录"""
    return tmp_path


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


class TestRebuild:
    def test_rebuild_same_db_twice_does_not_fail(self, tmp_project):
        """rebuild=True 两次构建同一 DB 不应失败"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        # 第一次：a.py
        _write_file(str(tmp_project / "a.py"), "x = 1\n")
        builder = CodeGraphBuilder(str(tmp_project), db_path, config,
                                   rebuild=True, generate_embeddings=False, generate_comments=False)
        stats1 = builder.build()
        assert stats1["files"] == 1

        # 第二次：b.py（a.py 已不存在于文件系统但仍存在于旧DB）
        os.remove(str(tmp_project / "a.py"))
        _write_file(str(tmp_project / "b.py"), "y = 2\n")
        builder2 = CodeGraphBuilder(str(tmp_project), db_path, config,
                                    rebuild=True, generate_embeddings=False, generate_comments=False)
        stats2 = builder2.build()
        assert stats2["files"] == 1

        # 验证 DB 只有 b.py
        nodes = load_all_nodes(db_path)
        file_nodes = [n for n in nodes if n.get("type") == "file"]
        assert len(file_nodes) == 1
        assert file_nodes[0]["name"] == "b.py"


class TestIncremental:
    def test_incremental_without_rebuild_uses_next_id(self, tmp_project):
        """rebuild=False 两次构建不应出现 id 冲突"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        # 第一次
        _write_file(str(tmp_project / "a.py"), "x = 1\n")
        builder1 = CodeGraphBuilder(str(tmp_project), db_path, config,
                                    rebuild=True, generate_embeddings=False, generate_comments=False)
        stats1 = builder1.build()
        assert stats1["files"] == 1

        # 第二次（incremental）— 增加文件
        _write_file(str(tmp_project / "b.py"), "y = 2\n")
        builder2 = CodeGraphBuilder(str(tmp_project), db_path, config,
                                    rebuild=False, generate_embeddings=False, generate_comments=False)
        stats2 = builder2.build()
        # 不应报错
        assert stats2["files"] == 2  # a.py + b.py

        # 所有 node id 唯一
        nodes = load_all_nodes(db_path)
        ids = [n["id"] for n in nodes]
        assert len(ids) == len(set(ids))


class TestImportParsing:
    def test_parse_imports_structure(self, tmp_project):
        """parse_imports 返回结构化信息"""
        fp = str(tmp_project / "main.py")
        _write_file(fp, "import os\nfrom sys import path\nfrom . import utils\nfrom .foo import bar\n")
        result = parse_imports(fp)

        assert len(result) == 4
        # import os
        assert result[0] == {"module": "os", "level": 0, "names": [], "kind": "import"}
        # from sys import path
        assert result[1] == {"module": "sys", "level": 0, "names": ["path"], "kind": "from"}
        # from . import utils
        assert result[2] == {"module": "", "level": 1, "names": ["utils"], "kind": "from"}
        # from .foo import bar
        assert result[3] == {"module": "foo", "level": 1, "names": ["bar"], "kind": "from"}


class TestModuleIndex:
    def test_build_module_index(self, tmp_project):
        """模块索引正确映射"""
        _write_file(str(tmp_project / "pkg" / "__init__.py"), "")
        _write_file(str(tmp_project / "pkg" / "a.py"), "x = 1")
        _write_file(str(tmp_project / "main.py"), "")

        files = scan_python_files(str(tmp_project))
        index = build_module_index(files, str(tmp_project))

        assert "pkg" in index
        assert "pkg.a" in index
        assert "main" in index
        assert index["pkg"].replace("\\", "/").endswith("pkg/__init__.py")
        assert index["pkg.a"].replace("\\", "/").endswith("pkg/a.py")


class TestImportResolution:
    def test_import_does_not_fanout_package(self, tmp_project):
        """import pkg 只指向 __init__.py，不 fan-out"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        _write_file(str(tmp_project / "pkg" / "__init__.py"), "")
        _write_file(str(tmp_project / "pkg" / "a.py"), "x = 1")
        _write_file(str(tmp_project / "pkg" / "b.py"), "y = 2")
        _write_file(str(tmp_project / "main.py"), "import pkg\n")

        builder = CodeGraphBuilder(str(tmp_project), db_path, config,
                                   rebuild=True, generate_embeddings=False, generate_comments=False)
        stats = builder.build()

        edges = load_edges(db_path)
        import_edges = [e for e in edges if e["properties"].get("type") == "imports"]

        # import_edges 应该只有 main -> pkg/__init__.py
        assert len(import_edges) == 1
        target_id = import_edges[0]["target"]
        nodes = load_all_nodes(db_path)
        target_node = next(n for n in nodes if n["id"] == target_id)
        assert target_node["name"] == "__init__.py" or target_node["path"] == "pkg"

    def test_import_specific_module_resolves_exact_file(self, tmp_project):
        """import pkg.a 精确解析到 pkg/a.py"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        _write_file(str(tmp_project / "pkg" / "__init__.py"), "")
        _write_file(str(tmp_project / "pkg" / "a.py"), "x = 1")
        _write_file(str(tmp_project / "pkg" / "b.py"), "y = 2")
        _write_file(str(tmp_project / "main.py"), "import pkg.a\n")

        builder = CodeGraphBuilder(str(tmp_project), db_path, config,
                                   rebuild=True, generate_embeddings=False, generate_comments=False)
        stats = builder.build()

        edges = load_edges(db_path)
        import_edges = [e for e in edges if e["properties"].get("type") == "imports"]

        assert len(import_edges) == 1
        target_id = import_edges[0]["target"]
        nodes = load_all_nodes(db_path)
        target_node = next(n for n in nodes if n["id"] == target_id)
        assert target_node["name"] == "a.py"

    def test_from_import_resolves_correctly(self, tmp_project):
        """from pkg import a 解析到 pkg/a.py"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        _write_file(str(tmp_project / "pkg" / "__init__.py"), "")
        _write_file(str(tmp_project / "pkg" / "a.py"), "x = 1")
        _write_file(str(tmp_project / "main.py"), "from pkg import a\n")

        builder = CodeGraphBuilder(str(tmp_project), db_path, config,
                                   rebuild=True, generate_embeddings=False, generate_comments=False)
        stats = builder.build()

        edges = load_edges(db_path)
        import_edges = [e for e in edges if e["properties"].get("type") == "imports"]

        assert len(import_edges) == 1
        target_id = import_edges[0]["target"]
        nodes = load_all_nodes(db_path)
        target_node = next(n for n in nodes if n["id"] == target_id)
        assert target_node["name"] == "a.py"

    def test_import_no_init_no_edge(self, tmp_project):
        """import pkg 但无 __init__.py 时不创建 imports 边"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        _write_file(str(tmp_project / "pkg" / "a.py"), "x = 1")
        _write_file(str(tmp_project / "main.py"), "import pkg\n")

        builder = CodeGraphBuilder(str(tmp_project), db_path, config,
                                   rebuild=True, generate_embeddings=False, generate_comments=False)
        stats = builder.build()

        edges = load_edges(db_path)
        import_edges = [e for e in edges if e["properties"].get("type") == "imports"]

        # pkg 没有 __init__.py，import pkg 不应解析到 pkg/a.py
        assert len(import_edges) == 0

    def test_import_external_library_no_edge(self, tmp_project):
        """import os 等外部库不应创建边"""
        db_path = str(tmp_project / "test.sqlite")
        config = _default_config()

        _write_file(str(tmp_project / "main.py"), "import os\nimport json\n")

        builder = CodeGraphBuilder(str(tmp_project), db_path, config,
                                   rebuild=True, generate_embeddings=False, generate_comments=False)
        stats = builder.build()

        edges = load_edges(db_path)
        import_edges = [e for e in edges if e["properties"].get("type") == "imports"]
        assert len(import_edges) == 0

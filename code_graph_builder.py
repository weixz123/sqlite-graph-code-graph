#!/usr/bin/env python3
"""
Python代码图谱构建器
扫描指定路径下的Python文件，构建包含文件、文件夹、引用关系的代码图谱。
- 节点类型: file(文件), folder(文件夹)
- 关系类型: imports(引用), contains(包含)
- 每个文件节点附带LLM生成的功能注释和embedding向量
- metadata存储在SQLite，向量存储在FAISS（完全解耦）
- 支持 rebuild 可重复构建、精确 import 解析
"""

import os
import ast
import json
import re
import logging
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI

from simple_graph_sqlite import database as db
from faiss_store import FaissVectorStore
from code_graph_db import next_numeric_id, clear_graph_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("code_graph_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIG = {
    "llm_base_url": "http://localhost:1234/v1/",
    "llm_api_key": "not-needed",
    "llm_model": "qwen3_30ba3b",
    "embedding_base_url": "http://localhost:1234/v1/",
    "embedding_api_key": "not-needed",
    "embedding_model": "text-embedding-bge-m3",
}

CONFIG_FILE = "llm_config.json"


def load_config() -> dict:
    """加载持久化的LLM配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULT_CONFIG, **saved}
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """持久化保存LLM配置"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"配置已保存到 {CONFIG_FILE}")


def _get_llm_client(config: dict) -> OpenAI:
    return OpenAI(base_url=config["llm_base_url"], api_key=config["llm_api_key"])


def _get_embedding_client(config: dict) -> OpenAI:
    return OpenAI(base_url=config["embedding_base_url"], api_key=config["embedding_api_key"])


def generate_comment(source_code: str, file_path: str, config: dict) -> str:
    """
    调用LLM为源代码生成功能注释（约300字，只写功能描述）
    """
    client = _get_llm_client(config)
    prompt = f"""请为以下Python源代码生成一段约300字以内的功能描述注释。
要求：
1. 只描述代码的功能和作用，不要包含使用方法或安装说明
2. 提及主要的类、函数和它们的作用
3. 语言简洁，重点突出
4. 使用中文

文件路径: {file_path}

源代码:
{source_code[:8000]}
"""
    try:
        response = client.chat.completions.create(
            model=config["llm_model"],
            messages=[
                {"role": "system", "content": "你是一个专业的Python代码分析助手，擅长用简洁的中文描述代码功能。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024
        )
        text = response.choices[0].message.content or ""
        text = re.sub(r'<think[\s\S]*?</think\s*>', '', text).strip()
        text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).strip('`').strip(), text)
        return text.strip()
    except Exception as e:
        logger.error(f"生成注释失败 [{file_path}]: {e}")
        return ""


def get_embedding(text: str, config: dict) -> Optional[list]:
    """
    使用LM Studio的embedding模型生成文本向量
    """
    client = _get_embedding_client(config)
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=config["embedding_model"]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"生成embedding失败: {e}")
        return None


def scan_python_files(root_path: str) -> List[str]:
    """扫描目录下所有.py文件（排除 .venv 等虚拟环境目录）"""
    python_files = []
    exclude_dirs = {".venv", "venv", "__pycache__", ".git", "node_modules", ".idea", ".vscode"}
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            if f.endswith(".py"):
                python_files.append(os.path.join(dirpath, f))
    return sorted(python_files)


# ==================== 精确 import 解析 ====================

def parse_imports(file_path: str) -> List[Dict]:
    """
    解析Python文件的import语句，返回结构化的import信息列表。

    Returns:
        [{"module": "pkg.a", "level": 0, "names": [], "kind": "import"},
         {"module": "pkg", "level": 0, "names": ["a"], "kind": "from"},
         {"module": ".", "level": 1, "names": ["a"], "kind": "from"}, ...]
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "level": 0,
                    "names": [],
                    "kind": "import",
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            names = [alias.name for alias in node.names]
            imports.append({
                "module": module,
                "level": level,
                "names": names,
                "kind": "from",
            })
    return imports


def build_module_index(python_files: List[str], root_path: str) -> Dict[str, str]:
    """
    构建模块名 -> 文件路径的索引。

    例如：
      pkg/a.py -> "pkg.a" -> /path/to/pkg/a.py
      pkg/__init__.py -> "pkg" -> /path/to/pkg/__init__.py
    """
    root_norm = os.path.normpath(root_path)
    index = {}
    for fp in python_files:
        rel = os.path.relpath(fp, root_norm).replace("\\", "/")
        if rel.endswith("/__init__.py"):
            module = rel[:-12]  # 去掉 /__init__.py
        elif rel.endswith(".py"):
            module = rel[:-3]   # 去掉 .py
        else:
            continue
        module = module.replace("/", ".")
        index[module] = fp
    return index


def resolve_import(import_info: Dict, current_file: str,
                   module_index: Dict[str, str], root_path: str) -> Optional[str]:
    """
    将一条 import 信息解析为目标文件路径。

    规则：
      - import pkg.a -> pkg/a.py（精确）
      - import pkg -> pkg/__init__.py（仅 __init__.py，不 fan-out）
      - from pkg import a -> 先尝试 pkg.a，再尝试 pkg/__init__.py
      - from . import a -> 基于 current_file 所在包解析
      - from .a import b -> 基于 current_file 所在包解析 .a 再 .a.b

    解析不到返回 None（外部库，跳过）。
    """
    kind = import_info["kind"]
    module = import_info["module"]
    level = import_info["level"]
    names = import_info["names"]

    root_norm = os.path.normpath(root_path)

    if level > 0:
        # 相对导入：基于 current_file 所在目录
        current_dir = os.path.dirname(current_file)
        # 向上 level-1 层
        base_dir = current_dir
        for _ in range(level - 1):
            base_dir = os.path.dirname(base_dir)

        if module:
            # from .a import b
            base_module_parts = os.path.relpath(base_dir, root_norm).replace("\\", "/").split("/")
            full_module = ".".join(base_module_parts + module.split("."))
            if full_module.startswith("."):
                full_module = full_module.lstrip(".")
        else:
            # from . import a
            full_module = None

        # 尝试解析 names
        for name in names:
            candidate = f"{full_module}.{name}" if full_module else name
            if candidate in module_index:
                return module_index[candidate]
            # 也尝试在当前包下直接查找 name
            rel_dir = os.path.relpath(base_dir, root_norm).replace("\\", "/")
            if rel_dir == ".":
                candidate2 = name
            else:
                candidate2 = rel_dir.replace("/", ".") + "." + name
            if candidate2 in module_index:
                return module_index[candidate2]

        # 没有 names 时尝试 module 本身
        if full_module and full_module in module_index:
            return module_index[full_module]

        return None

    # 绝对导入
    if kind == "import":
        # import pkg.a -> 精确匹配 pkg.a
        if module in module_index:
            return module_index[module]
        # import pkg -> 只匹配 __init__.py（已由 module_index 保证）
        # 如果 pkg 不在 index 中（没有 __init__.py），不 fan-out
        return None

    if kind == "from":
        # from pkg import a
        for name in names:
            # 先尝试 pkg.a
            candidate = f"{module}.{name}" if module else name
            if candidate in module_index:
                return module_index[candidate]
        # 再尝试 module 本身（__init__.py）
        if module in module_index:
            return module_index[module]
        return None

    return None


def get_folders(file_paths: List[str], root_path: str) -> List[str]:
    """从文件列表中提取所有文件夹（去重）"""
    folders = set()
    root_path = os.path.normpath(root_path)
    for fp in file_paths:
        dir_path = os.path.dirname(fp)
        dir_path = os.path.normpath(dir_path)
        while dir_path.startswith(root_path) and dir_path != root_path:
            folders.add(dir_path)
            parent = os.path.dirname(dir_path)
            if parent == dir_path:
                break
            dir_path = parent
    return sorted(folders)


class CodeGraphBuilder:
    """Python代码图谱构建器"""

    def __init__(self, root_path: str, graph_db_path: str, config: dict,
                 rebuild: bool = False,
                 generate_embeddings: bool = True,
                 generate_comments: bool = True):
        self.root_path = os.path.normpath(root_path)
        self.graph_db_path = graph_db_path
        self.config = config
        self.generate_embeddings = generate_embeddings
        self.generate_comments = generate_comments

        if rebuild:
            clear_graph_files(graph_db_path)
            db.initialize(graph_db_path)
            logger.info(f"图数据库(rebuild)初始化: {graph_db_path}")
            self.next_id = 1
        else:
            if not os.path.exists(graph_db_path):
                db.initialize(graph_db_path)
                logger.info(f"图数据库(新建)初始化: {graph_db_path}")
            self.next_id = next_numeric_id(graph_db_path)

        # 初始化FAISS向量存储
        base_name = os.path.splitext(graph_db_path)[0]
        self.vector_store = FaissVectorStore(base_name)
        logger.info(f"FAISS向量存储初始化: {base_name}.*")

        # 缓存映射: path -> node_id
        self.path_to_id: Dict[str, int] = {}

    def _add_node(self, data: dict) -> int:
        """添加节点并返回ID"""
        node_id = self.next_id
        self.next_id += 1
        db.atomic(self.graph_db_path, db.add_node(data, node_id))
        return node_id

    def _add_edge(self, source_id: int, target_id: int, properties: dict):
        """添加边"""
        db.atomic(self.graph_db_path, db.connect_nodes(source_id, target_id, properties))

    def build(self, progress_callback=None) -> dict:
        """
        构建代码图谱

        Args:
            progress_callback: 可选的回调函数 callback(step, total, message)
        """
        stats = {"files": 0, "folders": 0, "import_edges": 0, "contain_edges": 0, "comments": 0}

        # 1. 扫描文件
        python_files = scan_python_files(self.root_path)
        total_steps = len(python_files) * (3 if self.generate_comments and self.generate_embeddings
                                            else 2 if self.generate_comments or self.generate_embeddings
                                            else 1)
        current_step = 0
        logger.info(f"扫描到 {len(python_files)} 个Python文件")

        if not python_files:
            return stats

        # 2. 构建模块索引
        module_index = build_module_index(python_files, self.root_path)

        # 3. 解析所有文件的import（结构化）
        file_imports = {}
        for fp in python_files:
            file_imports[fp] = parse_imports(fp)

        # 4. 提取文件夹
        folders = get_folders(python_files, self.root_path)

        # 5. 创建文件夹节点
        for folder in folders:
            rel = os.path.relpath(folder, self.root_path).replace("\\", "/")
            node_id = self._add_node({
                "name": os.path.basename(folder) or os.path.basename(self.root_path),
                "type": "folder",
                "path": rel,
            })
            self.path_to_id[folder] = node_id
            stats["folders"] += 1

        # 6. 创建文件节点
        for fp in python_files:
            rel = os.path.relpath(fp, self.root_path).replace("\\", "/")
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    source_code = f.read()
            except Exception:
                source_code = ""

            # LLM生成注释
            comment = ""
            if self.generate_comments and source_code.strip():
                comment = generate_comment(source_code, rel, self.config)
                if comment:
                    stats["comments"] += 1

            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, f"生成注释: {rel}")

            # 创建metadata节点
            node_id = self._add_node({
                "name": os.path.basename(fp),
                "type": "file",
                "path": rel,
                "comment": comment,
                "source_preview": source_code[:500] if source_code else "",
            })
            self.path_to_id[fp] = node_id

            # 生成embedding
            if self.generate_embeddings:
                meta = {"name": os.path.basename(fp), "path": rel, "type": "file"}

                if comment:
                    comment_emb = get_embedding(comment, self.config)
                    if comment_emb:
                        self.vector_store.add_comment_embedding(node_id, comment_emb, meta)

                if source_code.strip():
                    code_emb = get_embedding(source_code[:4000], self.config)
                    if code_emb:
                        self.vector_store.add_code_embedding(node_id, code_emb, meta)

                current_step += 2
                if progress_callback:
                    progress_callback(current_step, total_steps, f"完成: {rel}")

            stats["files"] += 1

        # 7. 创建contains关系
        for folder in folders:
            folder_id = self.path_to_id.get(folder)
            if folder_id is None:
                continue
            for fp in python_files:
                file_dir = os.path.dirname(fp)
                if os.path.normpath(file_dir) == folder:
                    file_id = self.path_to_id.get(fp)
                    if file_id:
                        self._add_edge(folder_id, file_id, {"type": "contains"})
                        stats["contain_edges"] += 1
            for sub in folders:
                if os.path.normpath(os.path.dirname(sub)) == folder and sub != folder:
                    sub_id = self.path_to_id.get(sub)
                    if sub_id:
                        self._add_edge(folder_id, sub_id, {"type": "contains"})
                        stats["contain_edges"] += 1

        # 8. 创建imports关系（精确解析）
        for fp, imports in file_imports.items():
            source_id = self.path_to_id.get(fp)
            if source_id is None:
                continue
            for imp_info in imports:
                target_fp = resolve_import(imp_info, fp, module_index, self.root_path)
                if target_fp is None or target_fp == fp:
                    continue
                target_id = self.path_to_id.get(target_fp)
                if target_id is None:
                    continue

                # 构建 import_name 用于边属性
                if imp_info["kind"] == "import":
                    import_name = imp_info["module"]
                else:
                    names_str = ", ".join(imp_info["names"])
                    if imp_info["module"]:
                        import_name = f"from {imp_info['module']} import {names_str}"
                    else:
                        dots = "." * imp_info["level"]
                        import_name = f"from {dots} import {names_str}"

                # 解析到的模块名
                resolved_rel = os.path.relpath(target_fp, self.root_path).replace("\\", "/")
                if resolved_rel.endswith("/__init__.py"):
                    resolved_module = resolved_rel[:-12].replace("/", ".")
                else:
                    resolved_module = resolved_rel[:-3].replace("/", ".")

                self._add_edge(source_id, target_id, {
                    "type": "imports",
                    "import_name": import_name,
                    "resolved_module": resolved_module,
                    "confidence": 1.0,
                })
                stats["import_edges"] += 1

        # 9. 保存FAISS索引到磁盘
        if self.generate_embeddings:
            self.vector_store.save()

        logger.info(f"图谱构建完成: {stats}")
        return stats


def main():
    """CLI入口"""
    import argparse
    parser = argparse.ArgumentParser(description="Python代码图谱构建器")
    parser.add_argument("path", help="要扫描的Python项目路径")
    parser.add_argument("--output", default="code_graph.sqlite", help="输出图数据库路径")
    parser.add_argument("--rebuild", action="store_true", help="清空旧数据重新构建")
    parser.add_argument("--no-embeddings", action="store_true", help="不生成embedding向量")
    parser.add_argument("--no-comments", action="store_true", help="不生成LLM注释")
    args = parser.parse_args()

    config = load_config()
    builder = CodeGraphBuilder(
        args.path, args.output, config,
        rebuild=args.rebuild,
        generate_embeddings=not args.no_embeddings,
        generate_comments=not args.no_comments,
    )

    def progress(step, total, msg):
        print(f"[{step}/{total}] {msg}")

    stats = builder.build(progress_callback=progress)
    print(f"\n构建完成: {json.dumps(stats, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()

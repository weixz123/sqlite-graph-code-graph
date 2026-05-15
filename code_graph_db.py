#!/usr/bin/env python3
"""
代码图谱统一数据库适配器
屏蔽 simple_graph_sqlite 的真实 schema（nodes 表只有 body TEXT 列，id 从 body JSON 生成）。
所有对 code_graph.sqlite 的读写都通过此模块，避免直接写 SELECT properties FROM nodes。
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# FAISS 相关文件后缀
FAISS_SUFFIXES = [
    ".comment.index",
    ".comment.ids.json",
    ".comment.meta.json",
    ".code.index",
    ".code.ids.json",
    ".code.meta.json",
]


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_body(body_str: str) -> dict:
    """解析 body JSON，保证返回的 dict 中 id 为 int"""
    d = json.loads(body_str)
    if "id" in d:
        try:
            d["id"] = int(d["id"])
        except (ValueError, TypeError):
            pass
    return d


def load_all_nodes(db_path: str) -> List[dict]:
    """
    加载全部节点。
    读取 SELECT id, body FROM nodes，解析 body JSON。
    """
    if not os.path.exists(db_path):
        return []
    conn = _connect(db_path)
    try:
        cursor = conn.execute("SELECT id, body FROM nodes")
        nodes = []
        for row in cursor:
            body = row["body"]
            if not body:
                continue
            d = _parse_body(body)
            # 确保 id 一致：优先用 body 中的 id
            if "id" not in d:
                d["id"] = int(row["id"]) if row["id"] is not None else 0
            nodes.append(d)
        return nodes
    finally:
        conn.close()


def load_node(db_path: str, node_id) -> Optional[dict]:
    """加载单个节点"""
    if not os.path.exists(db_path):
        return None
    conn = _connect(db_path)
    try:
        cursor = conn.execute("SELECT body FROM nodes WHERE id = ?", (str(node_id),))
        row = cursor.fetchone()
        if row and row["body"]:
            d = _parse_body(row["body"])
            if "id" not in d:
                d["id"] = int(node_id)
            return d
        return None
    finally:
        conn.close()


def load_edges(db_path: str) -> List[dict]:
    """
    加载全部边。
    source/target 从 TEXT 转为 int 以匹配 node id。
    """
    if not os.path.exists(db_path):
        return []
    conn = _connect(db_path)
    try:
        cursor = conn.execute("SELECT source, target, properties FROM edges")
        edges = []
        for row in cursor:
            props = json.loads(row["properties"]) if row["properties"] else {}
            edges.append({
                "source": int(row["source"]),
                "target": int(row["target"]),
                "properties": props,
            })
        return edges
    finally:
        conn.close()


def delete_edge(db_path: str, source, target, properties: Optional[dict] = None) -> int:
    """
    Delete edges matching source and target.

    If properties is provided, only edges whose parsed JSON properties exactly
    match that dict are deleted. Returns the number of deleted rows.
    """
    if not os.path.exists(db_path):
        return 0

    conn = _connect(db_path)
    try:
        if properties is None:
            cursor = conn.execute(
                "DELETE FROM edges WHERE source = ? AND target = ?",
                (str(source), str(target)),
            )
            deleted = cursor.rowcount
        else:
            rows = conn.execute(
                "SELECT rowid, properties FROM edges WHERE source = ? AND target = ?",
                (str(source), str(target)),
            ).fetchall()
            rowids = []
            for row in rows:
                edge_props = json.loads(row["properties"]) if row["properties"] else {}
                if edge_props == properties:
                    rowids.append(row["rowid"])

            deleted = 0
            for rowid in rowids:
                cursor = conn.execute("DELETE FROM edges WHERE rowid = ?", (rowid,))
                deleted += cursor.rowcount

        conn.commit()
        return deleted
    finally:
        conn.close()


def next_numeric_id(db_path: str) -> int:
    """
    从现有 nodes 中找最大数字 id，返回 max+1。
    如果 DB 不存在或为空，返回 1。
    """
    if not os.path.exists(db_path):
        return 1
    conn = _connect(db_path)
    try:
        cursor = conn.execute("SELECT body FROM nodes")
        max_id = 0
        for row in cursor:
            if row["body"]:
                d = json.loads(row["body"])
                nid = d.get("id", 0)
                try:
                    nid = int(nid)
                except (ValueError, TypeError):
                    continue
                if nid > max_id:
                    max_id = nid
        return max_id + 1
    finally:
        conn.close()


def clear_graph_files(db_path: str):
    """
    删除 SQLite 文件和同 base name 的 FAISS 文件。
    """
    # 删除 SQLite
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"已删除数据库: {db_path}")

    # 删除 FAISS 文件
    base_name = os.path.splitext(db_path)[0]
    for suffix in FAISS_SUFFIXES:
        fpath = base_name + suffix
        if os.path.exists(fpath):
            os.remove(fpath)
            logger.info(f"已删除FAISS文件: {fpath}")

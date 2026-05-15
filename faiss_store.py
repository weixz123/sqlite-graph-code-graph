#!/usr/bin/env python3
"""
FAISS向量存储模块
将embedding向量存储与SQLite的metadata存储完全解耦。

存储方案：
  - 向量索引: <base_name>.index  (FAISS IndexFlatIP)
  - 元数据:   <base_name>.meta.json  (node_id -> metadata映射)
  - ID映射:   <base_name>.ids.json   (FAISS内部序号 -> node_id映射)
"""

import os
import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """基于FAISS的向量存储，与SQLite元数据解耦"""

    def __init__(self, base_path: str, dim: int = 1024):
        """
        Args:
            base_path: 基础路径，不含扩展名。例如 "code_graph" 会生成:
                       code_graph.comment.index, code_graph.code.index
                       code_graph.comment.meta.json, code_graph.code.meta.json
                       code_graph.comment.ids.json, code_graph.code.ids.json
            dim: 向量维度 (bge-m3 默认 1024)
        """
        self.base_path = base_path
        self.dim = dim

        # 两个独立的索引：comment embedding 和 code embedding
        self._comment_index: Optional[faiss.IndexFlatIP] = None
        self._code_index: Optional[faiss.IndexFlatIP] = None

        # FAISS序号 -> node_id 的映射
        self._comment_ids: List[int] = []
        self._code_ids: List[int] = []

        # node_id -> metadata 映射
        self._comment_meta: Dict[int, dict] = {}
        self._code_meta: Dict[int, dict] = {}

        # 尝试加载已有索引
        self.load()

    # ==================== 文件路径 ====================

    @property
    def comment_index_path(self):
        return f"{self.base_path}.comment.index"

    @property
    def code_index_path(self):
        return f"{self.base_path}.code.index"

    @property
    def comment_ids_path(self):
        return f"{self.base_path}.comment.ids.json"

    @property
    def code_ids_path(self):
        return f"{self.base_path}.code.ids.json"

    @property
    def comment_meta_path(self):
        return f"{self.base_path}.comment.meta.json"

    @property
    def code_meta_path(self):
        return f"{self.base_path}.code.meta.json"

    # ==================== 加载/保存 ====================

    def load(self):
        """从磁盘加载已有的索引和元数据"""
        if os.path.exists(self.comment_index_path):
            self._comment_index = faiss.read_index(self.comment_index_path)
            with open(self.comment_ids_path, "r", encoding="utf-8") as f:
                self._comment_ids = json.load(f)
            with open(self.comment_meta_path, "r", encoding="utf-8") as f:
                self._comment_meta = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"已加载comment索引: {self._comment_index.ntotal} 条向量")
        else:
            self._comment_index = faiss.IndexFlatIP(self.dim)
            self._comment_ids = []
            self._comment_meta = {}

        if os.path.exists(self.code_index_path):
            self._code_index = faiss.read_index(self.code_index_path)
            with open(self.code_ids_path, "r", encoding="utf-8") as f:
                self._code_ids = json.load(f)
            with open(self.code_meta_path, "r", encoding="utf-8") as f:
                self._code_meta = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"已加载code索引: {self._code_index.ntotal} 条向量")
        else:
            self._code_index = faiss.IndexFlatIP(self.dim)
            self._code_ids = []
            self._code_meta = {}

    def save(self):
        """将索引和元数据保存到磁盘"""
        os.makedirs(os.path.dirname(self.base_path) or ".", exist_ok=True)

        if self._comment_index and self._comment_index.ntotal > 0:
            faiss.write_index(self._comment_index, self.comment_index_path)
            with open(self.comment_ids_path, "w", encoding="utf-8") as f:
                json.dump(self._comment_ids, f)
            with open(self.comment_meta_path, "w", encoding="utf-8") as f:
                json.dump(self._comment_meta, f, ensure_ascii=False, indent=2)

        if self._code_index and self._code_index.ntotal > 0:
            faiss.write_index(self._code_index, self.code_index_path)
            with open(self.code_ids_path, "w", encoding="utf-8") as f:
                json.dump(self._code_ids, f)
            with open(self.code_meta_path, "w", encoding="utf-8") as f:
                json.dump(self._code_meta, f, ensure_ascii=False, indent=2)

        logger.info("FAISS索引和元数据已保存")

    # ==================== 添加向量 ====================

    def add_comment_embedding(self, node_id: int, embedding: list, metadata: dict):
        """添加一条comment embedding"""
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self._comment_index.add(vec)
        self._comment_ids.append(node_id)
        self._comment_meta[node_id] = metadata

    def add_code_embedding(self, node_id: int, embedding: list, metadata: dict):
        """添加一条code embedding"""
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self._code_index.add(vec)
        self._code_ids.append(node_id)
        self._code_meta[node_id] = metadata

    def add_node_embedding(self, node_id: int, embedding: list, metadata: dict):
        """添加通用节点embedding（用于航空图谱等单embedding场景）"""
        # 复用comment索引存放
        self.add_comment_embedding(node_id, embedding, metadata)

    # ==================== 搜索 ====================

    def search(self, query_embedding: list, top_k: int = 10,
               threshold: float = 0.0, search_type: str = "comment") -> List[Dict]:
        """
        语义搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回最相似的K个结果
            threshold: 相似度阈值（余弦相似度，因已normalize等于内积）
            search_type: "comment" | "code" | "both"

        Returns:
            [{"node_id": int, "score": float, "metadata": dict}, ...]
        """
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        results = {}  # node_id -> best_score, metadata

        if search_type in ("comment", "both") and self._comment_index.ntotal > 0:
            scores, indices = self._comment_index.search(query_vec, min(top_k, self._comment_index.ntotal))
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or float(score) < threshold:
                    continue
                node_id = self._comment_ids[idx]
                if node_id not in results or float(score) > results[node_id]["score"]:
                    results[node_id] = {"score": float(score), "metadata": self._comment_meta.get(node_id, {})}

        if search_type in ("code", "both") and self._code_index.ntotal > 0:
            scores, indices = self._code_index.search(query_vec, min(top_k, self._code_index.ntotal))
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or float(score) < threshold:
                    continue
                node_id = self._code_ids[idx]
                if node_id not in results or float(score) > results[node_id]["score"]:
                    results[node_id] = {"score": float(score), "metadata": self._code_meta.get(node_id, {})}

        # 按score排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        return [{"node_id": nid, **info} for nid, info in sorted_results[:top_k]]

    def search_by_node_embedding(self, query_embedding: list, top_k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """搜索通用节点embedding（航空图谱场景）"""
        return self.search(query_embedding, top_k, threshold, search_type="comment")

    # ==================== 信息 ====================

    @property
    def comment_count(self) -> int:
        return self._comment_index.ntotal if self._comment_index else 0

    @property
    def code_count(self) -> int:
        return self._code_index.ntotal if self._code_index else 0

    def get_all_node_ids(self) -> set:
        """获取所有已索引的node_id"""
        return set(self._comment_ids) | set(self._code_ids)

    def has_node(self, node_id: int) -> bool:
        """检查某个node是否已有embedding"""
        return node_id in self._comment_meta or node_id in self._code_meta

    def clear(self):
        """清空所有索引（仅内存）"""
        self._comment_index = faiss.IndexFlatIP(self.dim)
        self._code_index = faiss.IndexFlatIP(self.dim)
        self._comment_ids = []
        self._code_ids = []
        self._comment_meta = {}
        self._code_meta = {}

    def delete_files(self):
        """删除磁盘上的索引和元数据文件"""
        for path in [
            self.comment_index_path, self.comment_ids_path, self.comment_meta_path,
            self.code_index_path, self.code_ids_path, self.code_meta_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"已删除: {path}")

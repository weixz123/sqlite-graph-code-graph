#!/usr/bin/env python3
"""
图数据库查询与嵌入生成工具集
metadata在SQLite，向量在FAISS（完全解耦）
"""

import sqlite3
import logging
import json
import re
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional

from simple_graph_sqlite import database as db
from faiss_store import FaissVectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_query.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== LMStudio & Embedding Functions ====================

client = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key="not-needed"
)


def call_lmstudio_chat(prompt="你好,请介绍一下自己"):
    """
    使用OpenAI聊天补全API格式调用LMStudio的语言模型
    """
    try:
        response = client.chat.completions.create(
            model="qwen3_30ba3b",
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=32768
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"调用LMStudio聊天API时出错: {e}")
        return None


def get_embedding(text, model="bge-m3:latest"):
    """
    使用LMStudio的嵌入API生成文本的向量表示
    """
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"生成嵌入向量时出错: {e}")
        return None


# ==================== Graph Query System ====================

class GraphQuerySystem:
    """图查询系统（metadata在SQLite, 向量在FAISS）"""

    def __init__(self, graph_db_path: str):
        self.graph_db_path = graph_db_path

        # FAISS向量存储（base_path与SQLite同目录同名）
        import os
        base_name = os.path.splitext(graph_db_path)[0]
        self.vector_store = FaissVectorStore(base_name)

        self._all_nodes_cache = None

    def load_or_generate_embeddings(self, force_regenerate: bool = False) -> List[Dict]:
        """
        加载FAISS中已有的向量，仅为缺少向量的节点生成新embedding

        参数:
            force_regenerate: 是否强制重新生成所有embedding
        """
        if self._all_nodes_cache is not None and not force_regenerate:
            return self._all_nodes_cache

        all_nodes = db.atomic(self.graph_db_path, db.find_nodes([], ()))
        if not all_nodes:
            self._all_nodes_cache = []
            return []

        if force_regenerate:
            self.vector_store.clear()

        cached_ids = self.vector_store.get_all_node_ids()
        nodes_with_embeddings = []
        need_generate = []

        for node in all_nodes:
            if node['id'] in cached_ids and not force_regenerate:
                # 已有FAISS缓存
                nodes_with_embeddings.append({
                    'id': node['id'],
                    'name': node.get('name', 'Unknown'),
                    'type': node.get('type', 'unknown'),
                    'description': node.get('description', ''),
                })
            else:
                need_generate.append(node)

        if need_generate:
            logger.info(f"正在为 {len(need_generate)} 个节点生成嵌入向量（已有 {len(nodes_with_embeddings)} 个缓存）...")
            for node in need_generate:
                text_for_embedding = f"{node.get('name', '')} {node.get('description', '')}"
                embedding = get_embedding(text_for_embedding)

                if embedding:
                    meta = {
                        'name': node.get('name', 'Unknown'),
                        'type': node.get('type', 'unknown'),
                        'description': node.get('description', ''),
                    }
                    self.vector_store.add_node_embedding(node['id'], embedding, meta)

                    nodes_with_embeddings.append({
                        'id': node['id'],
                        'name': node.get('name', 'Unknown'),
                        'type': node.get('type', 'unknown'),
                        'description': node.get('description', ''),
                    })

            # 持久化保存到FAISS
            self.vector_store.save()
            logger.info(f"embedding生成完成，总计 {len(nodes_with_embeddings)} 个节点。")
        else:
            logger.info(f"已从FAISS缓存加载 {len(nodes_with_embeddings)} 个节点的嵌入向量。")

        self._all_nodes_cache = nodes_with_embeddings
        return nodes_with_embeddings

    def get_all_nodes_with_embeddings(self, force_reload: bool = False) -> List[Dict]:
        """获取所有节点及其嵌入向量 (带缓存)"""
        if self._all_nodes_cache is not None and not force_reload:
            return self._all_nodes_cache
        return self.load_or_generate_embeddings(force_regenerate=force_reload)

    def find_similar_nodes(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """使用FAISS查找与查询最相似的节点"""
        logger.info(f"使用FAISS查找与 '{query}' 相似的节点 (阈值: {threshold})...")

        query_embedding = get_embedding(query)
        if not query_embedding:
            logger.error("无法生成查询嵌入向量")
            return []

        # FAISS搜索
        faiss_results = self.vector_store.search_by_node_embedding(query_embedding, top_k=top_k, threshold=threshold)

        # 补充metadata
        results = []
        for r in faiss_results:
            meta = r.get("metadata", {})
            results.append({
                'id': r["node_id"],
                'name': meta.get('name', 'Unknown'),
                'type': meta.get('type', 'unknown'),
                'description': meta.get('description', ''),
                'score': r["score"]
            })

        logger.info(f"找到 {len(results)} 个相似节点 (阈值以上)")
        return results

    def get_neighbors(self, node_id: int) -> List[Dict]:
        """获取节点的邻居及连接关系"""
        conn = sqlite3.connect(self.graph_db_path)
        cursor = conn.cursor()

        neighbors = []
        try:
            cursor.execute("SELECT target, properties FROM edges WHERE source = ?", (node_id,))
            for row in cursor.fetchall():
                target_id, properties = row
                neighbors.append({
                    'source': node_id,
                    'target': target_id,
                    'edge': json.loads(properties) if properties else {}
                })

            cursor.execute("SELECT source, properties FROM edges WHERE target = ?", (node_id,))
            for row in cursor.fetchall():
                source_id, properties = row
                neighbors.append({
                    'source': source_id,
                    'target': node_id,
                    'edge': json.loads(properties) if properties else {}
                })
        finally:
            conn.close()
        return neighbors

    def n_hop_query(self, start_node_id: int, n_hops: int = 1) -> Dict:
        """从起始节点进行N跳查询"""
        logger.info(f"从节点 {start_node_id} 开始进行 {n_hops}-跳查询...")

        if n_hops <= 0:
            return {'nodes': [], 'edges': []}

        all_nodes = {start_node_id}
        all_edges = []
        current_frontier = {start_node_id}

        for _ in range(n_hops):
            next_frontier = set()
            for node_id in current_frontier:
                neighbors = self.get_neighbors(node_id)
                for neighbor_info in neighbors:
                    edge = (neighbor_info['source'], neighbor_info['target'])
                    source_id, target_id = int(edge[0]), int(edge[1])
                    if source_id > target_id:
                        edge = (target_id, source_id)
                    else:
                        edge = (source_id, target_id)

                    if edge not in all_edges:
                        all_edges.append(edge)

                    neighbor_id = neighbor_info['target'] if neighbor_info['source'] == node_id else neighbor_info['source']
                    if neighbor_id not in all_nodes:
                        next_frontier.add(neighbor_id)
                        all_nodes.add(neighbor_id)

            current_frontier = next_frontier
            if not current_frontier:
                break

        nodes_data = []
        for node_id in all_nodes:
            node = db.atomic(self.graph_db_path, db.find_node(node_id))
            if node:
                nodes_data.append(node)

        edges_data = []
        conn = sqlite3.connect(self.graph_db_path)
        cursor = conn.cursor()
        try:
            for source_id, target_id in all_edges:
                cursor.execute(
                    "SELECT properties FROM edges WHERE (source = ? AND target = ?) OR (source = ? AND target = ?)",
                    (source_id, target_id, target_id, source_id)
                )
                row = cursor.fetchone()
                properties = json.loads(row[0]) if row and row[0] else {}
                edges_data.append({
                    'source': source_id,
                    'target': target_id,
                    'properties': properties
                })
        finally:
            conn.close()

        logger.info(f"N跳查询找到 {len(nodes_data)} 个节点, {len(edges_data)} 条边")
        return {'nodes': nodes_data, 'edges': edges_data}

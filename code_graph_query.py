#!/usr/bin/env python3
"""
Python代码图谱查询工具
支持通过Cypher查询和Embedding语义搜索两种方式查询code_graph.sqlite
metadata在SQLite，向量在FAISS（完全解耦）
"""

import json
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from faiss_store import FaissVectorStore
from code_graph_db import load_all_nodes, load_node, load_edges

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置 ====================

DEFAULT_CONFIG = {
    "embedding_base_url": "http://localhost:1234/v1/",
    "embedding_api_key": "not-needed",
    "embedding_model": "text-embedding-bge-m3",
}


def load_config() -> dict:
    """加载embedding配置"""
    config_file = Path(__file__).parent / "llm_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULT_CONFIG, **saved}
    return DEFAULT_CONFIG.copy()


def get_embedding(text: str, config: dict):
    """生成文本embedding"""
    client = OpenAI(base_url=config["embedding_base_url"], api_key=config["embedding_api_key"])
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=config["embedding_model"])
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"生成embedding失败: {e}")
        return None


# ==================== 查询引擎 ====================

class CodeGraphQueryEngine:
    """代码图谱查询引擎（metadata在SQLite, 向量在FAISS）"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if not Path(db_path).exists():
            raise FileNotFoundError(f"数据库不存在: {db_path}")

        # 初始化FAISS向量存储
        base_name = str(Path(db_path).with_suffix(""))
        self.vector_store = FaissVectorStore(base_name)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== Cypher-like 查询（metadata层面） ====================

    def cypher_match_nodes(self, label: str = None, where: str = None, limit: int = 50) -> List[Dict]:
        """类Cypher查询: MATCH (n:label) WHERE ... RETURN n"""
        all_nodes = load_all_nodes(self.db_path)
        nodes = []
        for props in all_nodes:
            if label and props.get('type') != label:
                continue
            if where and '=' in where:
                key, val = where.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if str(props.get(key, '')) != val:
                    continue
            nodes.append(props)
            if len(nodes) >= limit:
                break
        logger.info(f"cypher_match_nodes: 找到 {len(nodes)} 个节点")
        return nodes

    def cypher_match_edges(self, edge_type: str = None, limit: int = 100) -> List[Dict]:
        """类Cypher查询: MATCH ()-[r:type]->() RETURN r"""
        all_edges = load_edges(self.db_path)
        result = []
        for e in all_edges:
            props = {**e["properties"], "source": e["source"], "target": e["target"]}
            if edge_type and props.get('type') != edge_type:
                continue
            result.append(props)
            if len(result) >= limit:
                break
        logger.info(f"cypher_match_edges: 找到 {len(result)} 条边")
        return result

    def cypher_match_path(self, node_name: str, hops: int = 1, direction: str = "both") -> Dict:
        """类Cypher查询: MATCH path = (n)-[*1..hops]->() WHERE n.name = ... RETURN path"""
        all_nodes = load_all_nodes(self.db_path)

        # 找起点
        start_id = None
        for n in all_nodes:
            if n.get('name') == node_name:
                start_id = n['id']
                break

        if start_id is None:
            logger.warning(f"未找到节点: {node_name}")
            return {"nodes": [], "edges": []}

        # 确保 start_id 是 int 用于 SQLite 比较
        start_id = int(start_id)
        visited_nodes = {start_id}
        all_edges = []
        frontier = {start_id}

        conn = self._get_conn()
        try:
            for _ in range(hops):
                next_frontier = set()
                for nid in frontier:
                    if direction in ("outgoing", "both"):
                        cursor = conn.execute("SELECT target, properties FROM edges WHERE source = ?", (str(nid),))
                        for row in cursor:
                            target = int(row['target'])
                            props = json.loads(row['properties']) if row['properties'] else {}
                            all_edges.append({"source": nid, "target": target, **props})
                            if target not in visited_nodes:
                                visited_nodes.add(target)
                                next_frontier.add(target)
                    if direction in ("incoming", "both"):
                        cursor = conn.execute("SELECT source, properties FROM edges WHERE target = ?", (str(nid),))
                        for row in cursor:
                            source = int(row['source'])
                            props = json.loads(row['properties']) if row['properties'] else {}
                            all_edges.append({"source": source, "target": nid, **props})
                            if source not in visited_nodes:
                                visited_nodes.add(source)
                                next_frontier.add(source)
                frontier = next_frontier
                if not frontier:
                    break
        finally:
            conn.close()

        # 补充节点信息
        nodes = []
        for nid in visited_nodes:
            n = load_node(self.db_path, nid)
            if n:
                nodes.append(n)

        logger.info(f"cypher_match_path: {len(nodes)} 个节点, {len(all_edges)} 条边")
        return {"nodes": nodes, "edges": all_edges}

    # ==================== Embedding 语义搜索（FAISS） ====================

    def embedding_search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[Dict]:
        """
        基于FAISS的语义搜索（同时搜索comment和code向量）
        """
        config = load_config()
        query_emb = get_embedding(query, config)
        if not query_emb:
            logger.error("无法生成查询embedding")
            return []

        # FAISS搜索
        faiss_results = self.vector_store.search(query_emb, top_k=top_k, threshold=threshold, search_type="both")

        # 从SQLite补充完整metadata
        results = []
        for r in faiss_results:
            node_id = r["node_id"]
            n = load_node(self.db_path, node_id)
            if n:
                results.append({
                    "id": node_id,
                    "name": n.get('name', ''),
                    "path": n.get('path', ''),
                    "comment": n.get('comment', ''),
                    "score": round(r["score"], 4),
                })

        logger.info(f"embedding_search: 找到 {len(results)} 个结果 (阈值: {threshold})")
        return results

    # ==================== 综合查询 ====================

    def query(self, text: str, top_k: int = 10, hops: int = 1, threshold: float = 0.3) -> Dict:
        """
        综合查询：先FAISS搜索找最相关的文件，再做N跳扩展
        """
        matches = self.embedding_search(text, top_k=top_k, threshold=threshold)
        if not matches:
            return {"matches": [], "graph": {"nodes": [], "edges": []}}

        all_nodes = {}
        all_edges = []

        for match in matches:
            path_result = self.cypher_match_path(match['name'], hops=hops)
            for n in path_result['nodes']:
                if n['id'] not in all_nodes:
                    all_nodes[n['id']] = n
            for e in path_result['edges']:
                edge_key = f"{e['source']}-{e['target']}"
                if edge_key not in {f"{ee['source']}-{ee['target']}" for ee in all_edges}:
                    all_edges.append(e)

        return {
            "matches": matches,
            "graph": {"nodes": list(all_nodes.values()), "edges": all_edges}
        }


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Python代码图谱查询工具")
    parser.add_argument("--db", default="code_graph.sqlite", help="代码图谱数据库路径")

    subparsers = parser.add_subparsers(dest="command", help="查询命令")

    emb_parser = subparsers.add_parser("search", help="语义搜索 (FAISS Embedding)")
    emb_parser.add_argument("query", help="搜索文本")
    emb_parser.add_argument("--top-k", type=int, default=10, help="返回数量")
    emb_parser.add_argument("--threshold", type=float, default=0.3, help="相似度阈值")

    comp_parser = subparsers.add_parser("query", help="综合查询 (FAISS Embedding + Cypher)")
    comp_parser.add_argument("text", help="查询文本")
    comp_parser.add_argument("--top-k", type=int, default=10)
    comp_parser.add_argument("--hops", type=int, default=1)
    comp_parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()
    engine = CodeGraphQueryEngine(args.db)

    if args.command == "search":
        results = engine.embedding_search(args.query, top_k=args.top_k, threshold=args.threshold)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['name']} ({r['path']})")
            if r['comment']:
                print(f"         {r['comment'][:100]}...")

    elif args.command == "query":
        result = engine.query(args.text, top_k=args.top_k, hops=args.hops, threshold=args.threshold)
        print(f"=== 匹配文件 ({len(result['matches'])}) ===")
        for m in result['matches']:
            print(f"  [{m['score']:.4f}] {m['name']} ({m['path']})")
        print(f"\n=== 扩展子图: {len(result['graph']['nodes'])} 节点, {len(result['graph']['edges'])} 边 ===")
        for n in result['graph']['nodes']:
            print(f"  [{n.get('type')}] {n.get('name')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

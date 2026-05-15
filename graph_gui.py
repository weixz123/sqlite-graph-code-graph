#!/usr/bin/env python3
"""
知识图谱交互式GUI查询系统 - Flask后端
支持航空知识图谱查询 + Python代码图谱构建
metadata在SQLite，向量在FAISS（完全解耦）
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from code_graph_builder import CodeGraphBuilder, load_config, save_config
from faiss_store import FaissVectorStore
from code_graph_db import load_all_nodes, load_node, load_edges, delete_edge as delete_graph_edge
from simple_graph_sqlite import database as db

# ==================== 配置 ====================
GRAPH_DB_PATH = 'code_graph.sqlite'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== Flask 应用初始化 ====================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 代码图谱构建状态
build_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "result": None,
}


def _edge_key(edge):
    return (
        edge["source"],
        edge["target"],
        json.dumps(edge.get("properties", {}), ensure_ascii=False, sort_keys=True),
    )


def _collect_code_subgraph(db_path, start_ids, hops):
    """Collect an undirected n-hop subgraph for Web UI semantic search."""
    all_nodes = load_all_nodes(db_path)
    all_edges = load_edges(db_path)
    node_map = {n["id"]: n for n in all_nodes}
    visited = {node_id for node_id in start_ids if node_id in node_map}
    frontier = set(visited)
    subgraph_edges = []
    seen_edges = set()

    for _ in range(max(0, int(hops))):
        next_frontier = set()
        for node_id in frontier:
            for edge in all_edges:
                source = edge["source"]
                target = edge["target"]
                if source == node_id:
                    neighbor = target
                elif target == node_id:
                    neighbor = source
                else:
                    continue

                edge_key = _edge_key(edge)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    subgraph_edges.append(edge)

                if neighbor in node_map and neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return {
        "nodes": [node_map[node_id] for node_id in visited if node_id in node_map],
        "edges": subgraph_edges,
    }


# ==================== 页面路由 ====================

@app.route('/')
def index():
    return render_template('index.html')


# ==================== CRUD API ====================

@app.route('/api/node', methods=['POST'])
def add_node():
    data = request.json
    name = data.get('name')
    node_type = data.get('type')
    description = data.get('description', '')
    if not name or not node_type:
        return jsonify({"error": "节点名称和类型不能为空"}), 400
    try:
        all_nodes = db.atomic(GRAPH_DB_PATH, db.find_nodes([], ()))
        max_id = max([n['id'] for n in all_nodes]) if all_nodes else 0
        new_id = max_id + 1
        node_data = {'name': name, 'type': node_type, 'description': description}
        db.atomic(GRAPH_DB_PATH, db.add_node(node_data, new_id))
        return jsonify({"message": "节点添加成功", "id": new_id}), 201
    except Exception as e:
        logger.error(f"添加节点失败: {e}")
        return jsonify({"error": "添加节点失败"}), 500


@app.route('/api/node/<int:node_id>', methods=['PUT'])
def update_node(node_id):
    data = request.json
    try:
        db.atomic(GRAPH_DB_PATH, db.upsert_node(node_id, data))
        return jsonify({"message": f"节点 {node_id} 更新成功"})
    except Exception as e:
        logger.error(f"更新节点 {node_id} 失败: {e}")
        return jsonify({"error": "更新节点失败"}), 500


@app.route('/api/node/<int:node_id>', methods=['DELETE'])
def delete_node(node_id):
    try:
        db.atomic(GRAPH_DB_PATH, db.remove_node(node_id))
        return jsonify({"message": f"节点 {node_id} 删除成功"})
    except Exception as e:
        logger.error(f"删除节点 {node_id} 失败: {e}")
        return jsonify({"error": "删除节点失败"}), 500


@app.route('/api/edge', methods=['POST'])
def add_edge():
    data = request.json
    source = data.get('source')
    target = data.get('target')
    properties = data.get('properties', {})
    if source is None or target is None:
        return jsonify({"error": "源节点和目标节点不能为空"}), 400
    try:
        db.atomic(GRAPH_DB_PATH, db.connect_nodes(int(source), int(target), properties))
        return jsonify({"message": "关系添加成功"}), 201
    except Exception as e:
        logger.error(f"添加关系失败: {e}")
        return jsonify({"error": "添加关系失败"}), 500


@app.route('/api/edge', methods=['DELETE'])
def delete_edge():
    data = request.json
    source = data.get('source')
    target = data.get('target')
    properties = data.get('properties')
    if source is None or target is None:
        return jsonify({"error": "源节点和目标节点不能为空"}), 400
    try:
        deleted = delete_graph_edge(GRAPH_DB_PATH, int(source), int(target), properties)
        if deleted == 0:
            return jsonify({"error": "关系不存在"}), 404
        return jsonify({"message": "关系删除成功", "deleted": deleted})
    except Exception as e:
        logger.error(f"删除关系失败: {e}")
        return jsonify({"error": "删除关系失败"}), 500


# ==================== 代码图谱 API ====================

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取LLM配置"""
    return jsonify(load_config())


@app.route('/api/config', methods=['POST'])
def update_config():
    """保存LLM配置"""
    config = request.json
    save_config(config)
    return jsonify({"message": "配置保存成功"})


@app.route('/api/browse-path', methods=['POST'])
def browse_path():
    """浏览路径 - 返回目录下的py文件统计"""
    data = request.json
    path = data.get('path', '')
    if not path or not os.path.isdir(path):
        return jsonify({"error": "路径不存在或不是目录"}), 400

    from code_graph_builder import scan_python_files
    py_files = scan_python_files(path)
    return jsonify({
        "path": os.path.normpath(path),
        "file_count": len(py_files),
        "files": [os.path.relpath(f, path).replace("\\", "/") for f in py_files[:50]]
    })


@app.route('/api/build-code-graph', methods=['POST'])
def build_code_graph():
    """启动代码图谱构建（异步，默认 rebuild=True）"""
    global build_status

    if build_status["running"]:
        return jsonify({"error": "构建任务正在进行中"}), 409

    data = request.json
    path = data.get('path', '')
    output = data.get('output', 'code_graph.sqlite')
    rebuild = data.get('rebuild', True)
    generate_embeddings = data.get('generate_embeddings', True)
    generate_comments = data.get('generate_comments', True)

    if not path or not os.path.isdir(path):
        return jsonify({"error": "路径无效"}), 400

    config = load_config()
    build_status = {
        "running": True,
        "progress": 0,
        "total": 0,
        "message": "初始化...",
        "result": None,
    }

    def progress_callback(step, total, msg):
        build_status["progress"] = step
        build_status["total"] = total
        build_status["message"] = msg

    def run_build():
        try:
            builder = CodeGraphBuilder(
                path, output, config,
                rebuild=rebuild,
                generate_embeddings=generate_embeddings,
                generate_comments=generate_comments,
            )
            stats = builder.build(progress_callback=progress_callback)
            build_status["result"] = stats
            build_status["message"] = "构建完成"
        except Exception as e:
            logger.error(f"代码图谱构建失败: {e}")
            build_status["result"] = {"error": str(e)}
            build_status["message"] = f"构建失败: {e}"
        finally:
            build_status["running"] = False

    thread = threading.Thread(target=run_build, daemon=True)
    thread.start()
    return jsonify({"message": "构建任务已启动"})


@app.route('/api/build-status', methods=['GET'])
def get_build_status():
    """获取构建进度"""
    return jsonify(build_status)


# ==================== 代码图谱查询 API（FAISS） ====================

@app.route('/api/code-graph/search', methods=['POST'])
def search_code_graph():
    """在代码图谱中进行FAISS语义搜索"""
    data = request.json
    query = data.get('query', '')
    db_path = data.get('db_path', 'code_graph.sqlite')
    top_k = int(data.get('top_k', 10))
    threshold = float(data.get('threshold', 0.3))
    n_hops = max(0, int(data.get('n_hops', data.get('hops', 1))))

    if not query:
        return jsonify({"error": "查询不能为空"}), 400

    if not os.path.exists(db_path):
        return jsonify({"error": f"数据库 {db_path} 不存在，请先构建代码图谱"}), 404

    config = load_config()
    from code_graph_builder import get_embedding

    query_emb = get_embedding(query, config)
    if not query_emb:
        return jsonify({"error": "无法生成查询embedding"}), 500

    try:
        # FAISS向量搜索
        base_name = os.path.splitext(db_path)[0]
        vector_store = FaissVectorStore(base_name)
        faiss_results = vector_store.search(query_emb, top_k=top_k, threshold=threshold, search_type="both")

        # 从SQLite补充完整metadata
        results = []
        for r in faiss_results:
            node_id = r["node_id"]
            n = load_node(db_path, node_id)
            if n:
                results.append({
                    "id": node_id,
                    "name": n.get('name', ''),
                    "path": n.get('path', ''),
                    "comment": n.get('comment', ''),
                    "score": r["score"],
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        initial_nodes = {str(r["id"]): r["score"] for r in results}
        graph_data = _collect_code_subgraph(db_path, {r["id"] for r in results}, n_hops)
        graph_data["query"] = query
        graph_data["initial_nodes"] = initial_nodes
        return jsonify({
            "results": results,
            "graph_data": graph_data,
            "n_hops": n_hops,
        })
    except Exception as e:
        logger.error(f"代码图谱搜索失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/code-graph/overview', methods=['POST'])
def code_graph_overview():
    """返回代码图谱的统计概览和全部可视化数据"""
    data = request.json or {}
    db_path = data.get('db_path', 'code_graph.sqlite')

    if not os.path.exists(db_path):
        return jsonify({"error": f"数据库 {db_path} 不存在，请先构建代码图谱"}), 404

    try:
        all_nodes = load_all_nodes(db_path)
        edges = load_edges(db_path)

        node_ids = {n['id'] for n in all_nodes}
        edges = [e for e in edges if e['source'] in node_ids and e['target'] in node_ids]

        # 统计
        file_nodes = [n for n in all_nodes if n.get('type') == 'file']
        folder_nodes = [n for n in all_nodes if n.get('type') == 'folder']
        import_edges = [e for e in edges if e.get('properties', {}).get('type') == 'imports']
        contain_edges = [e for e in edges if e.get('properties', {}).get('type') == 'contains']
        commented = sum(1 for n in file_nodes if n.get('comment'))

        stats = {
            "total_nodes": len(all_nodes),
            "total_edges": len(edges),
            "file_count": len(file_nodes),
            "folder_count": len(folder_nodes),
            "import_count": len(import_edges),
            "contain_count": len(contain_edges),
            "commented_count": commented,
            "uncommented_count": len(file_nodes) - commented,
        }

        return jsonify({"stats": stats, "nodes": all_nodes, "edges": edges})
    except Exception as e:
        logger.error(f"获取代码图谱概览失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/code-graph/graph-data', methods=['POST'])
def get_code_graph_data():
    """获取代码图谱的可视化数据（metadata from SQLite）"""
    data = request.json
    db_path = data.get('db_path', 'code_graph.sqlite')
    node_filter = data.get('filter', '')

    if not os.path.exists(db_path):
        return jsonify({"error": f"数据库 {db_path} 不存在"}), 404

    try:
        all_nodes = load_all_nodes(db_path)

        if node_filter:
            all_nodes = [n for n in all_nodes if node_filter.lower() in n.get('path', '').lower() or node_filter.lower() in n.get('name', '').lower()]

        edges = load_edges(db_path)

        node_ids = {n['id'] for n in all_nodes}
        edges = [e for e in edges if e['source'] in node_ids and e['target'] in node_ids]

        return jsonify({"nodes": all_nodes, "edges": edges})
    except Exception as e:
        logger.error(f"获取代码图谱数据失败: {e}")
        return jsonify({"error": str(e)}), 500


# ==================== Query Sessions API ====================

SESSIONS_DIR = 'query_sessions'


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _load_session(session_id):
    fpath = _session_path(session_id)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_initial_nodes(session):
    graph_data = session.get("graph_data", {}) or {}
    initial_nodes = graph_data.get("initial_nodes")
    if initial_nodes:
        return initial_nodes

    try:
        parsed_result = json.loads(session.get("result_text", "") or "{}")
    except json.JSONDecodeError:
        parsed_result = {}

    matches = parsed_result.get("matches", []) if isinstance(parsed_result, dict) else []
    if matches:
        return {
            str(match.get("id")): match.get("score")
            for match in matches
            if match.get("id") is not None
        }

    if session.get("tool_name") == "search_code":
        return {
            str(node.get("id")): None
            for node in graph_data.get("nodes", [])[:10]
            if node.get("id") is not None
        }

    return {}


def _merge_session_graphs(sessions):
    nodes = {}
    edges = {}
    timeline = []

    for index, session in enumerate(sessions, start=1):
        graph_data = session.get("graph_data", {}) or {}
        session_id = session.get("id", f"session_{index}")
        created_at = session.get("created_at", "")
        query = graph_data.get("query") or session.get("query", "")

        for node in graph_data.get("nodes", []):
            nodes[node.get("id")] = node

        for edge in graph_data.get("edges", []):
            edges[_edge_key(edge)] = edge

        initial_nodes = _extract_initial_nodes(session)
        if query and initial_nodes:
            query_node_id = f"query_{index}_{session_id}"
            nodes[query_node_id] = {
                "id": query_node_id,
                "name": f"Query {index}",
                "type": "query",
                "path": query,
                "comment": f"{session.get('tool_name', '')} {created_at}",
            }
            for target_id, score in initial_nodes.items():
                try:
                    parsed_target_id = int(target_id)
                except (TypeError, ValueError):
                    parsed_target_id = target_id
                if parsed_target_id not in nodes:
                    continue
                edge = {
                    "id": f"similarity_{query_node_id}_{target_id}",
                    "source": query_node_id,
                    "target": parsed_target_id,
                    "properties": {
                        "type": "similarity",
                        "score": score,
                        "query": query,
                        "session_id": session_id,
                    },
                    "dashes": True,
                }
                edges[(edge["source"], edge["target"], edge["id"])] = edge

        timeline.append({
            "id": session_id,
            "tool_name": session.get("tool_name", ""),
            "query": session.get("query", ""),
            "created_at": created_at,
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
        })

    return {
        "tool_name": "merged_sessions",
        "query": "Merged query history",
        "result_text": json.dumps({"timeline": timeline}, ensure_ascii=False, indent=2),
        "graph_data": {
            "view_mode": "merged_sessions",
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        },
        "sessions": timeline,
    }


@app.route('/api/sessions', methods=['POST'])
def create_session():
    """
    创建一个查询 session（由 MCP Server 或前端调用）。
    body: {tool_name, query, result_text, graph_data: {nodes, edges}}
    """
    data = request.json
    _ensure_sessions_dir()

    session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    session = {
        "id": session_id,
        "tool_name": data.get("tool_name", "unknown"),
        "query": data.get("query", ""),
        "result_text": data.get("result_text", ""),
        "graph_data": data.get("graph_data", {"nodes": [], "edges": []}),
        "created_at": datetime.now().isoformat(),
    }

    fpath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

    logger.info(f"查询session已保存: {session_id} ({session['tool_name']})")
    return jsonify({"id": session_id, "message": "session已保存"}), 201


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """列出所有查询 session"""
    _ensure_sessions_dir()
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith('.json'):
            fpath = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    s = json.load(f)
                sessions.append({
                    "id": s["id"],
                    "tool_name": s.get("tool_name", ""),
                    "query": s.get("query", ""),
                    "created_at": s.get("created_at", ""),
                    "node_count": len(s.get("graph_data", {}).get("nodes", [])),
                    "edge_count": len(s.get("graph_data", {}).get("edges", [])),
                })
            except Exception:
                pass
    return jsonify({"sessions": sessions})


@app.route('/api/sessions', methods=['DELETE'])
def delete_sessions():
    """Delete multiple query sessions."""
    data = request.json or {}
    session_ids = data.get("ids", [])
    if not isinstance(session_ids, list) or not session_ids:
        return jsonify({"error": "ids must be a non-empty list"}), 400

    _ensure_sessions_dir()
    deleted = []
    missing = []
    for session_id in session_ids:
        fpath = _session_path(session_id)
        if os.path.exists(fpath):
            os.remove(fpath)
            deleted.append(session_id)
        else:
            missing.append(session_id)
    return jsonify({"deleted": deleted, "missing": missing})


@app.route('/api/sessions/merge', methods=['POST'])
def merge_sessions():
    """Merge multiple query sessions for observing an LLM agent search flow."""
    data = request.json or {}
    session_ids = data.get("ids", [])
    if not isinstance(session_ids, list) or len(session_ids) < 2:
        return jsonify({"error": "select at least two sessions"}), 400

    _ensure_sessions_dir()
    sessions = []
    missing = []
    for session_id in session_ids:
        session = _load_session(session_id)
        if session:
            sessions.append(session)
        else:
            missing.append(session_id)

    if len(sessions) < 2:
        return jsonify({"error": "not enough existing sessions", "missing": missing}), 404

    merged = _merge_session_graphs(sessions)
    merged["missing"] = missing
    return jsonify(merged)


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """获取单个 session 的完整数据"""
    _ensure_sessions_dir()
    fpath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(fpath):
        return jsonify({"error": "session不存在"}), 404
    with open(fpath, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """删除一个 session"""
    _ensure_sessions_dir()
    fpath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
        return jsonify({"message": "已删除"})
    return jsonify({"error": "session不存在"}), 404


@app.route('/api/sessions/latest', methods=['GET'])
def get_latest_session():
    """获取最新一个 session（用于前端轮询 MCP 推送的新查询）"""
    _ensure_sessions_dir()
    files = sorted([f for f in os.listdir(SESSIONS_DIR) if f.endswith('.json')], reverse=True)
    if not files:
        return jsonify({"session": None})
    fpath = os.path.join(SESSIONS_DIR, files[0])
    with open(fpath, "r", encoding="utf-8") as f:
        return jsonify({"session": json.load(f)})


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    logger.info("启动Flask服务器，请在浏览器中访问 http://127.0.0.1:9961")
    app.run(host='127.0.0.1', port=9961, debug=False)

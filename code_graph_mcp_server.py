#!/usr/bin/env python3
"""
Code Graph MCP Server.

The server exposes three LLM-facing tools:
  1. get_overview
  2. search_code
  3. get_file_info

Query results are optionally pushed to the Flask Web UI and saved as sessions.
"""

import argparse
import json
import logging
import os
import urllib.request

from mcp.server.fastmcp import FastMCP

from code_graph_db import load_all_nodes, load_node, load_edges

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


mcp = FastMCP(
    "code-graph",
    instructions=(
        "Code graph query tools. Use get_overview first to understand the "
        "project, then search_code to find and visualize relevant files, then "
        "get_file_info for exact file details and import summaries. Query "
        "results are pushed to the Web UI when it is running."
    ),
)

DB_PATH = "code_graph.sqlite"
WEB_UI_URL = "http://127.0.0.1:9961"


def _set_db(path: str):
    global DB_PATH
    DB_PATH = path


def _push_to_web(tool_name: str, query: str, result_text: str,
                 graph_data: dict | None = None):
    """Push a query session to the Flask Web UI if it is running."""
    if graph_data is None:
        graph_data = {"nodes": [], "edges": []}
    try:
        payload = json.dumps({
            "tool_name": tool_name,
            "query": query,
            "result_text": result_text[:5000],
            "graph_data": graph_data,
        }, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            f"{WEB_UI_URL}/api/sessions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 201:
                logger.info("Query result pushed to Web UI: %s", tool_name)
    except Exception as e:
        logger.debug("Web UI push failed without affecting tool result: %s", e)


def _find_file_node(nodes: list, file_path: str) -> dict | None:
    file_path_lower = file_path.lower().replace("\\", "/")
    for n in nodes:
        if n.get("type") == "file" and n.get("path", "").lower() == file_path_lower:
            return n
    for n in nodes:
        if n.get("type") == "file" and n.get("path", "").lower().endswith(file_path_lower):
            return n
    for n in nodes:
        if n.get("type") == "file" and n.get("name", "").lower() == file_path_lower.split("/")[-1]:
            return n
    return None


def _vis_node(node: dict) -> dict:
    return {
        "id": node["id"],
        "name": node.get("name", ""),
        "type": node.get("type", ""),
        "path": node.get("path", ""),
        "comment": node.get("comment", ""),
        "source_preview": node.get("source_preview", ""),
    }


def _edge_type(edge: dict) -> str:
    return edge.get("properties", {}).get("type", "")


def _collect_subgraph(start_ids: set[int], nodes: list[dict], edges: list[dict],
                      hops: int) -> tuple[list[dict], list[dict]]:
    """Collect an undirected n-hop subgraph from start node ids."""
    node_map = {n["id"]: n for n in nodes}
    visited = {nid for nid in start_ids if nid in node_map}
    frontier = set(visited)
    subgraph_edges = []
    seen_edges = set()

    for _ in range(max(0, int(hops))):
        next_frontier = set()
        for nid in frontier:
            for edge in edges:
                source = edge["source"]
                target = edge["target"]
                if source == nid:
                    neighbor = target
                elif target == nid:
                    neighbor = source
                else:
                    continue

                edge_key = (
                    source,
                    target,
                    json.dumps(edge.get("properties", {}), sort_keys=True, ensure_ascii=False),
                )
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    subgraph_edges.append(edge)

                if neighbor not in visited and neighbor in node_map:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return [_vis_node(node_map[nid]) for nid in visited if nid in node_map], subgraph_edges


def _summarize_imports(node: dict, nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    node_id = node["id"]
    node_map = {n["id"]: n for n in nodes}
    outgoing = []
    incoming = []
    import_edges = []

    for edge in edges:
        if _edge_type(edge) != "imports":
            continue
        props = edge.get("properties", {})
        if edge["source"] == node_id:
            target = node_map.get(edge["target"])
            if target:
                outgoing.append({
                    "target_file": target.get("path", target.get("name", "")),
                    "target_name": target.get("name", ""),
                    "import_name": props.get("import_name", ""),
                    "resolved_module": props.get("resolved_module", ""),
                })
                import_edges.append(edge)
        elif edge["target"] == node_id:
            source = node_map.get(edge["source"])
            if source:
                incoming.append({
                    "source_file": source.get("path", source.get("name", "")),
                    "source_name": source.get("name", ""),
                    "import_name": props.get("import_name", ""),
                    "resolved_module": props.get("resolved_module", ""),
                })
                import_edges.append(edge)

    return outgoing, incoming, import_edges


@mcp.tool()
def get_overview() -> str:
    """Return lightweight code graph statistics."""
    path = DB_PATH
    if not os.path.exists(path):
        return f"Error: database {path} does not exist. Build the code graph first."

    nodes = load_all_nodes(path)
    edges = load_edges(path)
    file_nodes = [n for n in nodes if n.get("type") == "file"]
    folder_nodes = [n for n in nodes if n.get("type") == "folder"]
    import_edges = [e for e in edges if _edge_type(e) == "imports"]
    contain_edges = [e for e in edges if _edge_type(e) == "contains"]
    commented = sum(1 for n in file_nodes if n.get("comment"))

    result = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "files": len(file_nodes),
        "folders": len(folder_nodes),
        "import_relations": len(import_edges),
        "contain_relations": len(contain_edges),
        "files_with_comments": commented,
        "files_without_comments": len(file_nodes) - commented,
        "sample_files": [n.get("path", n.get("name", "")) for n in file_nodes[:10]],
    }
    result_text = json.dumps(result, ensure_ascii=False, indent=2)
    _push_to_web("get_overview", "overview", result_text, {"nodes": [], "edges": []})
    return result_text


@mcp.tool()
def search_code(query: str, top_k: int = 10, threshold: float = 0.3,
                hops: int = 1) -> str:
    """
    Search code semantically and push the matched dependency subgraph to Web UI.

    hops=0 returns only matched nodes in the visual session.
    hops>=1 expands imports/contains neighbors around the matched files.
    """
    path = DB_PATH
    if not os.path.exists(path):
        return f"Error: database {path} does not exist."

    try:
        from code_graph_builder import get_embedding, load_config
        from faiss_store import FaissVectorStore
    except Exception as e:
        return f"Error: failed to import embedding modules: {e}"

    config = load_config()
    query_emb = get_embedding(query, config)
    if not query_emb:
        return "Error: failed to generate query embedding. Check the embedding service."

    vector_store = FaissVectorStore(os.path.splitext(path)[0])
    faiss_results = vector_store.search(
        query_emb,
        top_k=top_k,
        threshold=threshold,
        search_type="both",
    )

    if not faiss_results:
        result_text = json.dumps({"matches": [], "graph": {"nodes": 0, "edges": 0}}, ensure_ascii=False, indent=2)
        _push_to_web("search_code", query, result_text, {"nodes": [], "edges": []})
        return result_text

    all_nodes = load_all_nodes(path)
    all_edges = load_edges(path)
    matches = []
    matched_ids = set()
    initial_nodes = {}

    for result in faiss_results:
        node = load_node(path, result["node_id"])
        if not node:
            continue
        matched_ids.add(node["id"])
        score = round(result["score"], 4)
        initial_nodes[str(node["id"])] = score
        matches.append({
            "id": node.get("id"),
            "name": node.get("name", ""),
            "path": node.get("path", ""),
            "score": score,
            "comment_preview": (node.get("comment") or "")[:200],
        })

    graph_nodes, graph_edges = _collect_subgraph(matched_ids, all_nodes, all_edges, hops)
    response = {
        "matches": matches,
        "graph": {
            "nodes": len(graph_nodes),
            "edges": len(graph_edges),
            "hops": max(0, int(hops)),
        },
    }
    result_text = json.dumps(response, ensure_ascii=False, indent=2)
    _push_to_web(
        "search_code",
        query,
        result_text,
        {
            "nodes": graph_nodes,
            "edges": graph_edges,
            "query": query,
            "initial_nodes": initial_nodes,
        },
    )
    return result_text


@mcp.tool()
def get_file_info(file_path: str, include_imports: bool = True) -> str:
    """Return file metadata, source preview, comments, and optional import summaries."""
    path = DB_PATH
    if not os.path.exists(path):
        return f"Error: database {path} does not exist."

    nodes = load_all_nodes(path)
    node = _find_file_node(nodes, file_path)
    if not node:
        all_files = [n.get("path", n.get("name", "")) for n in nodes if n.get("type") == "file"]
        similar = [f for f in all_files if file_path.lower() in f.lower()][:10]
        hint = {"similar_files": similar} if similar else {"sample_files": all_files[:20]}
        return json.dumps({"error": f"File not found: {file_path}", **hint}, ensure_ascii=False, indent=2)

    result = {
        "id": node.get("id"),
        "name": node.get("name", ""),
        "path": node.get("path", ""),
        "type": node.get("type", ""),
        "comment": node.get("comment", ""),
        "source_preview": node.get("source_preview", ""),
    }

    edges = load_edges(path)
    graph_edges = []
    neighbor_ids = {node["id"]}
    if include_imports:
        outgoing, incoming, graph_edges = _summarize_imports(node, nodes, edges)
        result["outgoing_imports"] = outgoing
        result["imported_by"] = incoming
        for edge in graph_edges:
            neighbor_ids.add(edge["source"])
            neighbor_ids.add(edge["target"])

    graph_nodes = [_vis_node(n) for n in nodes if n["id"] in neighbor_ids]
    result_text = json.dumps(result, ensure_ascii=False, indent=2)
    _push_to_web("get_file_info", file_path, result_text, {"nodes": graph_nodes, "edges": graph_edges})
    return result_text


def main():
    parser = argparse.ArgumentParser(description="Code Graph MCP Server")
    parser.add_argument("--db", default="code_graph.sqlite", help="Code graph database path")
    parser.add_argument("--web-url", default="http://127.0.0.1:9961", help="Web UI URL")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], help="Transport")
    args = parser.parse_args()

    _set_db(args.db)
    global WEB_UI_URL
    WEB_UI_URL = args.web_url
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

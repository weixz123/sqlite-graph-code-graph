# sqlite-graph-code-graph

基于 `simple-graph-sqlite` + FAISS + LLM 的 **Python 代码图谱构建与查询系统**。

扫描 Python 项目源码，自动构建文件-文件夹-imports 依赖图谱，并通过 LLM 生成代码注释、FAISS 存储语义向量，支持语义搜索、多跳路径查询和交互式 Web 可视化。

## 系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ code_graph_      │     │  code_graph_     │     │  graph_gui.py   │
│ builder.py       │────▶│  mcp_server.py   │────▶│  (Flask Web UI) │
│ (图谱构建)       │     │  (MCP Tool 服务) │     │  可视化查询界面  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐
│ code_graph.sqlite│     │  FAISS 索引      │
│ (元数据存储)     │     │  (语义向量存储)   │
└─────────────────┘     └──────────────────┘
```

**核心设计**: SQLite 存储图谱元数据（节点/边），FAISS 存储语义向量，两者完全解耦。

## 功能特性

- **代码图谱构建**: 扫描 Python 项目，提取文件、文件夹、import 依赖关系
- **精确 import 解析**: 基于 AST 解析，支持绝对导入和相对导入的精确解析
- **LLM 自动注释**: 调用 LLM 为每个源文件生成功能描述注释
- **语义向量搜索**: 使用 embedding 模型（bge-m3）生成向量，存入 FAISS 索引
- **Cypher 风格查询**: 支持类 Cypher 的节点/边/路径查询
- **MCP Tool 服务**: 提供 `get_overview`、`search_code`、`get_file_info` 三个 MCP 工具
- **Web 可视化**: Flask Web 界面，支持图谱可视化、节点展开、交互查询
- **综合查询**: 语义搜索 + 多跳图扩展，查询结果自动推送到 Web UI

## 项目结构

```
├── code_graph_builder.py      # 代码图谱构建器（扫描+解析+构建）
├── code_graph_db.py           # SQLite 数据库适配器（屏蔽底层 schema）
├── code_graph_query.py        # 查询引擎（Cypher查询 + 语义搜索）
├── code_graph_mcp_server.py   # MCP Tool 服务（供 LLM 调用）
├── faiss_store.py             # FAISS 向量存储模块
├── graph_gui.py               # Flask Web 可视化界面
├── graph_utils.py             # 通用工具函数
├── simple_graph.py            # 航空知识图谱示例（独立）
├── llm_config.json            # LLM / Embedding 配置文件
├── start.bat                  # Windows 一键启动脚本
├── pyproject.toml             # 项目配置与依赖
├── requirements.txt           # pip 依赖列表
├── templates/
│   └── index.html             # Web UI 前端页面
└── tests/                     # 单元测试
```

## 快速开始

### 1. 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- 本地 LLM 服务（用于注释生成和 embedding）:
  - LM Studio 或兼容 OpenAI API 的服务
  - 聊天模型（如 `deepseek-chat`）
  - Embedding 模型（如 `bge-m3`）

### 2. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 3. 配置 LLM 服务

编辑 `llm_config.json`，填入你的 LLM 服务地址和 API Key：

```json
{
  "llm_base_url": "https://api.deepseek.com/v1",
  "llm_api_key": "your-api-key",
  "llm_model": "deepseek-chat",
  "embedding_base_url": "http://localhost:1234/v1/",
  "embedding_api_key": "not-needed",
  "embedding_model": "text-embedding-bge-m3"
}
```

### 4. 构建代码图谱

```bash
# 扫描指定 Python 项目，构建图谱
python code_graph_builder.py /path/to/your/python/project

# 重新构建（清空旧数据）
python code_graph_builder.py /path/to/your/python/project --rebuild

# 跳过注释或 embedding 生成（加速构建）
python code_graph_builder.py /path/to/project --no-comments
python code_graph_builder.py /path/to/project --no-embeddings
```

构建完成后会生成:
- `code_graph.sqlite` — 图谱元数据
- `code_graph.code.index` / `code_graph.comment.index` — FAISS 向量索引
- 对应的 `.ids.json` / `.meta.json` — ID 映射和元数据

### 5. 查询图谱

```bash
# 语义搜索
python code_graph_query.py search "数据查询逻辑" --top-k 10

# 综合查询（语义搜索 + 多跳扩展）
python code_graph_query.py query "数据库连接" --hops 2 --top-k 5
```

### 6. 启动 Web UI

```bash
# 方式一：使用启动脚本（Windows）
start.bat

# 方式二：直接启动
python graph_gui.py
```

启动后访问 http://127.0.0.1:9961

### 7. MCP Tool 服务

MCP 服务供 LLM 客户端（如 Claude Code）调用：

```bash
# 启动 MCP 服务
python code_graph_mcp_server.py --db code_graph.sqlite
```

MCP 客户端配置示例：
```json
{
  "mcpServers": {
    "code-graph": {
      "command": "uv",
      "args": ["--directory", "/path/to/project", "run", "code_graph_mcp_server.py", "--db", "/path/to/project/code_graph.sqlite"]
    }
  }
}
```

MCP 工具说明：

| 工具 | 说明 |
|------|------|
| `get_overview` | 获取代码图谱统计概览（节点数、边数、文件列表等） |
| `search_code` | 语义搜索代码，返回匹配结果和依赖子图 |
| `get_file_info` | 查询单个文件的详情、注释、import 关系 |

## 依赖

- [simple-graph-sqlite](https://pypi.org/project/simple-graph-sqlite/) — SQLite 图数据库
- [faiss-cpu](https://pypi.org/project/faiss-cpu/) — 向量相似性搜索
- [openai](https://pypi.org/project/openai/) — LLM / Embedding API 客户端
- [Flask](https://pypi.org/project/Flask/) — Web UI 后端
- [pyvis](https://pypi.org/project/pyvis/) — 网络可视化
- [mcp](https://pypi.org/project/mcp/) — MCP 协议服务端

## 许可

MIT License

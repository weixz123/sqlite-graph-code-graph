#!/usr/bin/env python3
"""
航空知识图谱构建与查询系统
基于 rag_system.db 构建 aviation_graph.sqlite 图数据库
复用现有的 LMStudio 和 BGE 嵌入代码
"""

import sqlite3
import json
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time
import re
from openai import OpenAI

from simple_graph_sqlite import database as db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aviation_graph_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== LMStudio 相关函数（复用原有代码）====================

# 配置OpenAI客户端连接到LMStudio的本地服务器
client = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key="not-needed"
)


def call_lmstudio_chat(prompt="你好,请介绍一下自己"):
    """
    使用OpenAI聊天补全API格式调用LMStudio的语言模型
    
    参数:
        prompt (str): 发送给模型的用户提示
        
    返回:
        str: 模型的响应文本
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


def extract_summary_from_response(response_text):
    """
    从LLM响应中提取真正的摘要内容，去除<think>标签及其内容
    
    参数:
        response_text (str): LLM返回的完整响应文本
        
    返回:
        str: 提取出的摘要内容
    """
    if not response_text:
        return ""
    
    # 移除<think>...</think>标签及其内容
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    cleaned_text = re.sub(think_pattern, '', response_text)
    
    if cleaned_text == response_text and '<think>' not in response_text:
        return response_text.strip()
    
    # 清理多余空白
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text




# ==================== 图数据库构建类 ====================

class AviationGraphBuilder:
    """航空知识图谱构建器"""
    
    def __init__(self, rag_db_path: str, graph_db_path: str):
        """
        初始化图谱构建器
        
        参数:
            rag_db_path: RAG系统数据库路径
            graph_db_path: 图数据库路径
        """
        self.rag_db_path = rag_db_path
        self.graph_db_path = graph_db_path
        
        # 初始化图数据库
        db.initialize(graph_db_path)
        logger.info(f"已初始化图数据库: {graph_db_path}")
        
        # 节点ID映射，避免重复
        self.node_id_map = {}
        self.next_id = 1
        
        # 统计信息
        self.stats = {
            'processed_chunks': 0,
            'extracted_entities': 0,
            'extracted_relations': 0,
            'failed_chunks': 0
        }
    
    def connect_rag_db(self) -> sqlite3.Connection:
        """连接到RAG数据库"""
        try:
            conn = sqlite3.connect(self.rag_db_path)
            logger.info(f"已连接到RAG数据库: {self.rag_db_path}")
            return conn
        except Exception as e:
            logger.error(f"连接RAG数据库失败: {str(e)}")
            raise
    
    def fetch_all_chunks(self, conn: sqlite3.Connection, limit: Optional[int] = None) -> List[Dict]:
        """
        从RAG数据库获取所有文档块
        
        参数:
            conn: 数据库连接
            limit: 限制返回数量（用于测试）
            
        返回:
            包含所有chunks的列表
        """
        cursor = conn.cursor()
        
        query = """
            SELECT c.id, d.source, c.text, c.summary, c.start_idx, c.end_idx, c.pages
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        chunks = []
        for row in cursor.fetchall():
            chunk_id, source, text, summary, start_idx, end_idx, pages_json = row
            chunks.append({
                'id': chunk_id,
                'source': source,
                'text': text,
                'summary': summary if summary else text[:500],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'pages': json.loads(pages_json) if pages_json else []
            })
        
        logger.info(f"从RAG数据库获取了 {len(chunks)} 个文档块")
        return chunks
    
    def extract_entities_and_relations(self, text: str, chunk_id: int) -> Dict:
        """
        使用LLM从文本中提取实体和关系
        
        参数:
            text: 要分析的文本
            chunk_id: 文档块ID
            
        返回:
            包含实体和关系的字典
        """
        prompt = f"""请分析以下航空与人因工程领域的文本，提取其中的实体和关系。

文本内容：
{text}

请按照以下JSON格式返回结果（只返回JSON，不要其他内容）：
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型（aircraft_model | airline | airport | technology | component | person | organization）",
            "description": "简短描述"
        }}
    ],
    "relations": [
        {{
            "source": "源实体名称",
            "target": "目标实体名称",
            "type": "关系类型（uses | manufactures | operates | located_at | part_of | developed_by）",
            "description": "关系描述"
        }}
    ]
}}

注意：
1. 只提取明确的航空与人因工程相关实体
2. 实体名称要准确、规范
3. 关系要清晰、有意义
4. 如果文本中没有明确的实体或关系，返回空列表"""

        try:
            response_text = call_lmstudio_chat(prompt)
            
            if not response_text:
                logger.warning(f"Chunk {chunk_id}: LLM未返回响应")
                return {'entities': [], 'relations': []}
            
            # 提取JSON内容
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # 尝试直接查找JSON对象
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
            
            # 解析JSON
            result = json.loads(json_text)
            
            entities = result.get('entities', [])
            relations = result.get('relations', [])
            
            logger.info(f"Chunk {chunk_id}: 提取了 {len(entities)} 个实体, {len(relations)} 个关系")
            
            return {'entities': entities, 'relations': relations}
            
        except json.JSONDecodeError as e:
            logger.error(f"Chunk {chunk_id}: JSON解析失败 - {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return {'entities': [], 'relations': []}
        except Exception as e:
            logger.error(f"Chunk {chunk_id}: 提取失败 - {str(e)}")
            return {'entities': [], 'relations': []}
    
    def get_or_create_node_id(self, entity_name: str, entity_type: str, description: str) -> int:
        """
        获取或创建节点ID
        
        参数:
            entity_name: 实体名称
            entity_type: 实体类型
            description: 实体描述
            
        返回:
            节点ID
        """
        # 使用实体名称作为唯一标识
        key = entity_name.lower().strip()
        
        if key in self.node_id_map:
            # 节点已存在，更新描述
            existing_id = self.node_id_map[key]
            db.atomic(self.graph_db_path, db.upsert_node(existing_id, {'description': description}))
            return existing_id
        else:
            # 创建新节点
            node_data = {
                'name': entity_name,
                'type': entity_type,
                'description': description
            }
            
            node_id = self.next_id
            db.atomic(self.graph_db_path, db.add_node(node_data, node_id))
            
            self.node_id_map[key] = node_id
            self.next_id += 1
            
            logger.debug(f"创建节点: {entity_name} (ID: {node_id})")
            return node_id
    
    def add_relation(self, source_name: str, target_name: str, relation_type: str, description: str):
        """
        添加关系
        
        参数:
            source_name: 源实体名称
            target_name: 目标实体名称
            relation_type: 关系类型
            description: 关系描述
        """
        source_key = source_name.lower().strip()
        target_key = target_name.lower().strip()
        
        if source_key not in self.node_id_map or target_key not in self.node_id_map:
            logger.warning(f"关系 {source_name} -> {target_name} 的实体不存在")
            return
        
        source_id = self.node_id_map[source_key]
        target_id = self.node_id_map[target_key]
        
        edge_data = {
            'type': relation_type,
            'description': description
        }
        
        db.atomic(self.graph_db_path, db.connect_nodes(source_id, target_id, edge_data))
        logger.debug(f"添加关系: {source_name} -> {target_name} ({relation_type})")
    
    def process_chunk(self, chunk: Dict):
        """
        处理单个chunk
        
        参数:
            chunk: 文档块数据
        """
        chunk_id = chunk['id']
        text = chunk['text']
        
        try:
            # 提取实体和关系
            extraction_result = self.extract_entities_and_relations(text, chunk_id)
            
            entities = extraction_result['entities']
            relations = extraction_result['relations']
            
            # 添加实体到图数据库
            for entity in entities:
                entity_name = entity.get('name', '').strip()
                entity_type = entity.get('type', 'unknown')
                description = entity.get('description', '')
                
                if entity_name:
                    self.get_or_create_node_id(entity_name, entity_type, description)
                    self.stats['extracted_entities'] += 1
            
            # 添加关系到图数据库
            for relation in relations:
                source_name = relation.get('source', '').strip()
                target_name = relation.get('target', '').strip()
                relation_type = relation.get('type', 'related')
                description = relation.get('description', '')
                
                if source_name and target_name:
                    self.add_relation(source_name, target_name, relation_type, description)
                    self.stats['extracted_relations'] += 1
            
            self.stats['processed_chunks'] += 1
            
        except Exception as e:
            logger.error(f"处理chunk {chunk_id} 失败: {str(e)}")
            self.stats['failed_chunks'] += 1
    
    def build_graph(self, batch_size: int = 10, delay: float = 0.5, limit: Optional[int] = None):
        """
        构建知识图谱
        
        参数:
            batch_size: 批处理大小
            delay: 批处理之间的延迟（秒）
            limit: 限制处理的chunk数量（用于测试）
        """
        logger.info("开始构建知识图谱...")
        
        # 连接RAG数据库
        rag_conn = self.connect_rag_db()
        
        try:
            # 获取所有chunks
            chunks = self.fetch_all_chunks(rag_conn, limit=limit)
            
            if not chunks:
                logger.warning("没有可处理的chunks")
                return
            
            # 批处理chunks
            total_chunks = len(chunks)
            logger.info(f"开始处理 {total_chunks} 个文档块...")
            
            with tqdm(total=total_chunks, desc="构建图谱") as pbar:
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i+batch_size]
                    
                    for chunk in batch:
                        self.process_chunk(chunk)
                        pbar.update(1)
                    
                    # 批处理延迟，避免过载
                    if i + batch_size < total_chunks:
                        time.sleep(delay)
            
            # 打印统计信息
            logger.info("\n" + "="*70)
            logger.info("构建完成!")
            logger.info(f"处理的文档块: {self.stats['processed_chunks']}")
            logger.info(f"失败的文档块: {self.stats['failed_chunks']}")
            logger.info(f"提取的实体: {self.stats['extracted_entities']}")
            logger.info(f"提取的关系: {self.stats['extracted_relations']}")
            logger.info(f"节点总数: {self.next_id - 1}")
            logger.info("="*70)
            
        finally:
            rag_conn.close()




# ==================== 主程序 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='航空知识图谱构建与查询系统')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # build 命令
    build_parser = subparsers.add_parser('build', help='构建知识图谱')
    build_parser.add_argument('--rag-db', type=str, default='./rag_data/rag_system.db',
                             help='RAG数据库路径')
    build_parser.add_argument('--graph-db', type=str, default='aviation_graph.sqlite',
                             help='图数据库路径')
    build_parser.add_argument('--batch-size', type=int, default=10,
                             help='批处理大小')
    build_parser.add_argument('--delay', type=float, default=0.5,
                             help='批处理延迟（秒）')
    build_parser.add_argument('--limit', type=int, default=None,
                             help='限制处理的chunk数量（用于测试）')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        # 构建图谱
        logger.info("开始构建航空知识图谱...")
        builder = AviationGraphBuilder(
            rag_db_path=args.rag_db,
            graph_db_path=args.graph_db
        )
        builder.build_graph(
            batch_size=args.batch_size,
            delay=args.delay,
            limit=args.limit
        )
        logger.info(f"\n图数据库已保存到: {args.graph_db}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

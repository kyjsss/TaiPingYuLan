#!/usr/bin/env python3
"""
《太平御览》GraphRAG 系统主入口模块

本模块是系统的命令行界面，负责集成所有模块，提供完整的 GraphRAG 功能。

使用方法:
    python main.py [OPTIONS] QUERY

示例:
    python main.py "荧惑守心有何预兆？"
    python main.py --mode macro "天人感应的原理"
    python main.py --output json "五行相生相克"
"""

import argparse
import sys
import logging
import asyncio
import json
from typing import Optional, Dict, Any
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入所有模块
from data_manager import DataManager
from query_processor import QueryProcessor
from retriever import create_retriever, MicroRetriever, MacroRetriever
from processor import Processor
from fusion import Fusion, FusionMode
from generator import Generator
from schemas import MicroRetrievalParams, MacroRetrievalParams

# 系统版本
SYSTEM_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='《太平御览》GraphRAG 系统 - 基于知识图谱的智能检索与生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "荧惑守心有何预兆？"
  python main.py --mode macro "天人感应的原理"
  python main.py --output json "五行相生相克"
  python main.py --verbose --no-cache "日食的记载"
        """
    )
    
    # 必需参数
    parser.add_argument(
        'query',
        type=str,
        help='用户查询（必需）'
    )
    
    # 主要参数
    parser.add_argument(
        '--mode',
        type=str,
        choices=['micro', 'macro'],
        default='micro',
        help='检索模式：micro（微观考据，默认）或 macro（宏观综述）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        choices=['text', 'json', 'markdown'],
        default='text',
        help='输出格式：text（纯文本，默认）、json（JSON格式）、markdown（Markdown格式）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（可选，默认使用 config.py）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细日志输出'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='禁用 QueryProcessor 缓存'
    )
    
    # Micro 模式参数
    micro_group = parser.add_argument_group('Micro 模式参数')
    
    micro_group.add_argument(
        '--graph-enable',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='是否启用图谱扩展（默认：True）'
    )
    
    micro_group.add_argument(
        '--anchor-quota-ratio',
        type=float,
        default=0.3,
        help='锚点配额比例（默认：0.3）'
    )
    
    micro_group.add_argument(
        '--neighbor-quota-ratio',
        type=float,
        default=0.3,
        help='邻居配额比例（默认：0.3）'
    )
    
    micro_group.add_argument(
        '--force-recall-count',
        type=int,
        default=20,
        help='语义保底数量（默认：20）'
    )
    
    micro_group.add_argument(
        '--neighbor-fanout',
        type=int,
        default=5,
        help='每个锚点允许扩展的最大邻居数量（默认：5）'
    )
    
    # Macro 模式参数
    macro_group = parser.add_argument_group('Macro 模式参数')
    
    macro_group.add_argument(
        '--top-k-macro',
        type=int,
        default=3,
        help='宏观锚点数（默认：3）'
    )
    
    macro_group.add_argument(
        '--bridge-fanout',
        type=int,
        default=5,
        help='桥接扇出数（默认：5）'
    )
    
    macro_group.add_argument(
        '--max-findings-per-query',
        type=int,
        default=5,
        help='核心观点数（默认：5）'
    )
    
    macro_group.add_argument(
        '--max-evidence-per-finding',
        type=int,
        default=2,
        help='观点例证数（默认：2）'
    )
    
    macro_group.add_argument(
        '--force-recall-count-macro',
        type=int,
        default=10,
        help='强制补全数（默认：10）'
    )
    
    return parser.parse_args()


def initialize_system(args: argparse.Namespace) -> dict:
    """
    初始化系统所有模块。
    
    Args:
        args: 命令行参数
        
    Returns:
        dict: 包含所有初始化模块的字典
    """
    logger.info("=" * 60)
    logger.info("系统初始化")
    logger.info("=" * 60)
    
    modules = {}
    
    # 1. 初始化 DataManager
    logger.info("初始化 DataManager...")
    try:
        data_manager = DataManager()
        data_manager.load_all_assets()
        modules['data_manager'] = data_manager
        logger.info(f"DataManager 初始化完成")
        logger.info(f"  - 社区数量: {len(data_manager.community_map)}")
        logger.info(f"  - 文本单元数量: {len(data_manager.text_unit_map)}")
        logger.info(f"  - 社区报告数量: {len(data_manager.community_reports)}")
    except Exception as e:
        logger.error(f"DataManager 初始化失败: {e}")
        raise
    
    # 2. 初始化 QueryProcessor
    logger.info("初始化 QueryProcessor...")
    try:
        query_processor = QueryProcessor(enable_cache=not args.no_cache)
        modules['query_processor'] = query_processor
        logger.info(f"QueryProcessor 初始化完成（缓存: {'启用' if not args.no_cache else '禁用'}）")
    except Exception as e:
        logger.error(f"QueryProcessor 初始化失败: {e}")
        raise
    
    # 3. 初始化 Retriever
    logger.info("初始化 Retriever...")
    try:
        # 根据模式构建检索参数
        if args.mode == 'micro':
            # 从 config.py 导入相似度阈值
            from config import SIMILARITY_THRESHOLD, RELATION_WEIGHT_THRESHOLD
            retrieval_params = MicroRetrievalParams(
                graph_enable=args.graph_enable,
                anchor_quota_ratio=args.anchor_quota_ratio,
                neighbor_quota_ratio=args.neighbor_quota_ratio,
                force_recall_count=args.force_recall_count,
                neighbor_fanout=args.neighbor_fanout,
                similarity_threshold=SIMILARITY_THRESHOLD,
                relation_weight_threshold=RELATION_WEIGHT_THRESHOLD,
            )
        else:  # macro
            retrieval_params = MacroRetrievalParams(
                top_k_macro=args.top_k_macro,
                bridge_fanout=args.bridge_fanout,
                max_findings_per_query=args.max_findings_per_query,
                max_evidence_per_finding=args.max_evidence_per_finding,
                force_recall_count=args.force_recall_count_macro,
            )
        
        # 创建检索器
        retriever = create_retriever(
            mode=args.mode,
            data_manager=data_manager,
            config=retrieval_params.model_dump() if hasattr(retrieval_params, 'model_dump') else retrieval_params.__dict__
        )
        modules['retriever'] = retriever
        modules['retrieval_params'] = retrieval_params
        logger.info(f"Retriever 初始化完成（模式: {args.mode}）")
    except Exception as e:
        logger.error(f"Retriever 初始化失败: {e}")
        raise
    
    # 4. 初始化 Processor
    logger.info("初始化 Processor...")
    try:
        processor = Processor(data_manager=data_manager)
        modules['processor'] = processor
        logger.info("Processor 初始化完成")
    except Exception as e:
        logger.error(f"Processor 初始化失败: {e}")
        raise
    
    # 5. 初始化 Fusion
    logger.info("初始化 Fusion...")
    try:
        fusion_mode = FusionMode.MICRO if args.mode == 'micro' else FusionMode.MACRO
        fusion = Fusion(mode=fusion_mode, system_prompt="", user_query=args.query)
        modules['fusion'] = fusion
        logger.info(f"Fusion 初始化完成（模式: {fusion_mode.value}）")
    except Exception as e:
        logger.error(f"Fusion 初始化失败: {e}")
        raise
    
    # 6. 初始化 Generator
    logger.info("初始化 Generator...")
    try:
        generator = Generator()
        modules['generator'] = generator
        logger.info("Generator 初始化完成")
    except Exception as e:
        logger.error(f"Generator 初始化失败: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("系统初始化完成")
    logger.info("=" * 60)
    
    return modules


def format_output_text(
    args: argparse.Namespace,
    translation_result,
    generation_result,
    fusion_stats: Dict[str, Any],
    processing_time: float
) -> str:
    """
    格式化文本输出。
    
    Args:
        args: 命令行参数
        translation_result: 查询翻译结果
        generation_result: 生成结果
        fusion_stats: 融合统计信息
        processing_time: 处理时间（秒）
        
    Returns:
        str: 格式化后的文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("查询结果")
    lines.append("=" * 60)
    lines.append(f"查询: {args.query}")
    lines.append(f"模式: {args.mode}")
    lines.append(f"处理时间: {processing_time:.2f} 秒")
    lines.append("")
    lines.append(generation_result.response_text)
    lines.append("")
    lines.append("=" * 60)
    lines.append("检索统计")
    lines.append("=" * 60)
    
    if fusion_stats:
        lines.append(f"检索结果数量: {fusion_stats.get('stuffed_count', 0)}")
        lines.append(f"上下文字符数: {fusion_stats.get('total_chars', 0)}")
        lines.append(f"预算使用率: {fusion_stats.get('budget_usage_percent', 0):.2f}%")
    
    lines.append(f"Prompt tokens: {generation_result.usage_info.get('prompt_tokens', 0)}")
    lines.append(f"Completion tokens: {generation_result.usage_info.get('completion_tokens', 0)}")
    lines.append(f"Total tokens: {generation_result.usage_info.get('total_tokens', 0)}")
    
    return "\n".join(lines)


def format_output_json(
    args: argparse.Namespace,
    translation_result,
    generation_result,
    fusion_stats: Dict[str, Any],
    processing_time: float
) -> str:
    """
    格式化 JSON 输出。
    
    Args:
        args: 命令行参数
        translation_result: 查询翻译结果
        generation_result: 生成结果
        fusion_stats: 融合统计信息
        processing_time: 处理时间（秒）
        
    Returns:
        str: 格式化后的 JSON 字符串
    """
    output = {
        "query": args.query,
        "mode": args.mode,
        "translated_query": translation_result.translated_query if translation_result else args.query,
        "response": generation_result.response_text,
        "statistics": {
            "processing_time": processing_time,
            "retrieval_count": fusion_stats.get('stuffed_count', 0) if fusion_stats else 0,
            "context_chars": fusion_stats.get('total_chars', 0) if fusion_stats else 0,
            "budget_usage": fusion_stats.get('budget_usage_percent', 0) if fusion_stats else 0,
            "prompt_tokens": generation_result.usage_info.get('prompt_tokens', 0),
            "completion_tokens": generation_result.usage_info.get('completion_tokens', 0),
            "total_tokens": generation_result.usage_info.get('total_tokens', 0),
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": SYSTEM_VERSION,
        }
    }
    
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_output_markdown(
    args: argparse.Namespace,
    translation_result,
    generation_result,
    fusion_stats: Dict[str, Any],
    processing_time: float
) -> str:
    """
    格式化 Markdown 输出。
    
    Args:
        args: 命令行参数
        translation_result: 查询翻译结果
        generation_result: 生成结果
        fusion_stats: 融合统计信息
        processing_time: 处理时间（秒）
        
    Returns:
        str: 格式化后的 Markdown 字符串
    """
    lines = []
    lines.append("# 《太平御览》GraphRAG 系统查询结果")
    lines.append("")
    lines.append(f"**查询**: {args.query}")
    lines.append(f"**模式**: {args.mode}")
    lines.append(f"**处理时间**: {processing_time:.2f} 秒")
    lines.append("")
    lines.append("## 回答")
    lines.append("")
    lines.append(generation_result.response_text)
    lines.append("")
    lines.append("## 统计信息")
    lines.append("")
    
    if fusion_stats:
        lines.append(f"- 检索结果数量: {fusion_stats.get('stuffed_count', 0)}")
        lines.append(f"- 上下文字符数: {fusion_stats.get('total_chars', 0)}")
        lines.append(f"- 预算使用率: {fusion_stats.get('budget_usage_percent', 0):.2f}%")
    
    lines.append(f"- Prompt tokens: {generation_result.usage_info.get('prompt_tokens', 0)}")
    lines.append(f"- Completion tokens: {generation_result.usage_info.get('completion_tokens', 0)}")
    lines.append(f"- Total tokens: {generation_result.usage_info.get('total_tokens', 0)}")
    
    return "\n".join(lines)


async def async_main(args: argparse.Namespace) -> int:
    """
    异步主函数：执行完整的 GraphRAG 查询流程。
    
    Args:
        args: 命令行参数
        
    Returns:
        int: 退出码（0 表示成功，非 0 表示失败）
    """
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("《太平御览》GraphRAG 系统启动")
    logger.info("=" * 60)
    logger.info(f"查询: {args.query}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"输出格式: {args.output}")
    
    # 批次2 - 系统初始化
    logger.info("系统初始化中...")
    try:
        modules = initialize_system(args)
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        return 1
    
    # 批次3 - 查询处理流程
    logger.info("=" * 60)
    logger.info("查询处理阶段")
    logger.info("=" * 60)
    
    data_manager = modules['data_manager']
    query_processor = modules['query_processor']
    
    # 1. 使用 QueryProcessor 翻译用户查询
    logger.info(f"翻译查询: {args.query}")
    try:
        translation_result = await query_processor.translate_query_with_metadata(args.query)
        logger.info(f"查询翻译完成")
        logger.info(f"  - 原始查询: {args.query}")
        logger.info(f"  - 增强查询: {translation_result.translated_query}")
        logger.info(f"  - 关键词: {translation_result.keywords}")
    except Exception as e:
        logger.error(f"查询翻译失败: {e}")
        return 1
    
    # 2. 生成查询向量
    logger.info("生成查询向量...")
    try:
        query_vector = data_manager.embed_text(translation_result.translated_query)
        logger.info(f"查询向量生成完成（维度: {len(query_vector)}）")
    except Exception as e:
        logger.error(f"查询向量生成失败: {e}")
        return 1
    
    # 批次4 - 检索流程
    logger.info("=" * 60)
    logger.info("检索阶段")
    logger.info("=" * 60)
    
    retriever = modules['retriever']
    processor = modules['processor']
    
    # 1. 执行检索
    logger.info(f"执行检索（模式: {args.mode}）...")
    try:
        retrieval_results = retriever.retrieve(
            query_vector=query_vector,
            query_text=translation_result.translated_query
        )
        logger.info(f"检索完成，获得 {len(retrieval_results)} 个结果")
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return 1
    
    # 2. 使用 Processor 处理检索结果
    logger.info("处理检索结果...")
    try:
        processed_results = processor.process(
            retrieval_results=retrieval_results,
            query_vector=query_vector,
            query_keywords=translation_result.keywords
        )
        logger.info(f"处理完成，生成 {len(processed_results)} 个处理结果")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1
    
    # 批次5 - 融合和生成流程
    logger.info("=" * 60)
    logger.info("融合与生成阶段")
    logger.info("=" * 60)
    
    fusion = modules['fusion']
    generator = modules['generator']
    retrieval_params = modules['retrieval_params']
    
    # 1. 使用 Fusion 融合处理结果并构建上下文
    logger.info("融合处理结果并构建上下文...")
    try:
        context_str = fusion.fuse_and_build_context(
            processed_results=processed_results,
            retrieval_params=retrieval_params
        )
        logger.info(f"上下文构建完成，字符数: {len(context_str)}")
        
        # 输出融合统计信息
        if fusion.last_stats:
            stats = fusion.last_stats
            logger.info(f"融合统计:")
            logger.info(f"  - 输入数量: {stats['input_count']}")
            logger.info(f"  - 归并数量: {stats['merged_count']}")
            logger.info(f"  - 去重数量: {stats['deduplicated_count']}")
            logger.info(f"  - 排序数量: {stats['sorted_count']}")
            logger.info(f"  - 装填数量: {stats['stuffed_count']}")
            logger.info(f"  - 总字符数: {stats['total_chars']}")
            logger.info(f"  - 预算使用率: {stats['budget_usage_percent']:.2f}%")
    except Exception as e:
        logger.error(f"融合失败: {e}")
        return 1
    
    # 2. 使用 Generator 生成最终回答
    logger.info("生成最终回答...")
    try:
        generation_result = await generator.generate(
            context_str=context_str,
            user_query=args.query,
            mode=args.mode
        )
        logger.info(f"生成完成")
        logger.info(f"  - 回答长度: {len(generation_result.response_text)} 字符")
        logger.info(f"  - Prompt tokens: {generation_result.usage_info.get('prompt_tokens', 0)}")
        logger.info(f"  - Completion tokens: {generation_result.usage_info.get('completion_tokens', 0)}")
        logger.info(f"  - Total tokens: {generation_result.usage_info.get('total_tokens', 0)}")
    except Exception as e:
        logger.error(f"生成失败: {e}")
        return 1
    
    # 批次6 - 输出格式化和错误处理
    logger.info("=" * 60)
    logger.info("输出格式化")
    logger.info("=" * 60)
    
    # 计算总处理时间
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 根据输出格式格式化结果
    try:
        if args.output == 'json':
            output_str = format_output_json(
                args=args,
                translation_result=translation_result,
                generation_result=generation_result,
                fusion_stats=fusion.last_stats,
                processing_time=processing_time
            )
        elif args.output == 'markdown':
            output_str = format_output_markdown(
                args=args,
                translation_result=translation_result,
                generation_result=generation_result,
                fusion_stats=fusion.last_stats,
                processing_time=processing_time
            )
        else:  # text (默认)
            output_str = format_output_text(
                args=args,
                translation_result=translation_result,
                generation_result=generation_result,
                fusion_stats=fusion.last_stats,
                processing_time=processing_time
            )
        
        # 输出结果
        print(output_str)
        logger.info(f"输出完成（格式: {args.output}）")
        
    except Exception as e:
        logger.error(f"输出格式化失败: {e}")
        return 1
    
    logger.info("=" * 60)
    logger.info("系统执行完成")
    logger.info(f"总处理时间: {processing_time:.2f} 秒")
    logger.info("=" * 60)
    
    return 0


def main() -> int:
    """
    主函数：执行完整的 GraphRAG 查询流程。
    
    Returns:
        int: 退出码（0 表示成功，非 0 表示失败）
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行异步主函数
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())

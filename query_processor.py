"""
QueryProcessor 模块 - 负责解决"本体错配"问题，将现代自然语言转化为"类书语境"查询

核心功能：
1. 调用 LLM 将现代查询转化为《太平御览》中的核心概念或部类术语
2. 关键词增强：将生成的关键词附加在原始查询后
3. 异步处理、重试机制、缓存机制、错误处理
4. 详细的日志记录和元数据返回

技术栈：
- AsyncOpenAI (SiliconFlow)
- tenacity 重试机制
- loguru 日志记录
- config.py 配置管理
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    ENABLE_QUERY_ENHANCEMENT,
)


@dataclass
class TranslationResult:
    """翻译结果数据类"""
    original_query: str
    translated_query: str
    keywords: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_query": self.original_query,
            "translated_query": self.translated_query,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }


class QueryProcessor:
    """QueryProcessor 类 - 负责查询理解和翻译"""
    
    # Prompt 模板（根据 project.md 2.2 节）
    TRANSLATION_PROMPT_TEMPLATE = """用户正在查询《太平御览》。请将现代查询 '{query}' 转化为 3-5 个《太平御览》中的核心概念或部类术语。

要求：
1. 输出必须是纯文本，用空格分隔的关键词
2. 关键词必须是《太平御览》中实际存在的部类、概念或术语
3. 不要添加任何解释或额外文本

示例：
输入："通货膨胀"
输出："食货 钱法 轻重"

输入："{query}"
输出："""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        enable_query_enhancement: Optional[bool] = None,
    ):
        """
        初始化 QueryProcessor
        
        Args:
            api_key: SiliconFlow API 密钥，默认为 config.SILICONFLOW_API_KEY
            base_url: API 基础 URL，默认为 config.SILICONFLOW_BASE_URL
            model: LLM 模型名称，默认为 config.LLM_MODEL
            temperature: 温度参数，默认为 config.LLM_TEMPERATURE
            max_tokens: 最大生成令牌数，默认为 config.LLM_MAX_TOKENS
            timeout: API 超时时间（秒），默认为 config.LLM_TIMEOUT
            enable_cache: 是否启用缓存，默认为 True
            cache_dir: 缓存目录，默认为 "./.query_cache"
            enable_query_enhancement: 是否启用查询增强（将关键词附加到原始查询后），默认为 config.ENABLE_QUERY_ENHANCEMENT
        """
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url or SILICONFLOW_BASE_URL
        self.model = model or LLM_MODEL
        self.temperature = temperature or LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self.timeout = timeout or LLM_TIMEOUT
        self.enable_cache = enable_cache
        self.enable_query_enhancement = enable_query_enhancement if enable_query_enhancement is not None else ENABLE_QUERY_ENHANCEMENT
        
        logger.debug(f"enable_query_enhancement 参数: {enable_query_enhancement}, 导入的 ENABLE_QUERY_ENHANCEMENT: {ENABLE_QUERY_ENHANCEMENT}, 最终值: {self.enable_query_enhancement}")
        
        # 初始化缓存
        self.cache_dir = Path(cache_dir or "./.query_cache")
        if self.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"查询缓存目录: {self.cache_dir.absolute()}")
        
        # 初始化 AsyncOpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        
        logger.info(f"QueryProcessor 初始化完成，模型: {self.model}, 缓存: {self.enable_cache}, 查询增强: {self.enable_query_enhancement}")
    
    def _get_cache_key(self, query: str) -> str:
        """生成缓存键"""
        # 使用 SHA256 哈希确保唯一性，包含查询和查询增强开关状态
        # 这样当 enable_query_enhancement 变化时，缓存键不同，避免错误命中
        key_string = f"{query}:{self.enable_query_enhancement}"
        hash_obj = hashlib.sha256(key_string.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存加载结果"""
        if not self.enable_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                logger.debug(f"从缓存加载查询结果: {cache_key}")
                return cached_data
            except Exception as e:
                logger.warning(f"缓存读取失败 {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """保存结果到缓存"""
        if not self.enable_cache:
            return
        
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"查询结果已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存写入失败 {cache_file}: {e}")
    
    def _build_translation_prompt(self, query: str) -> str:
        """
        构建翻译 Prompt
        
        Args:
            query: 用户原始查询
            
        Returns:
            完整的 Prompt 字符串
        """
        return self.TRANSLATION_PROMPT_TEMPLATE.format(query=query)
    
    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call_llm_for_translation(self, prompt: str) -> str:
        """
        调用 LLM 进行翻译（带重试机制）
        
        Args:
            prompt: 翻译 Prompt
            
        Returns:
            LLM 返回的关键词字符串
        """
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位精通《太平御览》的古代文献专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 提取响应内容
            response_text = response.choices[0].message.content.strip()
            
            logger.info(
                f"LLM 翻译调用成功: "
                f"prompt_tokens={response.usage.prompt_tokens}, "
                f"completion_tokens={response.usage.completion_tokens}, "
                f"elapsed_time={elapsed_time:.2f}s"
            )
            
            return response_text
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(
                f"LLM 翻译调用失败: {str(e)}, elapsed_time={elapsed_time:.2f}s"
            )
            raise
    
    def _parse_keywords(self, llm_output: str) -> List[str]:
        """
        解析 LLM 输出的关键词
        
        Args:
            llm_output: LLM 返回的文本
            
        Returns:
            关键词列表
        """
        # 清理输出：移除可能的标点、换行等
        cleaned = llm_output.strip()
        # 按空格分割
        keywords = [kw.strip() for kw in cleaned.split() if kw.strip()]
        return keywords
    
    def _enhance_query(self, original_query: str, keywords: List[str]) -> str:
        """
        增强查询：将关键词附加到原始查询后
        
        根据 project.md 2.2 节要求：
        - 将生成的关键词附加在原始 Query 后：final_query = f"{query} {keywords}"
        - 此功能可通过 enable_query_enhancement 开关控制
        
        Args:
            original_query: 原始查询
            keywords: 关键词列表
            
        Returns:
            增强后的查询（如果启用增强）或原始查询（如果禁用增强）
        """
        # 如果禁用查询增强，直接返回原始查询
        if not self.enable_query_enhancement:
            logger.info(f"查询增强已禁用 (enable_query_enhancement={self.enable_query_enhancement})，返回原始查询: {original_query}")
            return original_query
        
        # 如果没有关键词，直接返回原始查询
        if not keywords:
            return original_query
        
        # 将关键词附加到原始查询后
        keywords_str = " ".join(keywords)
        enhanced_query = f"{original_query} {keywords_str}"
        logger.debug(f"查询增强: '{original_query}' -> '{enhanced_query}' (关键词: {keywords})")
        return enhanced_query
    
    async def translate_query(self, user_query: str) -> str:
        """
        主翻译方法 - 返回增强后的查询字符串
        
        Args:
            user_query: 用户原始查询
            
        Returns:
            增强后的查询字符串
        """
        result = await self.translate_query_with_metadata(user_query)
        return result.translated_query
    
    async def translate_query_with_metadata(self, user_query: str) -> TranslationResult:
        """
        返回包含元数据的完整翻译结果
        
        Args:
            user_query: 用户原始查询
            
        Returns:
            TranslationResult 对象
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(user_query)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result:
            # 从缓存恢复结果
            processing_time = time.time() - start_time
            cached_result["metadata"]["processing_time"] = processing_time
            cached_result["metadata"]["cached"] = True
            return TranslationResult(**cached_result)
        
        # 初始化元数据
        metadata = {
            "translation_success": False,
            "error_message": None,
            "processing_time": 0,
            "cached": False,
            "llm_called": False,
        }
        
        try:
            # 1. 构建 Prompt
            prompt = self._build_translation_prompt(user_query)
            
            # 2. 调用 LLM
            llm_output = await self._call_llm_for_translation(prompt)
            metadata["llm_called"] = True
            
            # 3. 解析关键词
            keywords = self._parse_keywords(llm_output)
            
            # 4. 增强查询
            translated_query = self._enhance_query(user_query, keywords)
            
            # 5. 更新元数据
            metadata.update({
                "translation_success": True,
                "keywords_count": len(keywords),
                "llm_output": llm_output,
            })
            
        except Exception as e:
            # LLM 调用失败，回退到原始查询
            logger.error(f"查询翻译失败，回退到原始查询: {str(e)}")
            keywords = []
            translated_query = user_query
            metadata.update({
                "error_message": str(e),
                "fallback": True,
            })
        
        # 计算处理时间
        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time
        
        # 构建结果
        result = TranslationResult(
            original_query=user_query,
            translated_query=translated_query,
            keywords=keywords,
            metadata=metadata,
        )
        
        # 缓存结果（仅当翻译成功时）
        if metadata["translation_success"] and self.enable_cache:
            self._save_to_cache(cache_key, result.to_dict())
        
        return result


# 全局实例（可选）
_query_processor_instance: Optional[QueryProcessor] = None

def get_query_processor(**kwargs) -> QueryProcessor:
    """
    获取全局 QueryProcessor 实例（单例模式）
    
    Args:
        **kwargs: 传递给 QueryProcessor 构造函数的参数
        
    Returns:
        QueryProcessor 实例
    """
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor(**kwargs)
    return _query_processor_instance


async def async_main():
    """测试函数"""
    # 创建 QueryProcessor 实例
    processor = QueryProcessor(enable_cache=True)
    
    # 测试查询
    test_queries = [
        "通货膨胀",
        "气候变化对农业的影响",
        "古代天文观测",
        "测试一个很长的查询，看看LLM会生成什么关键词",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"原始查询: {query}")
        
        try:
            # 测试 translate_query（只返回增强查询）
            translated = await processor.translate_query(query)
            print(f"增强查询: {translated}")
            
            # 测试 translate_query_with_metadata（返回完整结果）
            result = await processor.translate_query_with_metadata(query)
            print(f"关键词: {result.keywords}")
            print(f"元数据: {result.metadata}")
            
        except Exception as e:
            print(f"处理失败: {e}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(async_main())
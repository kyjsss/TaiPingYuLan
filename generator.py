"""
Generator 模块 - 负责与 LLM 交互、系统提示词管理、异步 API 调用和结果生成

核心功能：
1. 知识边界 (The Boundary) - 强制 LLM 仅基于提供的 Context 回答
2. 原文优先 (Verbatim Preference) - 尽量原样摘录古文原文
3. 动态加载 System Prompt 模板（Micro/Macro 模式）
4. 异步 API 调用与重试机制
5. 完善的错误处理和日志记录

技术栈：
- AsyncOpenAI (SiliconFlow)
- tenacity 重试机制
- loguru 日志记录
- config.py 配置管理
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger
import numpy as np

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)


@dataclass
class GenerationResult:
    """生成结果数据类"""
    response_text: str
    usage_info: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "response_text": self.response_text,
            "usage_info": self.usage_info,
            "metadata": self.metadata,
        }


class Generator:
    """Generator 类 - 负责 LLM 交互和结果生成"""
    
    # System Prompt 模板
    MICRO_SYSTEM_PROMPT_TEMPLATE = """# Role
你是一位精通中国历史文献学（目录学/考据学）的资深专家。你的任务是基于提供的资料，为用户构建一个详实、严谨的考据回答。

   "### 要求：\n"
    "1. **宏微结合，见微知著**：特别关注资料中提取的独特信息（如异说、禁忌、特殊数据），不要遗漏长尾知识点。\n"
    "2. **多维互证**：如果资料中包含不同性质的文献（如“经史”与“志怪”、“官方记载”与“民间传说”），请明确区分语境，展示不同视角下的认知差异。\n"
    "3. **源流辨析**：如果不同来源对同一事物记载有矛盾（如年代、人名、属性冲突），请如实陈述这种学术张力，而非强行统一。\n"
    "4. **精准引注**：在引用具体观点或事实时，必须在括号中注明出处（如《史记》、《太平广记》），确保言必有据。\n"
    "5. **知之为知之**：严格基于提供的资料回答，不要使用外部知识进行幻觉式补充。\n\n"

# Context
{context_str}

# User Query
{user_query}"""

    MACRO_SYSTEM_PROMPT_TEMPLATE = """# Role
你是一位精通中国古代思想史的历史学家。你的任务是基于提供的【宏观综述】（包含逻辑观点、关联和补充史料），对用户的宏观问题进行综合论述。

# Rules
1. **Structure First**: 回答必须结构清晰，先抛出核心观点（Claims），再列举证据（Evidence）。
2. **Logic over Fact**: 重点阐述"天人感应"、"五行对应"等深层逻辑，而不仅仅是堆砌灾异记录。
   - 利用 Context 中的 `Logical Bridge` 信息（例如 "天部 --[RespondsTo]--> 人事部"）来构建你的论述框架。
4. **Synthesis**: 将【补充史料】作为论据融合到你的论述中，不要把它们当作孤立的附录。

# Context
{context_str}

# User Query
{user_query}"""
    
    # 知识边界声明（必须添加到所有 System Prompt 的开头）
    BOUNDARY_DECLARATION = """重要声明：请基于提供的 Context 回答。\n\n"""
    
    # 原文优先声明（必须添加到所有 System Prompt 的末尾）
    VERBATIM_PREFERENCE = """\n\n原文优先要求：对于古文原文的引用，请尽量**原样摘录**，避免意译或现代化表达，确保准确传达古文的原意."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """
        初始化 Generator
        
        Args:
            api_key: SiliconFlow API 密钥，默认为 config.SILICONFLOW_API_KEY
            base_url: API 基础 URL，默认为 config.SILICONFLOW_BASE_URL
            model: LLM 模型名称，默认为 config.LLM_MODEL
            temperature: 温度参数，默认为 config.LLM_TEMPERATURE
            max_tokens: 最大生成令牌数，默认为 config.LLM_MAX_TOKENS
            timeout: API 超时时间（秒），默认为 config.LLM_TIMEOUT
        """
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url or SILICONFLOW_BASE_URL
        self.model = model or LLM_MODEL
        self.temperature = temperature or LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self.timeout = timeout or LLM_TIMEOUT
        
        # 初始化 AsyncOpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        
        logger.info(f"Generator 初始化完成，模型: {self.model}, 温度: {self.temperature}")
    
    def _build_system_prompt(
        self,
        context_str: str,
        user_query: str,
        mode: str = "micro"
    ) -> str:
        """
        构建 System Prompt
        
        Args:
            context_str: 上下文字符串
            user_query: 用户查询
            mode: 模式，"micro" 或 "macro"
            
        Returns:
            完整的 System Prompt 字符串
        """
        # 选择模板
        if mode.lower() == "micro":
            template = self.MICRO_SYSTEM_PROMPT_TEMPLATE
        elif mode.lower() == "macro":
            template = self.MACRO_SYSTEM_PROMPT_TEMPLATE
        else:
            raise ValueError(f"未知模式: {mode}，必须是 'micro' 或 'macro'")
        
        # 渲染模板
        prompt = template.format(
            context_str=context_str,
            user_query=user_query
        )
        
        # 添加知识边界声明到开头
        prompt = self.BOUNDARY_DECLARATION + prompt
        
        # 添加原文优先声明到末尾
        prompt = prompt + self.VERBATIM_PREFERENCE
        
        return prompt
    
    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call_llm_api(
        self,
        system_prompt: str,
        user_query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 LLM API（带重试机制）
        
        Args:
            system_prompt: System Prompt
            user_query: 用户查询
            **kwargs: 额外的 API 参数
            
        Returns:
            API 响应字典
        """
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 提取响应内容
            response_text = response.choices[0].message.content
            
            # 提取使用情况
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": self.model,
                "elapsed_time": elapsed_time,
            }
            
            logger.info(
                f"LLM API 调用成功: "
                f"prompt_tokens={usage_info['prompt_tokens']}, "
                f"completion_tokens={usage_info['completion_tokens']}, "
                f"elapsed_time={elapsed_time:.2f}s"
            )
            
            return {
                "response_text": response_text,
                "usage_info": usage_info,
                "raw_response": response,
            }
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(
                f"LLM API 调用失败: {str(e)}, elapsed_time={elapsed_time:.2f}s"
            )
            raise
    
    async def generate(
        self,
        context_str: str,
        user_query: str,
        mode: str = "micro",
        **kwargs
    ) -> GenerationResult:
        """
        生成回答
        
        Args:
            context_str: 由 Fusion 模块生成的上下文字符串
            user_query: 用户原始查询
            mode: 检索模式，"micro" 或 "macro"
            **kwargs: 额外的 API 参数
            
        Returns:
            GenerationResult 对象
        """
        start_time = time.time()
        
        try:
            # 1. 构建 System Prompt
            system_prompt = self._build_system_prompt(
                context_str=context_str,
                user_query=user_query,
                mode=mode
            )
            
            logger.debug(f"System Prompt 长度: {len(system_prompt)} 字符")
            
            # 2. 调用 LLM API
            api_result = await self._call_llm_api(
                system_prompt=system_prompt,
                user_query=user_query,
                **kwargs
            )
            
            # 3. 构建元数据
            metadata = {
                "mode": mode,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt_length": len(system_prompt),
                "context_length": len(context_str),
                "query_length": len(user_query),
                "processing_time": time.time() - start_time,
            }
            
            # 4. 返回结果
            return GenerationResult(
                response_text=api_result["response_text"],
                usage_info=api_result["usage_info"],
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            raise
    
    async def batch_generate(
        self,
        contexts: list[str],
        queries: list[str],
        modes: Optional[list[str]] = None,
        **kwargs
    ) -> list[GenerationResult]:
        """
        批量生成回答
        
        Args:
            contexts: 上下文列表
            queries: 查询列表
            modes: 模式列表（可选，默认为 "micro"）
            **kwargs: 额外的 API 参数
            
        Returns:
            GenerationResult 列表
        """
        if modes is None:
            modes = ["micro"] * len(contexts)
        
        if len(contexts) != len(queries) or len(contexts) != len(modes):
            raise ValueError("contexts、queries 和 modes 的长度必须相同")
        
        # 创建任务列表
        tasks = [
            self.generate(
                context_str=context,
                user_query=query,
                mode=mode,
                **kwargs
            )
            for context, query, mode in zip(contexts, queries, modes)
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"第 {i} 个任务失败: {str(result)}")
                # 创建错误结果
                error_result = GenerationResult(
                    response_text=f"生成失败: {str(result)}",
                    usage_info={"error": str(result)},
                    metadata={"error": True, "index": i},
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results


# 全局实例（可选）
_generator_instance: Optional[Generator] = None

def get_generator(**kwargs) -> Generator:
    """
    获取全局 Generator 实例（单例模式）
    
    Args:
        **kwargs: 传递给 Generator 构造函数的参数
        
    Returns:
        Generator 实例
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = Generator(**kwargs)
    return _generator_instance


async def async_main():
    """测试函数"""
    # 创建 Generator 实例
    generator = Generator()
    
    # 测试数据
    test_context = """# 史料证据列表 (Historical Evidence)

[Source: 天部·日][prepend_source] 
《太平御览》卷三引《淮南子》曰："日者，阳之精也。"

[Source: 天部·月][prepend_source] 
《太平御览》卷四引《淮南子》曰："月者，阴之精也。" [关联证据: 通过 "CorrespondsTo" 关联到 "日"]"""
    
    test_query = "请解释《太平御览》中关于日月的记载。"
    
    # 生成回答
    try:
        result = await generator.generate(
            context_str=test_context,
            user_query=test_query,
            mode="micro"
        )
        
        print("生成结果:")
        print(f"回答: {result.response_text}")
        print(f"使用情况: {result.usage_info}")
        print(f"元数据: {result.metadata}")
        
    except Exception as e:
        print(f"生成失败: {e}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(async_main())
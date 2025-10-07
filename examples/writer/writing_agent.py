#!/usr/bin/env python3
"""
写作智能体示例 - 基于 Letta 的智能写作助手

这个示例展示了如何使用 Letta 创建一个专业的写作智能体，
具有写作风格记忆、写作工具和协作功能。

使用方法:
1. 确保 Letta 服务器正在运行: `letta server`
2. 运行此脚本: `python writing_agent.py`
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from letta_client import CreateBlock, Letta, MessageCreate


class WritingAgent:
    """基于 Letta 的写作智能体类"""
    
    def __init__(self, base_url: str = "http://localhost:8283", token: Optional[str] = None):
        """
        初始化写作智能体
        
        Args:
            base_url: Letta 服务器地址
            token: API 密钥（如果使用 Letta Cloud）
        """
        if token:
            self.client = Letta(token=token)
        else:
            self.client = Letta(base_url=base_url)
        
        self.agent = None
        self.writing_style = "专业、清晰、有逻辑性"
        self.current_project = None
        
    async def create_writing_agent(self, name: str = "写作助手", style: str = None) -> str:
        """
        创建写作智能体
        
        Args:
            name: 智能体名称
            style: 写作风格描述
            
        Returns:
            智能体 ID
        """
        if style:
            self.writing_style = style
            
        # 创建写作智能体的记忆块
        memory_blocks = [
            CreateBlock(
                label="persona",
                value=f"""你是一个专业的写作助手，具有以下特点：
1. 写作风格：{self.writing_style}
2. 擅长各种文体：学术论文、商业报告、创意写作、技术文档等
3. 注重逻辑性、清晰度和可读性
4. 能够根据读者群体调整写作风格
5. 具备研究和分析能力，能够提供有深度的内容"""
            ),
            CreateBlock(
                label="writing_skills",
                value="""核心写作技能：
- 结构化写作：能够组织清晰的文章结构
- 语言表达：使用准确、生动的语言
- 逻辑推理：构建有力的论证
- 读者导向：根据目标读者调整内容
- 创意表达：在保持专业性的同时展现创意"""
            ),
            CreateBlock(
                label="current_project",
                value="写作项目"
            )
        ]
        
        # 创建智能体
        self.agent = self.client.agents.create(
            name=name,
            memory_blocks=memory_blocks,
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            tools=["web_search"]  # 基础工具
        )
        
        print(f"✅ 写作智能体 '{name}' 创建成功！")
        print(f"智能体 ID: {self.agent.id}")
        return self.agent.id
    
    async def start_writing_project(self, project_name: str, project_type: str, 
                                 target_audience: str, requirements: str = "") -> None:
        """
        开始新的写作项目
        
        Args:
            project_name: 项目名称
            project_type: 项目类型（如：学术论文、商业报告、博客文章等）
            target_audience: 目标读者
            requirements: 特殊要求
        """
        if not self.agent:
            raise ValueError("请先创建写作智能体")
            
        project_info = f"""
写作项目信息：
- 项目名称：{project_name}
- 项目类型：{project_type}
- 目标读者：{target_audience}
- 特殊要求：{requirements or '无'}
- 开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 更新项目记忆
        await self._update_memory_block("current_project", project_info)
        self.current_project = project_name
        
        print(f"📝 开始写作项目：{project_name}")
        print(f"项目类型：{project_type}")
        print(f"目标读者：{target_audience}")
    
    async def generate_outline(self, topic: str, structure_type: str = "standard") -> str:
        """
        生成文章大纲
        
        Args:
            topic: 文章主题
            structure_type: 结构类型（standard, academic, business, creative）
            
        Returns:
            生成的大纲
        """
        if not self.agent:
            raise ValueError("请先创建写作智能体")
            
        prompt = f"""
请为以下主题生成详细的写作大纲：

主题：{topic}
结构类型：{structure_type}

请提供：
1. 文章标题建议
2. 主要章节结构
3. 每个章节的关键要点
4. 逻辑流程说明
5. 建议的写作顺序

请确保大纲逻辑清晰，结构合理。
"""
        
        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[MessageCreate(role="user", content=prompt)]
        )
        
        outline = response.messages[-1].content
        print("📋 生成的文章大纲：")
        print(outline)
        return outline
    
    async def expand_content(self, section: str, key_points: List[str], 
                           word_count: int = 500) -> str:
        """
        扩展内容段落
        
        Args:
            section: 章节名称
            key_points: 关键要点列表
            word_count: 目标字数
            
        Returns:
            扩展后的内容
        """
        if not self.agent:
            raise ValueError("请先创建写作智能体")
            
        points_text = "\n".join([f"- {point}" for point in key_points])
        
        prompt = f"""
请将以下章节内容扩展为完整的段落：

章节：{section}
关键要点：
{points_text}

要求：
1. 字数控制在 {word_count} 字左右
2. 保持逻辑清晰，过渡自然
3. 使用具体的例子和细节
4. 确保内容有深度和说服力
5. 语言流畅，符合目标读者需求

请直接输出扩展后的内容，不需要额外说明。
"""
        
        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[MessageCreate(role="user", content=prompt)]
        )
        
        content = response.messages[-1].content
        print(f"📝 扩展内容（{section}）：")
        print(content)
        return content
    
    async def polish_content(self, content: str, focus_areas: List[str] = None) -> str:
        """
        润色内容
        
        Args:
            content: 需要润色的内容
            focus_areas: 重点润色方面（如：语言流畅度、逻辑性、专业性等）
            
        Returns:
            润色后的内容
        """
        if not self.agent:
            raise ValueError("请先创建写作智能体")
            
        focus_text = ""
        if focus_areas:
            focus_text = f"\n重点润色方面：{', '.join(focus_areas)}"
            
        prompt = f"""
请对以下内容进行润色改进：

{content}

润色要求：
1. 改进语言表达，使其更加生动有力
2. 优化句子结构，提高可读性
3. 增强逻辑性和连贯性
4. 确保用词准确，避免重复
5. 保持原文的核心观点和结构{focus_text}

请直接输出润色后的内容，并简要说明主要改进点。
"""
        
        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[MessageCreate(role="user", content=prompt)]
        )
        
        polished = response.messages[-1].content
        print("✨ 润色后的内容：")
        print(polished)
        return polished
    
    async def adjust_style(self, content: str, target_style: str) -> str:
        """
        调整写作风格
        
        Args:
            content: 需要调整的内容
            target_style: 目标风格（如：正式、轻松、学术、商业等）
            
        Returns:
            调整后的内容
        """
        if not self.agent:
            raise ValueError("请先创建写作智能体")
            
        prompt = f"""
请将以下内容调整为 {target_style} 的写作风格：

{content}

风格调整要求：
1. 语言风格：{target_style}
2. 保持原文的核心信息和逻辑结构
3. 调整语调和表达方式以符合目标风格
4. 确保风格转换自然，不显突兀
5. 保持内容的专业性和准确性

请直接输出调整后的内容。
"""
        
        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[MessageCreate(role="user", content=prompt)]
        )
        
        adjusted = response.messages[-1].content
        print(f"🎨 风格调整后的内容（{target_style}）：")
        print(adjusted)
        return adjusted
    
    async def research_topic(self, topic: str, depth: str = "medium") -> str:
        """
        研究主题并收集信息
        
        Args:
            topic: 研究主题
            depth: 研究深度（shallow, medium, deep）
            
        Returns:
            研究结果
        """
        if not self.agent:
            raise ValueError("请先创建写作智能体")
            
        depth_instructions = {
            "shallow": "提供基础信息和概述",
            "medium": "提供详细信息和多个角度",
            "deep": "提供深入分析和专业见解"
        }
        
        prompt = f"""
请对以下主题进行 {depth_instructions[depth]} 的研究：

主题：{topic}

请提供：
1. 主题的核心概念和定义
2. 相关的重要事实和数据
3. 不同观点和争议点
4. 实际应用和案例
5. 进一步研究的建议

研究要求：{depth_instructions[depth]}，确保信息的准确性和相关性。
"""
        
        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[MessageCreate(role="user", content=prompt)]
        )
        
        research = response.messages[-1].content
        print(f"🔍 研究结果（{topic}）：")
        print(research)
        return research
    
    async def _update_memory_block(self, block_label: str, new_value: str) -> None:
        """更新记忆块"""
        try:
            # 获取当前记忆块
            # block = self.client.agents.blocks.retrieve(self.agent.id, block_label=block_label)
            # 更新记忆块
            self.client.agents.blocks.modify(
                agent_id=self.agent.id,
                block_label=block_label,
                value=new_value,
                read_only=False
            )
        except Exception as e:
            print(f"更新记忆块失败: {e}")
    
    async def get_writing_progress(self) -> Dict:
        """获取写作进度"""
        if not self.agent:
            return {"error": "智能体未创建"}
            
        try:
            # 获取项目记忆
            project_block = self.client.agents.blocks.retrieve(
                self.agent.id, block_label="current_project"
            )
            
            return {
                "current_project": project_block.value,
                "agent_id": self.agent.id,
                "agent_name": self.agent.name
            }
        except Exception as e:
            return {"error": f"获取进度失败: {e}"}


async def main():
    """主函数 - 演示写作智能体的使用"""
    print("🚀 启动写作智能体演示...")
    
    # 创建写作智能体
    writer = WritingAgent(token="sk-let-ZjEwZDkzMmQtYzk3NC00YzFjLWFlZGItNWZkNDA1ZmQ1NTBkOmMxMGNkZGQ2LTllMTgtNGZmNC1hODk0LWMxNDA4MGYyMWE2NA==")
    
    try:
        # 创建智能体
        agent_id = await writer.create_writing_agent(
            name="writer_agent_v3",
            style="专业、清晰、有逻辑性，适合学术和商业写作"
        )
        
        # 开始写作项目
        await writer.start_writing_project(
            project_name="AI 技术发展报告",
            project_type="技术报告",
            target_audience="技术管理者和决策者",
            requirements="需要包含最新趋势和实际应用案例"
        )
        
        # 生成大纲
        print("\n" + "="*50)
        outline = await writer.generate_outline(
            topic="人工智能技术的最新发展趋势及其对商业的影响",
            structure_type="business"
        )
        
        # 研究主题
        print("\n" + "="*50)
        research = await writer.research_topic(
            topic="人工智能在商业中的应用",
            depth="medium"
        )
        
        # 扩展内容
        print("\n" + "="*50)
        content = await writer.expand_content(
            section="AI 技术概述",
            key_points=[
                "机器学习的发展历程",
                "深度学习的关键突破",
                "大语言模型的应用",
                "AI 技术的商业化进程"
            ],
            word_count=300
        )
        
        # 润色内容
        print("\n" + "="*50)
        polished = await writer.polish_content(
            content=content,
            focus_areas=["语言流畅度", "逻辑性", "专业性"]
        )
        
        # 调整风格
        print("\n" + "="*50)
        adjusted = await writer.adjust_style(
            content=polished,
            target_style="轻松易懂"
        )
        
        # 显示进度
        print("\n" + "="*50)
        progress = await writer.get_writing_progress()
        print("📊 写作进度：")
        print(json.dumps(progress, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    print("\n✅ 写作智能体演示完成！")


if __name__ == "__main__":
    asyncio.run(main())

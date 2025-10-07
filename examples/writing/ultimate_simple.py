#!/usr/bin/env python3


import os
from letta_client import CreateBlock, Letta, MessageCreate


def create_writer():
    """创建写作智能体"""
    # 检查 DeepSeek API 密钥是否已设置
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 未找到 DEEPSEEK_API_KEY 环境变量")
        print("💡 请先设置环境变量：")
        print("   export DEEPSEEK_API_KEY='sk-your-api-key-here'")
        print("   然后重新启动 Letta 服务器")
        raise ValueError("需要设置 DEEPSEEK_API_KEY 环境变量")
    
    # 方式1：使用 Letta Cloud（需要 API 密钥）
    # client = Letta(token="sk-let-ZjEwZDkzMmQtYzk3NC00YzFjLWFlZGItNWZkNDA1ZmQ1NTBkOmMxMGNkZGQ2LTllMTgtNGZmNC1hODk0LWMxNDA4MGYyMWE2NA==")
    # agent = client.agents.create(
    #     name="writer", 
    #     memory_blocks=[CreateBlock(label="persona", value="你是专业写作助手，使用 DeepSeek 模型")]
    # )
    # 方式2：使用本地服务器（API 密钥在服务器启动时配置）
    client = Letta(base_url="http://localhost:8283")
    

    agent = client.agents.create(
        name="deepseek_writer", 
        memory_blocks=[CreateBlock(label="persona", value="你是专业写作助手，使用 DeepSeek 模型")],
        model="deepseek/deepseek-chat",
        embedding="ollama/nomic-embed-text:latest"
    )
    return client, agent

def write_article(client, agent, topic):
    """写作 - 2 行代码！"""
    response = client.agents.messages.create(agent_id=agent.id, messages=[MessageCreate(role="user", content=f"写一篇关于 {topic} 的文章")])
    return response.messages[-1].content if response.messages[-1].message_type == "assistant_message" else "写作完成"

# 🎯 使用示例 - 超简单！
if __name__ == "__main__":
    # 创建智能体
    client, agent = create_writer()
    print("✅ DeepSeek 写作智能体创建成功！")
    
    # 开始写作
    topic = input("请输入写作主题: ") or "人工智能"
    article = write_article(client, agent, topic)
    
    print(f"\n📝 关于 '{topic}' 的文章：")
    print("-" * 40)
    print(article)

# 第十三章：Agentic RAG — 从被动检索到主动推理

> **本章定位**：Agentic RAG 是 RAG 技术演进的最高阶段。它不再是"检索→生成"的线性管道，而是让 AI Agent 主动规划、自主决策、动态调用检索工具来完成复杂任务。本章从原理到工程实现，完整剖析 Agentic RAG 体系。

---

## 目录

- [13.1 核心问题：Agent 如何调用 RAG？](#131-核心问题agent-如何调用-rag)
- [13.2 ReAct 模式](#132-react-模式)
- [13.3 Plan & Execute 模式](#133-plan--execute-模式)
- [13.4 Reflection 模式](#134-reflection-模式)
- [13.5 Memory 体系](#135-memory-体系)
- [13.6 Tool Calling 机制](#136-tool-calling-机制)
- [13.7 多知识库路由](#137-多知识库路由)
- [13.8 多 Agent 协作](#138-多-agent-协作)
- [13.9 企业级 Agentic RAG 架构](#139-企业级-agentic-rag-架构)
- [13.10 高级 Agentic RAG 模式](#1310-高级-agentic-rag-模式)
- [13.11 面试高频问题](#1311-面试高频问题)
- [13.12 企业级最佳实践](#1312-企业级最佳实践)

---

## 13.1 核心问题：Agent 如何调用 RAG？

### 13.1.1 概念定义

**Agentic RAG** = Agent（自主决策与控制） + RAG（检索增强生成），即让具有自主推理能力的 AI Agent 动态决定**何时检索、检索什么、如何检索、检索后如何处理**。

```
传统 RAG:
  User Query → [固定管道: 检索→生成] → Answer
  特点: 被动、线性、不可控

Agentic RAG:
  User Query → Agent [思考→决策→调用工具→评估→迭代] → Answer
                 ↕
            Tool Set: [RAG检索, 数据库查询, 计算器, API调用, ...]
  特点: 主动、迭代、可控
```

### 13.1.2 产生背景

1. **传统 RAG 的局限性**：
   - 检索是固定模式，无法根据查询复杂度自适应
   - 无法处理需要多步推理的复杂问题
   - 无法根据检索结果的质量动态调整策略
   - 检索和生成之间缺乏反馈循环

2. **技术催化剂**：
   - GPT-4/Claude 等模型的推理能力大幅提升
   - Function Calling / Tool Use 原生化
   - ReAct、Chain-of-Thought 等推理框架成熟
   - LangChain/LlamaIndex 等框架提供了 Agent 抽象

3. **业务需求驱动**：
   - 企业需要处理多知识库的复杂查询
   - 需要可解释的检索决策链路
   - 需要自适应的检索策略

### 13.1.3 解决的问题

| 问题 | 传统 RAG | Agentic RAG |
|------|---------|-------------|
| 简单事实查询 | ✅ 高效 | ✅ 高效（路由到直接检索） |
| 多步推理问题 | ❌ 力不从心 | ✅ 分步检索+推理 |
| 跨知识库查询 | ❌ 需要预配置 | ✅ 动态路由 |
| 检索结果矛盾 | ❌ 无法处理 | ✅ 反思+重新检索 |
| 答案验证 | ❌ 不验证 | ✅ 自我批评+修正 |
| 工具调用 | ❌ 仅检索 | ✅ 检索+计算+API调用 |

### 13.1.4 Agent 调用 RAG 的核心流程

```
                    ┌──────────────────────────────────┐
                    │         Agent Controller          │
                    │  ┌─────────────────────────────┐ │
User Query ────────▶│  │  1. THINK: 分析问题         │ │
                    │  │  2. DECIDE: 选择策略/工具    │ │
                    │  │  3. ACT: 调用 RAG Tool      │ │
                    │  │  4. OBSERVE: 评估结果        │ │
                    │  │  5. ITERATE: 是否继续?       │ │
                    │  └─────────────────────────────┘ │
                    │           │          ▲            │
                    │           ▼          │            │
                    │  ┌─────────────────────────────┐ │
                    │  │       Tool Registry          │ │
                    │  │  ┌─────────┐ ┌────────────┐ │ │
                    │  │  │ RAG     │ │ Calculator │ │ │
                    │  │  │ Search  │ │            │ │ │
                    │  │  ├─────────┤ ├────────────┤ │ │
                    │  │  │ KB      │ │ SQL        │ │ │
                    │  │  │ Router  │ │ Query      │ │ │
                    │  │  ├─────────┤ ├────────────┤ │ │
                    │  │  │ Web     │ │ Code       │ │ │
                    │  │  │ Search  │ │ Exec       │ │ │
                    │  │  └─────────┘ └────────────┘ │ │
                    │  └─────────────────────────────┘ │
                    └──────────────────────────────────┘
```

### 13.1.5 技术选型

| 维度 | 轻量级方案 | 企业级方案 |
|------|-----------|-----------|
| Agent 框架 | LangChain Agent | 自研 Agent + 状态机 |
| LLM 推理 | GPT-4o-mini | Claude Opus / GPT-4o |
| 工具注册 | 硬编码字典 | 插件化 Tool Registry |
| 状态管理 | 内存字典 | Redis/PostgreSQL 持久化 |
| 可观测性 | print 日志 | LangSmith / OpenTelemetry |

---

## 13.2 ReAct 模式

### 13.2.1 概念定义

**ReAct** = **Re**asoning + **Act**ing，即让 LLM 交替进行"思考"和"行动"的循环模式。由 Yao et al. (2022) 提出，是 Agentic RAG 最基础也最广泛使用的模式。

### 13.2.2 工作原理

```
ReAct 循环:

  Thought → Action → Observation → Thought → Action → ... → Final Answer

详细流程:

  User: "2024年Q3中国新能源汽车市场份额最高的品牌是什么？它的电池供应商有哪些？"

  Round 1:
    Thought: 我需要先查询2024年Q3中国新能源汽车市场份额数据
    Action: RAG_Search(query="2024年Q3 中国 新能源汽车 市场份额 排名")
    Observation: [检索到5个相关文档片段...比亚迪市场份额32.5%...]

  Round 2:
    Thought: 市场份额最高的是比亚迪(32.5%)，现在需要查比亚迪的电池供应商
    Action: RAG_Search(query="比亚迪 电池供应商 供应链 2024")
    Observation: [检索到3个相关文档...弗迪电池、宁德时代、国轩高科...]

  Round 3:
    Thought: 我已经获得了问题的完整答案，可以总结了
    Final Answer: 2024年Q3中国新能源汽车市场份额最高的是比亚迪(32.5%)，
                 其主要电池供应商包括弗迪电池(自研)、宁德时代和国轩高科。
```

### 13.2.3 核心算法与实现

```python
import re
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class ReActStep:
    """ReAct 循环中的单步"""
    thought: str
    action: str | None  # None 表示 Final Answer
    action_input: str | None
    observation: str | None

class ReActAgent:
    """ReAct Agent 实现 — 与 RAG 系统集成"""

    REACT_PROMPT = """你是一个智能助手，可以调用工具来回答问题。

可用工具:
{tool_descriptions}

请按以下格式回答:

Question: 用户的问题
Thought: 我应该如何解决这个问题？
Action: 工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复 Thought/Action/Action Input/Observation 多次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

开始!

Question: {question}
{history}"""

    def __init__(self, tools: Dict[str, Callable], llm, max_steps: int = 10):
        self.tools = tools
        self.llm = llm  # LLM 接口，需要实现 generate(prompt) 方法
        self.max_steps = max_steps
        self.history: List[ReActStep] = []

    def _parse_output(self, text: str) -> Dict[str, str]:
        """解析 LLM 输出，提取 Thought/Action/Action Input/Final Answer"""
        result = {}

        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final)|\Z)', text, re.DOTALL)
        if thought_match:
            result['thought'] = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(r'Action:\s*(.+?)(?=\n|\Z)', text)
        if action_match:
            result['action'] = action_match.group(1).strip()

        # 提取 Action Input
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n(?:Observation|Thought)|\Z)', text, re.DOTALL)
        if input_match:
            result['action_input'] = input_match.group(1).strip()

        # 提取 Final Answer
        final_match = re.search(r'Final Answer:\s*(.+?)(?=\Z)', text, re.DOTALL)
        if final_match:
            result['final_answer'] = final_match.group(1).strip()

        return result

    def _build_prompt(self, question: str) -> str:
        """构建 ReAct prompt"""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])

        history_str = ""
        for step in self.history:
            history_str += f"\nThought: {step.thought}"
            if step.action:
                history_str += f"\nAction: {step.action}"
                history_str += f"\nAction Input: {step.action_input}"
                history_str += f"\nObservation: {step.observation}"

        return self.REACT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            question=question,
            history=history_str
        )

    def run(self, question: str) -> str:
        """执行 ReAct 循环"""
        for step_num in range(self.max_steps):
            # 1. 构建 prompt
            prompt = self._build_prompt(question)

            # 2. LLM 推理
            response = self.llm.generate(prompt)

            # 3. 解析输出
            parsed = self._parse_output(response)

            # 4. 判断是否结束
            if 'final_answer' in parsed:
                return parsed['final_answer']

            # 5. 执行动作
            action_name = parsed.get('action')
            if action_name and action_name in self.tools:
                action_result = self.tools[action_name](parsed.get('action_input', ''))

                self.history.append(ReActStep(
                    thought=parsed.get('thought', ''),
                    action=action_name,
                    action_input=parsed.get('action_input', ''),
                    observation=str(action_result)
                ))
            else:
                self.history.append(ReActStep(
                    thought=parsed.get('thought', ''),
                    action=None,
                    action_input=None,
                    observation=f"Error: Tool '{action_name}' not found"
                ))

        return "Error: Max steps reached without reaching a final answer."


# ===== 使用示例：RAG Tool 集成 =====
def rag_search_tool(query: str) -> str:
    """RAG 检索工具 — 在知识库中搜索相关信息。输入：自然语言查询字符串"""
    # 实际项目中，这里调用完整的 RAG pipeline
    retriever = get_retriever()  # 获取向量检索引擎
    results = retriever.search(query, top_k=5)
    return format_retrieval_results(results)


def kb_router_tool(query: str) -> str:
    """知识库路由工具 — 根据查询选择合适的知识库并检索。输入：查询字符串"""
    router = get_kb_router()
    kb = router.route(query)  # 返回合适的知识库
    results = kb.search(query, top_k=5)
    return format_retrieval_results(results)


def calculator_tool(expression: str) -> str:
    """数学计算工具 — 安全地计算数学表达式。输入：数学表达式字符串"""
    import ast, operator
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mul: operator.mul, ast.Div: operator.truediv
    }
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 初始化 Agent
tools = {
    "RAG_Search": rag_search_tool,
    "KB_Router": kb_router_tool,
    "Calculator": calculator_tool,
}
agent = ReActAgent(tools=tools, llm=my_llm, max_steps=10)
answer = agent.run("2024年新能源汽车市场分析")
```

### 13.2.4 优缺点分析

| 优点 | 缺点 |
|------|------|
| 模式简单，易于实现和调试 | 固定格式解析脆弱，非结构化输出容易失败 |
| 可解释性强，每步推理可见 | Token 消耗大（每轮需要完整的 prompt + history） |
| 灵活性高，可组合任意工具 | 可能陷入循环或选择错误的工具 |
| LLM 原生支持，无需额外训练 | 复杂问题的步数可能过多 |

### 13.2.5 性能优化

1. **Prompt 压缩**：随着 history 增长，使用摘要替代完整历史
2. **Early Stopping**：设置置信度阈值，满足条件后直接输出
3. **并行 Tool Calling**：当多个工具调用无依赖时并行执行
4. **结果缓存**：相同 Action+Input 的结果缓存，减少重复调用
5. **工具限流**：限制每步调用工具数量和总步骤数，防止失控

---

## 13.3 Plan & Execute 模式

### 13.3.1 概念定义

**Plan & Execute** 是 Agent 的先规划后执行模式：Agent 首先制定完整的执行计划，然后逐步执行每个步骤，并根据执行结果动态调整计划。

### 13.3.2 工作原理

```
Phase 1: Planning (规划)
  ┌─────────────────────────────────────────────────┐
  │ User Query → LLM(Planner) → Execution Plan      │
  │                                                    │
  │ Plan:                                              │
  │   Step 1: RAG_Search("Q3新能源汽车市场份额")        │
  │   Step 2: Extract top brand from Step 1 results    │
  │   Step 3: RAG_Search("[top brand] 电池供应商")      │
  │   Step 4: Verify info consistency across sources   │
  │   Step 5: Compose final answer                     │
  └─────────────────────────────────────────────────┘

Phase 2: Execution (执行)
  ┌─────────────────────────────────────────────────┐
  │ for each step in plan:                           │
  │   result = execute(step)                         │
  │   if result insufficient:                        │
  │     replan()  ← 触发重规划                        │
  │   else:                                          │
  │     continue                                     │
  │                                                    │
  │ return synthesize(all_results)                   │
  └─────────────────────────────────────────────────┘
```

### 13.3.3 核心实现

```python
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLAN = "replan"

class PlanStep(BaseModel):
    """计划中的一个步骤"""
    step_id: int
    description: str
    tool_name: str
    tool_input: str
    depends_on: List[int] = []  # 依赖的步骤 ID
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None

class ExecutionPlan(BaseModel):
    """完整执行计划"""
    query: str
    steps: List[PlanStep]
    current_step: int = 0

class PlanAndExecuteAgent:
    """Plan & Execute Agent 实现"""

    PLANNER_PROMPT = """你是一个任务规划专家。请为以下问题制定详细的执行计划。

可用工具:
{tool_descriptions}

请输出 JSON 格式的执行计划:
{{
  "steps": [
    {{
      "step_id": 1,
      "description": "步骤描述",
      "tool_name": "使用的工具名",
      "tool_input": "工具输入参数",
      "depends_on": []
    }},
    ...
  ]
}}

规则:
1. 每个步骤必须使用指定的工具之一
2. 如果步骤 B 依赖步骤 A 的结果，在 depends_on 中标注
3. 尽量保持步骤原子化（一个步骤做一件事）
4. 最后一步应该是综合所有信息生成答案

Question: {question}"""

    REPLANNER_PROMPT = """当前计划执行遇到问题，请重新规划后续步骤。

原始问题: {question}
已执行的步骤和结果:
{completed_steps}
当前失败的步骤:
{failed_step}

请输出剩余步骤的 JSON 计划。"""

    def __init__(self, tools: Dict[str, Callable], llm):
        self.tools = tools
        self.llm = llm

    def _create_plan(self, question: str) -> ExecutionPlan:
        """阶段1: 生成执行计划"""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__}"
            for name, func in self.tools.items()
        ])

        prompt = self.PLANNER_PROMPT.format(
            tool_descriptions=tool_descriptions,
            question=question
        )

        response = self.llm.generate(prompt)
        plan_dict = json.loads(extract_json(response))

        steps = [PlanStep(**step) for step in plan_dict['steps']]
        return ExecutionPlan(query=question, steps=steps)

    def _replan(self, plan: ExecutionPlan, failed_step: PlanStep) -> ExecutionPlan:
        """阶段2+: 动态重规划"""
        completed = [
            f"Step {s.step_id}: {s.description} → {s.result}"
            for s in plan.steps
            if s.status == StepStatus.COMPLETED
        ]

        prompt = self.REPLANNER_PROMPT.format(
            question=plan.query,
            completed_steps="\n".join(completed),
            failed_step=f"Step {failed_step.step_id}: {failed_step.description}"
        )

        response = self.llm.generate(prompt)
        new_steps_dict = json.loads(extract_json(response))

        # 替换未执行的步骤
        remaining_steps = [s for s in plan.steps if s.status == StepStatus.PENDING]
        new_steps = [PlanStep(**step) for step in new_steps_dict['steps']]
        # 重新编号
        for i, step in enumerate(new_steps):
            step.step_id = remaining_steps[0].step_id + i if remaining_steps else len(plan.steps) + i

        plan.steps = [s for s in plan.steps if s.status == StepStatus.COMPLETED] + new_steps
        return plan

    def _execute_step(self, step: PlanStep, plan: ExecutionPlan) -> PlanStep:
        """执行单个步骤"""
        step.status = StepStatus.EXECUTING

        # 处理依赖：如果输入中包含 {step_N} 引用，替换为实际结果
        actual_input = step.tool_input
        for dep_id in step.depends_on:
            dep_step = next(s for s in plan.steps if s.step_id == dep_id)
            actual_input = actual_input.replace(
                f"{{step_{dep_id}}}", dep_step.result or ""
            )

        # 调用工具
        if step.tool_name in self.tools:
            try:
                result = self.tools[step.tool_name](actual_input)
                step.result = str(result)
                step.status = StepStatus.COMPLETED
            except Exception as e:
                step.result = f"Error: {str(e)}"
                step.status = StepStatus.FAILED
        else:
            step.result = f"Tool '{step.tool_name}' not found"
            step.status = StepStatus.FAILED

        return step

    def run(self, question: str) -> str:
        """执行 Plan & Execute 循环"""
        # Phase 1: 规划
        plan = self._create_plan(question)

        # Phase 2: 执行
        max_replans = 3
        replan_count = 0

        while plan.current_step < len(plan.steps):
            step = plan.steps[plan.current_step]

            # 检查依赖是否满足
            deps_ready = all(
                plan.steps[d-1].status == StepStatus.COMPLETED
                for d in step.depends_on
            )
            if not deps_ready:
                plan.current_step += 1
                continue

            # 执行步骤
            step = self._execute_step(step, plan)
            plan.steps[plan.current_step] = step

            # 失败处理
            if step.status == StepStatus.FAILED and replan_count < max_replans:
                plan = self._replan(plan, step)
                replan_count += 1
                continue

            plan.current_step += 1

        # 综合结果
        results = "\n".join([
            f"Step {s.step_id}: {s.description}\nResult: {s.result}"
            for s in plan.steps if s.status == StepStatus.COMPLETED
        ])

        final_prompt = f"""基于以下执行结果，回答问题：{question}

执行结果:
{results}

请给出综合性的最终答案。"""

        return self.llm.generate(final_prompt)
```

### 13.3.4 ReAct vs Plan & Execute 对比

| 维度 | ReAct | Plan & Execute |
|------|-------|----------------|
| **规划方式** | 逐步规划，每步决策 | 前置完整规划，可重规划 |
| **Token 效率** | 较低（每轮完整 prompt） | 中等（规划阶段一次） |
| **灵活性** | 高，可根据最新结果调整 | 中等，通过重规划机制调整 |
| **执行效率** | 可并行调用工具 | 步骤级串行（依赖感知） |
| **适用场景** | 探索性强、不确定性高 | 目标明确、步骤可预定义 |
| **错误恢复** | 自然重试 | 需要通过重规划机制 |
| **实现复杂度** | 较低 | 较高 |

---

## 13.4 Reflection 模式

### 13.4.1 概念定义

**Reflection**（反思）是 Agent 的自我评估机制：Agent 生成初步答案后，对自己的输出进行评估，识别问题（不完整、不一致、幻觉），然后触发补充检索或修正。

### 13.4.2 工作原理

```
Initial Query
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Retrieve   │────▶│   Generate   │────▶│   Reflect    │
│   (RAG)      │     │   (LLM)      │     │   (Critic)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                    ┌─────────────┴─────────────┐
                                    │                            │
                              答 案 满 意                    答案不满意
                                    │                            │
                                    ▼                            ▼
                            ┌──────────────┐          ┌──────────────┐
                            │   Return     │          │ Refine Query │
                            │   Answer     │          │ Re-retrieve  │
                            └──────────────┘          └──────────────┘
```

### 13.4.3 核心实现

```python
class ReflectionAgent:
    """带反思机制的 RAG Agent"""

    REFLECTION_PROMPT = """你是一个严格的答案审核专家。请评估以下答案的质量。

原始问题: {question}
检索到的上下文: {context}
生成的答案: {answer}

请评估以下维度 (1-5分):
1. 完整性: 答案是否完全回答了问题？
2. 准确性: 答案是否与上下文一致？是否有编造内容？
3. 一致性: 答案内部是否有矛盾？
4. 可验证性: 答案中的每个声明是否都有来源支持？

如果任何维度 ≤ 3 分，请指出具体问题并建议如何改进。

输出格式:
{{
  "scores": {{"completeness": 5, "accuracy": 5, "consistency": 5, "verifiability": 5}},
  "passed": true/false,
  "issues": ["问题描述"],
  "improvement_hints": ["改进建议"],
  "refined_query": "如果需要重新检索，提供精炼后的查询"
}}"""

    def __init__(self, retriever, llm, max_reflections: int = 2):
        self.retriever = retriever
        self.llm = llm
        self.max_reflections = max_reflections

    def _retrieve(self, query: str, context_history: List[str] = None) -> str:
        """检索相关上下文"""
        results = self.retriever.search(query, top_k=5)
        # 过滤掉已经使用过的上下文
        if context_history:
            results = [r for r in results if r.content not in context_history]
        return format_retrieval_results(results)

    def _generate(self, question: str, context: str, reflection_hints: List[str] = None) -> str:
        """基于上下文生成答案"""
        hints_text = ""
        if reflection_hints:
            hints_text = "\n改进建议（请特别注意）:\n" + "\n".join(
                f"- {h}" for h in reflection_hints
            )

        prompt = f"""基于以下上下文回答问题。请确保答案准确、完整、可追溯。

上下文:
{context}

{hints_text}

问题: {question}

要求:
1. 每个关键声明都要标注来源（citation）
2. 如果上下文信息不足，明确说明
3. 不要编造上下文没有的信息

答案:"""

        return self.llm.generate(prompt)

    def _reflect(self, question: str, context: str, answer: str) -> Dict:
        """反思评估"""
        prompt = self.REFLECTION_PROMPT.format(
            question=question, context=context, answer=answer
        )
        response = self.llm.generate(prompt)
        return json.loads(extract_json(response))

    def run(self, question: str) -> Dict:
        """执行 Reflection RAG 循环"""
        context_history = []
        reflection_history = []

        for i in range(self.max_reflections + 1):
            # 1. 检索（使用改进后的查询）
            refined_query = question
            if reflection_history:
                refined_query = reflection_history[-1].get('refined_query', question)

            context = self._retrieve(refined_query, context_history)
            context_history.append(context)

            # 2. 生成答案
            hints = reflection_history[-1].get('improvement_hints', []) if reflection_history else None
            answer = self._generate(question, context, hints)

            # 3. 反思评估
            reflection = self._reflect(question, context, answer)
            reflection_history.append(reflection)

            # 4. 判断是否通过
            if reflection.get('passed', False):
                return {
                    "answer": answer,
                    "reflections": reflection_history,
                    "iterations": i + 1,
                    "status": "success"
                }

        # 达到最大反思次数，返回最后一次答案
        return {
            "answer": answer,
            "reflections": reflection_history,
            "iterations": self.max_reflections + 1,
            "status": "max_reflections_reached"
        }
```

### 13.4.4 反思维度的设计

| 反思维度 | 检查内容 | 典型问题 | 修正策略 |
|---------|---------|---------|---------|
| **完整性** | 是否回答了问题的所有部分 | 只回答了部分问题 | 补充检索缺失信息 |
| **准确性** | 答案是否与检索结果一致 | 数字错误、张冠李戴 | 重新检索 + 严格溯源 |
| **一致性** | 答案内部是否有逻辑矛盾 | 前后矛盾的数据 | 识别矛盾源 → 重新验证 |
| **可验证性** | 每个声明是否都可溯源 | 无来源支撑的断言 | 添加 citation，或删除无支撑内容 |
| **粒度** | 答案的详细程度是否匹配问题 | 过于笼统或过于细节 | 调整答案粒度 |

---

## 13.5 Memory 体系

### 13.5.1 概念定义

Agent Memory 是 Agent 的持久化记忆系统，使 Agent 能够跨会话记住用户偏好、历史交互和学到的知识。

### 13.5.2 Memory 类型

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Memory Architecture                 │
│                                                              │
│  ┌─────────────────┐                                        │
│  │ Working Memory  │  ← 当前会话上下文（短期）              │
│  │ (Conversation   │     - 最近N轮对话                      │
│  │  Buffer)        │     - 当前任务的中间结果               │
│  └────────┬────────┘     - Token预算: 不超过模型上下文的50% │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ Episodic Memory │  ← 历史会话摘要（中期）                │
│  │ (Session        │     - 按会话/Task组织                   │
│  │  Summaries)     │     - 关键决策点记录                   │
│  └────────┬────────┘     - 检索方式: 向量相似度             │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ Semantic Memory │  ← 知识/事实记忆（长期）               │
│  │ (Knowledge      │     - 用户偏好/习惯                    │
│  │  Store)         │     - 学到的领域知识                   │
│  └─────────────────┘     - 检索方式: 向量+关键词            │
└─────────────────────────────────────────────────────────────┘
```

### 13.5.3 实现

```python
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

class AgentMemory:
    """Agent 三层记忆系统"""

    def __init__(self, vector_store, redis_client, llm):
        self.vector_store = vector_store
        self.redis = redis_client
        self.llm = llm
        self.working_memory: List[Dict] = []  # 当前会话
        self.max_working_memory = 20  # 最多保留20轮

    # ===== Working Memory (短期) =====
    def add_to_working(self, role: str, content: str, metadata: Dict = None):
        """添加到工作记忆"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.working_memory.append(entry)

        # 超出容量时，将最早的条目归档到 Episodic Memory
        if len(self.working_memory) > self.max_working_memory:
            evicted = self.working_memory.pop(0)
            self._archive_to_episodic([evicted])

    def get_working_context(self, last_n: int = 10) -> str:
        """获取最近 N 轮对话作为上下文"""
        recent = self.working_memory[-last_n:]
        return "\n".join([
            f"{entry['role']}: {entry['content'][:200]}"
            for entry in recent
        ])

    # ===== Episodic Memory (中期) =====
    def _archive_to_episodic(self, entries: List[Dict]):
        """将工作记忆归档为情节记忆"""
        # 生成摘要
        content = "\n".join([f"{e['role']}: {e['content']}" for e in entries])
        summary_prompt = f"""请用1-2句话总结以下对话片段的关键信息：

{content}

总结:"""

        summary = self.llm.generate(summary_prompt)

        # 存储到向量数据库
        doc_id = hashlib.md5(content.encode()).hexdigest()
        self.vector_store.add(
            id=f"episodic:{doc_id}",
            text=summary,
            metadata={
                "type": "episodic",
                "timestamp": datetime.now().isoformat(),
                "entry_count": len(entries)
            }
        )

    def search_episodic(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关的情节记忆"""
        results = self.vector_store.search(
            query, top_k=top_k,
            filter={"type": "episodic"}
        )
        return results

    # ===== Semantic Memory (长期) =====
    def save_knowledge(self, key: str, value: str, category: str = "general"):
        """保存长期知识"""
        self.redis.hset(
            f"memory:semantic:{category}",
            key,
            json.dumps({
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "access_count": 0
            })
        )

    def recall_knowledge(self, key: str, category: str = "general") -> Optional[str]:
        """召回长期知识"""
        data = self.redis.hget(f"memory:semantic:{category}", key)
        if data:
            record = json.loads(data)
            record['access_count'] += 1
            self.redis.hset(f"memory:semantic:{category}", key, json.dumps(record))
            return record['value']
        return None

    def semantic_search_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义搜索长期知识"""
        return self.vector_store.search(
            query, top_k=top_k,
            filter={"type": "semantic"}
        )

    # ===== 用户偏好学习 =====
    def learn_preference(self, user_id: str, preference_type: str, value: Any):
        """学习用户偏好"""
        key = f"preference:{user_id}:{preference_type}"
        self.save_knowledge(key, json.dumps(value), category="preferences")

    def get_user_preferences(self, user_id: str) -> Dict:
        """获取用户的所有偏好"""
        prefs = self.redis.hgetall(f"memory:semantic:preferences")
        return {
            k.decode().replace(f"preference:{user_id}:", ""): json.loads(v.decode())
            for k, v in prefs.items()
            if k.decode().startswith(f"preference:{user_id}:")
        }

    # ===== Memory 集成到 RAG Pipeline =====
    def enrich_query_with_memory(self, user_id: str, query: str) -> str:
        """用记忆增强查询上下文"""
        enriched_parts = [query]

        # 1. 获取用户偏好
        prefs = self.get_user_preferences(user_id)
        if prefs:
            pref_context = "用户偏好: " + ", ".join(
                f"{k}={v}" for k, v in prefs.items()
            )
            enriched_parts.append(pref_context)

        # 2. 获取相关历史
        episodic_results = self.search_episodic(query, top_k=3)
        if episodic_results:
            history_context = "相关历史: " + "; ".join(
                r.text for r in episodic_results
            )
            enriched_parts.append(history_context)

        return " | ".join(enriched_parts)
```

---

## 13.6 Tool Calling 机制

### 13.6.1 概念定义

**Tool Calling**（工具调用，也称为 Function Calling）是 LLM 原生的结构化输出能力：模型不是生成自由文本，而是生成结构化的函数调用请求（JSON Schema），由外部系统执行后返回结果。

### 13.6.2 RAG 的 Tool 定义

```python
# ===== RAG 相关 Tool 定义 (OpenAI/Claude Function Calling 格式) =====

RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "在知识库中搜索相关信息。用于回答需要领域知识的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，应该是清晰的自然语言问题"
                    },
                    "knowledge_base": {
                        "type": "string",
                        "enum": ["technical", "legal", "financial", "general"],
                        "description": "要搜索的知识库名称",
                        "default": "general"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "filter": {
                        "type": "object",
                        "description": "元数据过滤条件，如 {date: '2024', author: '张三'}",
                        "additionalProperties": True
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_knowledge_bases",
            "description": "列出所有可用的知识库及其描述",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_detail",
            "description": "获取某个搜索结果对应文档的详细信息，包括标题、作者、日期、全文预览",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "文档唯一标识符"
                    }
                },
                "required": ["document_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_citation",
            "description": "验证引用的声明是否与源文档一致。用于防止幻觉。",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "需要验证的声明"
                    },
                    "source_document_id": {
                        "type": "string",
                        "description": "来源文档 ID"
                    },
                    "source_text": {
                        "type": "string",
                        "description": "引用源文本"
                    }
                },
                "required": ["claim", "source_document_id", "source_text"]
            }
        }
    }
]

# ===== Tool 执行器 =====
class RAGToolExecutor:
    """RAG 工具执行器 — 将 Tool Call 映射到实际 RAG 操作"""

    def __init__(self, retriever_registry: Dict[str, Any], document_store):
        self.retrievers = retriever_registry
        self.doc_store = document_store

    def execute(self, tool_name: str, arguments: Dict) -> str:
        """执行工具调用并返回结果"""

        if tool_name == "rag_search":
            kb = arguments.get("knowledge_base", "general")
            retriever = self.retrievers.get(kb, self.retrievers["general"])

            results = retriever.search(
                query=arguments["query"],
                top_k=arguments.get("top_k", 5),
                filter=arguments.get("filter")
            )
            return self._format_search_results(results)

        elif tool_name == "list_knowledge_bases":
            kbs = []
            for name, retriever in self.retrievers.items():
                kbs.append({
                    "name": name,
                    "description": retriever.description,
                    "document_count": retriever.count(),
                    "last_updated": retriever.last_updated
                })
            return json.dumps(kbs, ensure_ascii=False)

        elif tool_name == "get_document_detail":
            doc = self.doc_store.get(arguments["document_id"])
            if doc:
                return json.dumps({
                    "id": doc.id,
                    "title": doc.title,
                    "author": doc.author,
                    "date": doc.date,
                    "preview": doc.content[:500],
                    "source": doc.source
                }, ensure_ascii=False)
            return f"Error: Document {arguments['document_id']} not found"

        elif tool_name == "verify_citation":
            doc = self.doc_store.get(arguments["source_document_id"])
            if doc and arguments["source_text"] in doc.content:
                return json.dumps({
                    "verified": True,
                    "match_type": "exact" if arguments["claim"] in doc.content else "semantic",
                    "context_window": doc.content[
                        max(0, doc.content.index(arguments["source_text"]) - 100):
                        doc.content.index(arguments["source_text"]) + len(arguments["source_text"]) + 100
                    ]
                }, ensure_ascii=False)
            return json.dumps({"verified": False, "reason": "Source text not found in document"})

        return f"Error: Unknown tool '{tool_name}'"

    def _format_search_results(self, results) -> str:
        """格式化检索结果"""
        formatted = []
        for i, r in enumerate(results):
            formatted.append(
                f"[{i+1}] (Score: {r.score:.3f} | Doc: {r.metadata.get('title', 'Unknown')})\n"
                f"{r.content[:300]}...\n"
            )
        return "\n".join(formatted)
```

### 13.6.3 Tool Calling 数据流转

```
User: "2024年新能源补贴政策对特斯拉有什么影响？"

Step 1: LLM → Tool Call
  {
    "name": "list_knowledge_bases",
    "arguments": {}
  }

Step 2: System → LLM
  [{"name": "policy_library", "description": "政策法规库", "document_count": 5000},
   {"name": "auto_industry", "description": "汽车行业库", "document_count": 12000}]

Step 3: LLM → Tool Call (parallel)
  [
    {"name": "rag_search", "arguments": {"query": "2024年 新能源汽车 补贴政策 最新", "knowledge_base": "policy_library"}},
    {"name": "rag_search", "arguments": {"query": "特斯拉 补贴 政策 影响 2024", "knowledge_base": "auto_industry"}}
  ]

Step 4: System → LLM
  [政策库3条结果..., 行业库3条结果...]

Step 5: LLM → Final Answer
  "根据2024年新能源补贴政策..."
```

---

## 13.7 多知识库路由

### 13.7.1 概念定义

**多知识库路由**（KB Router）是根据用户查询的意图和内容，自动选择最合适的知识库（或多个知识库）进行检索的机制。

### 13.7.2 架构设计

```
                     ┌────────────────────┐
                     │   Query Classifier  │
User Query ────────▶│   (意图识别+分类)      │
                     └────────┬───────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │Technical │ │  Legal   │ │ Financial│
         │    KB    │ │    KB    │ │    KB    │
         └────┬─────┘ └────┬─────┘ └────┬─────┘
              │             │             │
              └─────────────┼─────────────┘
                            │
                     ┌──────▼──────┐
                     │   Result    │
                     │   Merger    │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │   Reranker  │
                     └──────┬──────┘
                            │
                     Final Context
```

### 13.7.3 核心实现

```python
from enum import Enum
from typing import List, Tuple

class KBType(Enum):
    TECHNICAL = "technical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    HR = "hr"
    PRODUCT = "product"
    GENERAL = "general"

class KBRouter:
    """多知识库智能路由器"""

    # 规则定义（可作为快速路径，也可以作为 LLM Router 的 fallback）
    DOMAIN_KEYWORDS = {
        KBType.TECHNICAL: ["代码", "API", "bug", "架构", "数据库", "部署", "Python", "Java",
                           "微服务", "K8s", "Docker"],
        KBType.LEGAL: ["法律", "合同", "合规", "诉讼", "知识产权", "专利", "GDPR",
                       "条款", "违约"],
        KBType.FINANCIAL: ["财务", "预算", "营收", "利润", "税务", "审计", "发票",
                          "报销", "ROI"],
        KBType.HR: ["招聘", "入职", "离职", "绩效", "薪资", "福利", "培训",
                   "考勤", "年假"],
        KBType.PRODUCT: ["产品", "功能", "版本", "用户反馈", "需求", "PRD",
                        "UI", "UX", "竞品"],
    }

    ROUTER_PROMPT = """你是一个知识库路由专家。根据用户查询，判断应该使用哪个（些）知识库。

可用知识库:
{kb_descriptions}

用户查询: {query}

请分析查询的意图和内容，返回 JSON:
{{
  "primary_kb": "主要知识库名称",
  "secondary_kbs": ["次要知识库名称"],
  "confidence": 0.95,
  "reasoning": "路由理由",
  "query_decomposition": ["如果查询复杂，分解后的子查询"]
}}"""

    def __init__(self, kb_registry: Dict[str, Any], llm):
        self.kb_registry = kb_registry  # name → retriever mapping
        self.llm = llm

    def _keyword_route(self, query: str) -> Dict[str, float]:
        """基于关键词的快速路由（无需 LLM，延迟 < 1ms）"""
        scores = {}
        query_lower = query.lower()

        for kb_type, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                scores[kb_type.value] = score / len(keywords)

        return scores

    def _llm_route(self, query: str) -> Dict:
        """基于 LLM 的智能路由（延迟 ~100-500ms）"""
        kb_descriptions = "\n".join([
            f"- {name}: {kb.description}"
            for name, kb in self.kb_registry.items()
        ])

        prompt = self.ROUTER_PROMPT.format(
            kb_descriptions=kb_descriptions,
            query=query
        )

        response = self.llm.generate(prompt)
        return json.loads(extract_json(response))

    def route(self, query: str, method: str = "hybrid") -> List[Tuple[str, float]]:
        """
        路由查询到知识库，返回 [(kb_name, weight), ...]

        method:
          - "keyword": 仅关键词路由 (快速)
          - "llm": 仅 LLM 路由 (准确)
          - "hybrid": 混合路由 (推荐)
        """
        if method == "keyword":
            scores = self._keyword_route(query)
            # 归一化
            total = sum(scores.values()) or 1
            return [(kb, s/total) for kb, s in sorted(scores.items(), key=lambda x: -x[1])]

        elif method == "llm":
            result = self._llm_route(query)
            kbs = [(result['primary_kb'], 0.7)]
            for kb in result.get('secondary_kbs', []):
                kbs.append((kb, 0.3 / len(result.get('secondary_kbs', [1]))))
            return kbs

        elif method == "hybrid":
            # 先尝试关键词路由
            kw_scores = self._keyword_route(query)
            max_kw_score = max(kw_scores.values()) if kw_scores else 0

            # 如果关键词置信度高（>0.3），直接用关键词结果
            if max_kw_score > 0.3:
                total = sum(kw_scores.values())
                return [(kb, s/total) for kb, s in sorted(kw_scores.items(), key=lambda x: -x[1])]

            # 否则回退到 LLM 路由
            return self.route(query, method="llm")

    def search(self, query: str, top_k_per_kb: int = 5) -> List[Dict]:
        """跨知识库搜索 + 合并"""
        kb_weights = self.route(query)
        all_results = []

        for kb_name, weight in kb_weights:
            if kb_name in self.kb_registry:
                results = self.kb_registry[kb_name].search(query, top_k=top_k_per_kb)
                for r in results:
                    r.score *= weight  # 按知识库权重调整 score
                all_results.extend(results)

        # 去重 + 排序
        seen = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: -x.score):
            if r.content not in seen:
                seen.add(r.content)
                unique_results.append(r)

        return unique_results
```

---

## 13.8 多 Agent 协作

### 13.8.1 概念定义

**多 Agent RAG 协作**是指多个专门的 AI Agent 协同工作，共同完成复杂的 RAG 任务。每个 Agent 承担特定的角色和职责，通过消息传递或共享内存进行通信。

### 13.8.2 三种协作模式

```
模式1: Orchestrator (编排者模式)
  ┌────────────────────────────────────────┐
  │            Orchestrator                 │
  │    ┌─────────────────────────────┐     │
  │    │  Decompose + Assign + Merge  │     │
  │    └──────┬──────┬──────┬────────┘     │
  │           │      │      │               │
  │     ┌─────▼┐ ┌───▼──┐ ┌─▼─────┐       │
  │     │Legal │ │Tech  │ │Finance│       │
  │     │Agent │ │Agent │ │Agent  │       │
  │     └──────┘ └──────┘ └───────┘       │
  └────────────────────────────────────────┘
  适用: 跨领域复杂查询，需要专业分工

模式2: Debate (辩论模式)
  ┌────────────────────────────────────────┐
  │  Agent A ──┐                           │
  │            ├── Debate → Consensus      │
  │  Agent B ──┘      ↑                    │
  │  Agent C ────────┘                     │
  └────────────────────────────────────────┘
  适用: 需要多角度验证的事实性问题

模式3: Hierarchical (层级模式)
  ┌────────────────────────────────────────┐
  │           Supervisor                    │
  │    ┌─────────────────────────┐         │
  │    │  子任务分配 + 质量审核     │         │
  │    └──┬────────┬────────┬───┘         │
  │       │        │        │              │
  │  ┌────▼───┐┌──▼───┐┌───▼────┐        │
  │  │Planner ││Worker││Critic  │        │
  │  └────────┘└──────┘└────────┘        │
  └────────────────────────────────────────┘
  适用: 企业级复杂流水线
```

### 13.8.3 Orchestrator 模式实现

```python
from dataclasses import dataclass, field
from typing import List, Dict, Callable

@dataclass
class AgentMessage:
    """Agent 间消息"""
    sender: str
    receiver: str
    type: str  # "task", "result", "query", "feedback"
    content: str
    metadata: Dict = field(default_factory=dict)

class OrchestratorAgent:
    """编排者 Agent — 协调多个子 Agent 完成 RAG 任务"""

    DECOMPOSE_PROMPT = """你是一个任务分解专家。请将以下复杂问题分解为多个子任务。

问题: {question}

可用子 Agent:
- legal_agent: 法律知识检索和分析
- technical_agent: 技术文档和代码检索
- financial_agent: 财务数据和报告检索
- general_agent: 通用知识检索

请输出 JSON 格式的任务分解:
{{
  "subtasks": [
    {{
      "id": "subtask_1",
      "description": "子任务描述",
      "agent": "负责的 agent 名称",
      "query": "该 agent 的检索查询",
      "depends_on": []
    }}
  ],
  "merge_strategy": "如何合并结果的说明"
}}"""

    SYNTHESIS_PROMPT = """基于以下子 Agent 的结果，综合回答原始问题。

原始问题: {question}

子任务结果:
{subtask_results}

请综合所有结果，给出完整、一致、可追溯的答案。"""

    def __init__(self, agents: Dict[str, Any], llm):
        self.agents = agents  # name → agent instance
        self.llm = llm
        self.message_queue: List[AgentMessage] = []

    def _decompose(self, question: str) -> List[Dict]:
        """分解问题为子任务"""
        prompt = self.DECOMPOSE_PROMPT.format(question=question)
        response = self.llm.generate(prompt)
        return json.loads(extract_json(response))['subtasks']

    def _dispatch(self, subtask: Dict) -> str:
        """分发子任务到对应 Agent"""
        agent_name = subtask['agent']
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            return agent.execute(subtask['query'])
        else:
            # 回退到通用 Agent
            return self.agents.get('general_agent', lambda q: f"No agent for: {q}")(subtask['query'])

    def _synthesize(self, question: str, subtask_results: List[Tuple[Dict, str]]) -> str:
        """综合所有子任务结果"""
        results_text = "\n\n".join([
            f"## {task['description']} (执行者: {task['agent']})\n{result}"
            for task, result in subtask_results
        ])

        prompt = self.SYNTHESIS_PROMPT.format(
            question=question,
            subtask_results=results_text
        )

        return self.llm.generate(prompt)

    def execute(self, question: str) -> Dict:
        """执行多 Agent 协作 RAG"""
        # 1. 任务分解
        subtasks = self._decompose(question)

        # 2. 构建依赖图并排序
        subtasks = self._topological_sort(subtasks)

        # 3. 执行子任务（尊重依赖关系）
        results = {}
        for subtask in subtasks:
            # 等待依赖完成
            deps = subtask.get('depends_on', [])
            for dep_id in deps:
                if dep_id not in results:
                    raise RuntimeError(f"Dependency {dep_id} not found for {subtask['id']}")

            # 如果有依赖，在查询中注入结果上下文
            query = subtask['query']
            if deps:
                dep_context = "; ".join([results[d] for d in deps])
                query = f"{query}\n上下文: {dep_context}"

            # 执行
            results[subtask['id']] = self._dispatch({**subtask, 'query': query})

        # 4. 综合结果
        subtask_results = [(t, results[t['id']]) for t in subtasks]
        final_answer = self._synthesize(question, subtask_results)

        return {
            "answer": final_answer,
            "subtasks": subtasks,
            "results": results
        }

    def _topological_sort(self, subtasks: List[Dict]) -> List[Dict]:
        """按依赖关系拓扑排序"""
        task_map = {t['id']: t for t in subtasks}
        visited = set()
        temp_visited = set()
        order = []

        def visit(task_id):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected: {task_id}")
            if task_id not in visited:
                temp_visited.add(task_id)
                for dep_id in task_map[task_id].get('depends_on', []):
                    visit(dep_id)
                temp_visited.remove(task_id)
                visited.add(task_id)
                order.append(task_map[task_id])

        for t in subtasks:
            if t['id'] not in visited:
                visit(t['id'])

        return order
```

---

## 13.9 企业级 Agentic RAG 架构

### 13.9.1 完整架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE AGENTIC RAG ARCHITECTURE                    │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          API Gateway (Kong/APISIX)                     │   │
│  │                    Auth | Rate Limit | Routing | Logging               │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                            │
│  ┌──────────────────────────────▼───────────────────────────────────────┐   │
│  │                        Agent Orchestrator                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Session    │  │  Strategy   │  │  Execution  │  │  Result    │  │   │
│  │  │  Manager    │  │  Selector   │  │  Engine     │  │  Validator │  │   │
│  │  │             │  │             │  │             │  │            │  │   │
│  │  │ - User      │  │ - ReAct     │  │ - Parallel  │  │ - Factual  │  │   │
│  │  │   Context   │  │ - Plan&Exec │  │ - Sequential│  │ - Complete │  │   │
│  │  │ - Memory    │  │ - Reflection│  │ - Debate    │  │ - No Hallu │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                            │
│  ┌──────────────────────────────▼───────────────────────────────────────┐   │
│  │                         Tool Registry                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ RAG      │ │ SQL      │ │ Web      │ │ Code     │ │ API      │  │   │
│  │  │ Search   │ │ Query    │ │ Search   │ │ Exec     │ │ Call     │  │   │
│  │  │          │ │          │ │          │ │ (sandbox)│ │          │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  └───────┼────────────┼────────────┼────────────┼────────────┼────────┘   │
│          │            │            │            │            │              │
│  ┌───────▼────────────▼────────────▼────────────▼────────────▼────────┐   │
│  │                       Infrastructure Layer                           │   │
│  │                                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ Vector   │  │ Postgres │  │ Redis    │  │ Message Queue    │   │   │
│  │  │ DB       │  │ (Metadata│  │ (Cache + │  │ (Kafka/NATS)     │   │   │
│  │  │ (Milvus) │  │ + State) │  │  Memory) │  │                  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ Embedding│  │ Rerank   │  │ LLM      │  │ Object Storage   │   │   │
│  │  │ Service  │  │ Service  │  │ Service  │  │ (MinIO/S3)       │   │   │
│  │  │ (GPU)    │  │ (GPU)    │  │ (vLLM)   │  │                  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Observability Layer                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ Tracing  │  │ Metrics  │  │ Logging  │  │ Alerting         │   │   │
│  │  │ (Jaeger) │  │(Prometheus│  │ (ELK)    │  │ (AlertManager)   │   │   │
│  │  │          │  │+Grafana) │  │          │  │                  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 13.9.2 关键组件设计

```python
class EnterpriseAgenticRAG:
    """企业级 Agentic RAG 系统"""

    def __init__(self, config: Dict):
        # 策略选择器
        self.strategy_selector = StrategySelector(config)

        # Agent 注册表
        self.agents = {
            "react": ReActAgent(config.react_tools, config.llm),
            "plan_execute": PlanAndExecuteAgent(config.tools, config.llm),
            "reflection": ReflectionAgent(config.retriever, config.llm),
        }

        # 工具注册表
        self.tool_registry = ToolRegistry()
        self.tool_registry.register("rag_search", RAGSearchTool(config.retrievers))
        self.tool_registry.register("sql_query", SQLQueryTool(config.db))
        self.tool_registry.register("web_search", WebSearchTool(config.search_api_key))
        self.tool_registry.register("calculator", CalculatorTool())

    async def process(self, request: AgentRequest) -> AgentResponse:
        """处理 Agentic RAG 请求"""
        # 1. 查询分类 → 选择策略
        strategy = self.strategy_selector.select(request.query)

        # 2. 获取 Agent 实例
        agent = self.agents[strategy]

        # 3. 注入用户上下文/记忆
        user_context = await self.load_user_context(request.user_id)
        enriched_query = self.enrich_query(request.query, user_context)

        # 4. 执行 Agent 循环（带超时和断路保护）
        with self.circuit_breaker(), self.timeout(30):
            result = await agent.run(enriched_query)

        # 5. 验证结果
        validation = await self.validator.validate(result)

        # 6. 保存会话记忆
        await self.save_interaction(request.user_id, request.query, result)

        # 7. 记录审计日志
        await self.audit_log.record(request, strategy, result, validation)

        return AgentResponse(
            answer=result.answer,
            citations=result.citations,
            trace=result.trace,
            confidence=validation.confidence
        )
```

### 13.9.3 策略选择器

```python
class StrategySelector:
    """根据查询复杂度自动选择 Agent 策略"""

    COMPLEXITY_PROMPT = """分析以下查询的复杂度，返回 JSON:
{{
  "complexity": "simple|medium|complex",
  "requires_multi_step": true/false,
  "requires_cross_kb": true/false,
  "requires_verification": true/false
}}

查询: {query}"""

    STRATEGY_MAP = {
        # (complexity, multi_step, cross_kb, verification) → strategy
        ("simple", False, False, False): "direct_rag",    # 直接 RAG，无 Agent
        ("medium", True, False, False): "react",          # ReAct 单步推理
        ("medium", True, True, False): "plan_execute",    # Plan&Execute 跨 KB
        ("complex", True, True, True): "reflection",      # Reflection 带验证
    }

    def __init__(self, llm):
        self.llm = llm
        # 缓存：相同查询的复杂度分析结果
        self.cache = {}

    def select(self, query: str) -> str:
        # 规则快速路径
        word_count = len(query)
        if word_count < 10 and "?" not in query:
            return "direct_rag"

        # 缓存检查
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        # LLM 分析
        prompt = self.COMPLEXITY_PROMPT.format(query=query)
        response = self.llm.generate(prompt)
        analysis = json.loads(extract_json(response))

        # 匹配策略
        key = (
            analysis['complexity'],
            analysis['requires_multi_step'],
            analysis['requires_cross_kb'],
            analysis['requires_verification']
        )

        strategy = self.STRATEGY_MAP.get(key, "react")
        self.cache[cache_key] = strategy
        return strategy
```

---

## 13.10 高级 Agentic RAG 模式

### 13.10.1 Self-RAG

**Self-RAG** (Self-Reflective RAG, Asai et al. 2023) 是让 LLM 在生成过程中自主决定是否需要检索，并对检索结果和生成内容进行自我批判。

```
Self-RAG 流程:

  Generate → [需要检索?] ──Yes──→ Retrieve → [相关?] ──Yes──→ Generate
      │                              │                              │
      └────────── No ────────────────┘              No ──→ Skip     │
                                                                     │
  Final Answer ←── [支持?] ──Yes──← Verify ←────────────────────────┘
                      │
                      No ──→ Regenerate / Re-retrieve
```

**核心实现要点**：
- 使用特殊的 Reflection Token（`<RETRIEVE>`, `<ISREL>`, `<ISSUP>`, `<ISUSE>`）
- 训练或提示 LLM 在生成过程中插入这些 token 进行自我评估
- 支持度（Support）和有用度（Usefulness）双重评判

### 13.10.2 Corrective RAG (CRAG)

**Corrective RAG** 在检索后增加一个评估步骤：如果检索质量不够，自动触发 Web Search 作为补充。

```python
class CRAGAgent:
    """Corrective RAG — 检索质量纠正"""

    def run(self, query: str) -> str:
        # 1. 标准 RAG 检索
        docs = self.retriever.search(query, top_k=10)

        # 2. 检索质量评估
        eval_prompt = f"""评估以下检索结果对查询 "{query}" 的相关性。
对每个结果评分 (0-1)。如果平均分 < 0.5，标记为需要纠正。

检索结果:
{format_docs(docs)}

返回 JSON:
{{"scores": [0.8, 0.3, ...], "avg_score": 0.55, "needs_correction": false}}"""

        evaluation = json.loads(extract_json(self.llm.generate(eval_prompt)))

        # 3. 如果质量不足，触发知识精炼 + Web 补充
        if evaluation['needs_correction']:
            # 3a. 从文档中提取关键知识
            knowledge = self._extract_knowledge(docs, query)
            # 3b. 使用提取的知识改进查询
            refined_query = self._refine_query(query, knowledge)
            # 3c. Web Search 补充
            web_docs = self.web_search(refined_query)
            docs = self._merge_and_dedup(docs, web_docs)

        # 4. 生成最终答案
        return self._generate(query, docs)
```

### 13.10.3 Adaptive RAG

**Adaptive RAG** 根据查询类型动态选择最合适的检索策略。

```python
class AdaptiveRAG:
    """自适应 RAG — 动态策略选择"""

    STRATEGIES = {
        "factoid": {  # 事实查询
            "retrieval": "dense",     # 稠密向量检索
            "top_k": 3,
            "rerank": True,
        },
        "analytical": {  # 分析查询
            "retrieval": "hybrid",    # 混合检索
            "top_k": 10,
            "rerank": True,
            "decompose": True,        # 需要问题分解
        },
        "comparative": {  # 对比查询
            "retrieval": "multi_query",  # 多查询变体
            "top_k": 5,
            "rerank": True,
            "parallel": True,         # 并行检索
        },
        "procedural": {  # 步骤查询
            "retrieval": "hierarchical",  # 层级检索
            "top_k": 5,
            "chunk_strategy": "small",  # 小 chunk
        },
    }

    def select_and_execute(self, query: str):
        query_type = self._classify_query(query)
        strategy = self.STRATEGIES.get(query_type, self.STRATEGIES["factoid"])
        return self._execute_with_strategy(query, strategy)
```

### 13.10.4 ITER-RETGEN

**ITER-RETGEN** (Iterative Retrieval-Generation) 在检索和生成之间建立迭代循环，每次生成后检查是否有信息缺口，自动补充检索。

```python
class IterRetGen:
    """迭代检索生成"""

    def run(self, query: str, max_iterations: int = 3) -> str:
        all_docs = []
        previous_answer = ""

        for i in range(max_iterations):
            # 检索
            docs = self.retriever.search(query, top_k=5)
            docs = self._filter_duplicates(docs, all_docs)
            all_docs.extend(docs)

            # 生成
            answer = self._generate(query, all_docs)

            # 检查信息缺口
            gap_check = self._check_gaps(query, answer)
            if not gap_check['has_gaps']:
                return answer

            # 用缺口构造新查询继续
            query = gap_check['gap_query']
            previous_answer = answer

        return answer  # 达到最大迭代
```

---

## 13.11 面试高频问题

### Q1: ReAct 模式中，如何处理 LLM 输出解析失败的问题？

**标准答案**：
1. **结构化输出约束**：使用 Function Calling 替代文本解析，LLM 原生返回 JSON
2. **正则修复**：对常见格式错误（多余空格、换行）做预处理
3. **重试机制**：解析失败后，在 prompt 中提示格式错误并要求重新输出
4. **结构化输出 API**：使用 OpenAI 的 `response_format={"type": "json_object"}` 或 Claude 的 Tool Use

**追问**：如果 LLM 选择了不存在的工具怎么办？
- 在 Observation 中返回明确的错误信息 "Tool 'XXX' not found. Available tools: [list]"
- Agent 会自动根据错误信息调整下一步动作（这是 ReAct 的核心优势）

### Q2: Agentic RAG 比传统 RAG 慢多少？如何平衡质量和延迟？

**标准答案**：
- 延迟对比：传统 RAG p50 ~200ms, Agentic RAG p50 ~800ms-2s
- 核心策略：**查询分类 + 策略路由**，简单查询走快速通道（直接 RAG），复杂查询走 Agent
- 实现方式：95% 的查询可通过规则判断（查询长度、关键词、历史模式），仅 5% 需要 LLM 分类

### Q3: 如何防止 Agent 陷入无限循环？

**标准答案**：
1. **硬限制**：设置 `max_steps` (ReAct 推荐 10 步以内)
2. **重复检测**：如果连续 2 步 Action 相同但 Observation 不变，强制终止
3. **Token 预算**：累计 token 超过阈值后强制输出
4. **置信度阈值**：Final Answer 附带置信度，低于阈值时触发 Reflection 而非循环
5. **监管者 Agent**：独立的监督 Agent 监控执行循环并可在异常时中断

### Q4: 多知识库路由中，如何处理查询需要跨多个知识库的情况？

**标准答案**：
1. **查询分解**：将复杂查询分解为子查询，每个对应特定知识库
2. **并行检索**：同时对多个知识库发起检索
3. **RRF 合并**：使用 RRF 融合多知识库结果
4. **权重调整**：根据置信度给予不同知识库不同权重
5. **一致性检查**：跨知识库结果可能存在矛盾，需要后处理验证

### Q5: Agentic RAG 与传统 RAG 的 Token 成本对比？

| 场景 | 传统 RAG | Agentic RAG | 倍数 |
|------|---------|-------------|------|
| 简单事实查询 | ~500 tokens | ~600 tokens | 1.2x |
| 中等复杂度 | ~800 tokens | ~2,000 tokens | 2.5x |
| 高复杂度 | ~1,200 tokens | ~4,500 tokens | 3.8x |

**优化建议**：
- 对简单查询跳过 Agent，直接 RAG
- 使用更小更快的模型进行策略选择
- Prompt 压缩（摘要历史而非完整历史）
- 结果缓存（相同查询直接返回）

### Q6: Reflection 模式中，如何避免 LLM 对自己的输出"过于宽容"？

**标准答案**：
1. **角色分离**：使用不同的 LLM 实例或不同的 System Prompt 进行反思
2. **结构化评分**：要求输出 1-5 分的具体评分而非是/否
3. **要求具体证据**：每个扣分项必须指出具体的文本证据
4. **交叉验证**：两个独立的 Critic Agent 交叉评分
5. **外部验证器**：使用 NLI 模型做事实一致性检查

---

## 13.12 企业级最佳实践

### 13.12.1 架构层面

1. **渐进式采用**：从传统 RAG 开始 → 加入简单路由 → 引入 ReAct → 最后才用完整 Agent 体系
2. **95/5 规则**：95% 的查询用简单管道，5% 的复杂查询才路由到 Agent
3. **Agent 只做决策，不做执行**：决策层用强模型（Opus/GPT-4），执行层用弱模型（Haiku/GPT-4o-mini）
4. **工具最小化原则**：每个 Agent 只暴露完成其任务所必须的工具

### 13.12.2 可靠性层面

5. **多层防护**：Agent 层 + 工具层 + 基础设施层三个级别的超时和熔断
6. **结果验证**：任何 Agent 输出必须经过事实性验证（citation check, NLI）
7. **人工确认关键操作**：涉及数据修改的操作需要 HITL (Human-in-the-Loop)
8. **灰度发布**：新 Agent 策略用 1% → 10% → 100% 的流量灰度

### 13.12.3 成本控制层面

9. **三层模型架构**：
   - 路由/分类 → 小模型 (Haiku/4o-mini)
   - 常规 Agent 推理 → 中模型 (Sonnet/4o)
   - 复杂推理/验证 → 大模型 (Opus/4o)
10. **Token 预算管理**：每个 Agent 会话设定 token 上限，超限降级为简单模式
11. **缓存策略**：查询分类结果、检索结果、常见问题的 Agent 轨迹全部缓存

### 13.12.4 可观测性层面

12. **全链路追踪**：每次 Agent 调用的 Thought → Action → Observation 链路完整记录
13. **Agent 专属指标体系**：
    - 策略选择准确率
    - 工具调用成功率
    - 反思触发率（过低可能过于自信，过高可能过于保守）
    - 用户干预率
14. **成本归因**：每个查询的 Token 消耗精确归因到每个 Agent 步骤

### 13.12.5 安全层面

15. **工具调用沙箱**：代码执行、SQL 查询等危险操作必须在沙箱中执行
16. **PII 保护**：Agent 的记忆系统必须经过 PII 脱敏
17. **Prompt 注入防护**：用户输入在进入 Agent 之前必须经过注入检测

### 13.12.6 检查清单

```
☐ 查询分类器是否在 95% 以上的简单查询中跳过了 Agent？
☐ Agent 的最大步数是否设置（建议 ≤10）？
☐ 是否有重复动作检测和循环中断机制？
☐ 工具的可用性是否有熔断保护？
☐ Agent 的每次关键决策是否有审计日志？
☐ 是否有独立的验证步骤检查 Agent 输出的事实准确性？
☐ 多 Agent 协作时是否处理了死锁和资源竞争？
☐ Token 预算是否与查询复杂度匹配？
☐ 是否对用户展示了 Agent 的思考过程以建立信任？
☐ 是否有 HITL 机制处理低置信度输出？
```

---

## 13.13 章节总结

**Agentic RAG 不是传统 RAG 的替代，而是升级和增强。**

| 维度 | 传统 RAG | Agentic RAG |
|------|---------|-------------|
| 检索策略 | 固定 | 动态自适应 |
| 推理能力 | 单步 | 多步迭代 |
| 工具使用 | 仅检索 | 检索+计算+API+代码 |
| 错误恢复 | 无 | 反思+重试 |
| 知识库 | 单一 | 多库路由 |
| 可解释性 | 低 | 高（完整思考链） |
| 成本 | 低 | 中等（需优化） |
| 适用场景 | 简单问答 | 复杂分析+决策 |

**核心技术演进路径**：
```
RAG → ReAct RAG → Plan&Execute RAG → Reflection RAG
  → Multi-Agent RAG → Hierarchical Swarm RAG
```

**下一章预告**：第十四章将把这些技术落地为百万/千万文档级的企业级 RAG 系统架构设计。

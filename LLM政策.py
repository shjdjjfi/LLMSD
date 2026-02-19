# -*- coding: utf-8 -*-
"""
政策后果 DAG + 固定特征层 + DeepSeek 思考模式 + 过程可视化日志

功能：
- 中间层：由 DeepSeek 生成的“定性后果节点”（支持多父节点）
- 底层：固定特征节点（与数据集特征一一对应）
- 自动：
  1) 按层生成后果 DAG（每个子节点选择继承哪些父节点）
  2) 将中间节点映射到特征节点（feature linkage）
  3) 打印按层进度、特征映射进度、图结构概要

使用前：
- 安装依赖：pip install openai
- 把下面 DEEPSEEK_API_KEY 替换成你的 DeepSeek API key
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from openai import OpenAI

# ========================= 0. DeepSeek 客户端 =========================

# ⚠️ 在这里填你的 DeepSeek Key（字符串）
DEEPSEEK_API_KEY = "待填"


def make_deepseek_client() -> OpenAI:
    """
    构造一个指向 DeepSeek API 的 OpenAI 兼容客户端。
    不依赖环境变量，直接使用上面的 DEEPSEEK_API_KEY。
    """
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",  # DeepSeek 官方 base_url
    )


# ========================= 1. 基础数据结构 =========================


@dataclass
class Node:
    """
    后果图中的一个节点:
    - 中间后果节点: node_type="intermediate"
    - 特征节点:     node_type="feature"
    """
    id: str
    text: str              # 自然语言命题
    depth: int             # 在图中的深度 (根=0, 中间层=1,2,..., 特征层可统一特殊值)

    node_type: str = "intermediate"  # "intermediate" 或 "feature"

    # 一些离散标签 (纯定性, 不做数值计算)
    domain: str = "unspecified"      # fiscal / healthcare / labor / inequality / indicator ...
    timescale: str = "unspecified"   # short_term / mid_term / long_term / unspecified
    polarity: str = "unspecified"    # positive / negative / mixed / uncertain / unspecified
    confidence: str = "unspecified"  # high / medium / low / unspecified
    source: str = "model"            # root / model / human / feature

    parents: List[str] = field(default_factory=list)  # 父节点 id 列表 (支持多父, 可为 DAG)


@dataclass
class Edge:
    """父子节点之间的定性关系"""
    parent_id: str
    child_id: str
    relation_type: str = "causal"  # causal / reinforcing / inhibiting / conditional / feature_increase ...


@dataclass
class ConsequenceTree:
    """整棵后果图: 节点 + 边"""
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def get_children(self, node_id: str) -> List[Node]:
        child_ids = [e.child_id for e in self.edges if e.parent_id == node_id]
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parents(self, node_id: str) -> List[Node]:
        parent_ids = [e.parent_id for e in self.edges if e.child_id == node_id]
        return [self.nodes[pid] for pid in parent_ids if pid in self.nodes]


# ========================= 2. LLM：按层生成中间后果（多父） =========================

LAYER_SYSTEM_PROMPT = """
你是一个公共政策分析助手，负责以“定性”的方式推导政策的后果。
你只输出 JSON，不要输出多余文字。

本轮输入:
- 一个政策决策及其背景信息
- 上一层若干个后果节点 (父节点列表), 每个有一个整数索引 0..N-1

你需要输出:
- 下一层若干个“子节点” (后果节点), 每个子节点可以由一个或多个父节点共同导致

要求：
1. 每个子节点包含以下字段:
   - text: 自然语言描述, 句子要具体、清晰, 避免空泛。
   - domain: 所属领域, 如 fiscal, healthcare, labor, inequality, social_stability, pension, environment 等。
   - timescale: 发生时间尺度, short_term / mid_term / long_term。
   - polarity: 对社会整体或相关群体的影响方向, positive / negative / mixed / uncertain。
   - confidence: 经验或理论上的可信度, high / medium / low。
   - relation_type: 与其所有父节点的关系类型, 取值如 causal / reinforcing / inhibiting / conditional。
   - parent_indices: 一个非空整数列表, 表示该子节点直接依赖的父节点索引,
                     每个索引必须在 [0, N-1] 范围内, 不重复。
     例如: [0], [1,2], [0,1,2] 等。

2. 对于 parent_indices:
   - 至少包含一个父节点 (长度 >= 1)
   - 最多可以包含上一层的全部父节点 (长度 <= N)
   - 不能引用不存在的索引, 不能出现负数, 不能重复。

3. 只进行定性推理, 不要出现任何具体数值 (比如百分比、精确金额、具体金额)。

4. 后果节点必须是现实主义的、在学界或政策实践中常被提及的逻辑结果, 避免天马行空的猜测。

5. 注意控制发散:
   - 不要生成太多高度相似的子节点
   - 子节点总数不能超过 max_children_total (由用户输入)。

输出格式:
- 只输出一个 JSON 对象, 形如:
{
  "children": [
    {
      "text": "...",
      "domain": "healthcare",
      "timescale": "mid_term",
      "polarity": "negative",
      "confidence": "high",
      "relation_type": "causal",
      "parent_indices": [0, 2]
    },
    ...
  ]
}
"""

LAYER_USER_TEMPLATE = """
[决策描述]
{decision_description}

[背景信息]
{context}

[上一层后果节点列表]
{parents_block}

现在, 请你基于以上父节点, 推导出下一层的若干后果节点 (children)。
要求:
- 子节点总数不超过 {max_children_total} 个。
- 每个子节点的 parent_indices 至少包含 1 个索引, 最多可以包含上方所有父节点索引。
- 请严格使用 JSON 格式输出, 字段为:
  - children: 列表
    - 每个元素包含:
      - text: string
      - domain: string
      - timescale: string (short_term / mid_term / long_term)
      - polarity: string (positive / negative / mixed / uncertain)
      - confidence: string (high / medium / low)
      - relation_type: string (causal / reinforcing / inhibiting / conditional)
      - parent_indices: 非空整数列表, 所有元素在 [0, {max_parent_index}] 之间, 不重复。
"""


class LLMConsequenceGenerator:
    """
    按“层”生成下一层中间后果节点的 LLM 调用器 (DeepSeek):
    - 输入: 决策描述 + 背景 + 上一层 parent_nodes 列表
    - 输出: children 规格列表, 每个规格包含 text/domain/.../parent_indices
    """

    def __init__(
        self,
        model_name: str = "deepseek-reasoner",  # 使用思考模式
        temperature: float = 0.7,
    ):
        self.client = make_deepseek_client()
        self.model_name = model_name
        self.temperature = temperature

    def _build_parents_block(self, parent_nodes: List[Node]) -> str:
        """
        把上一层节点拼成带索引的文本块, 给 LLM 参考。
        例如:
        0. [domain/timescale/polarity] 文本...
        1. ...
        """
        lines = []
        for idx, node in enumerate(parent_nodes):
            tag = f"[{node.domain}/{node.timescale}/{node.polarity}]"
            text = node.text.replace("\n", " ")
            lines.append(f"{idx}. {tag} {text}")
        return "\n".join(lines)

    def generate_layer(
        self,
        decision_description: str,
        context: str,
        parent_nodes: List[Node],
        max_children_total: int,
    ) -> List[Dict]:
        """
        调用 DeepSeek, 对上一层所有 parent_nodes 一次性生成下一层子节点列表.
        返回: children_specs, 每个是 dict:
          {
            "text": ...,
            "domain": ...,
            "timescale": ...,
            "polarity": ...,
            "confidence": ...,
            "relation_type": ...,
            "parent_indices": [int, ...]
          }
        """
        if not parent_nodes:
            return []

        parents_block = self._build_parents_block(parent_nodes)
        max_parent_index = len(parent_nodes) - 1

        user_prompt = LAYER_USER_TEMPLATE.format(
            decision_description=decision_description.strip(),
            context=context.strip(),
            parents_block=parents_block,
            max_children_total=max_children_total,
            max_parent_index=max_parent_index,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LAYER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print("JSON 解析失败 (layer generation), 原始返回：", content)
            return []

        children = data.get("children", [])
        if not isinstance(children, list):
            return []

        # 截断到 max_children_total
        return children[:max_children_total]


# ========================= 3. 构图器：按层递归生成 DAG + 过程日志 =========================

class ConsequenceTreeBuilder:
    """
    使用 LLMConsequenceGenerator, 以“按层”的方式, 从根决策出发,
    递归生成一个有限深度 / 有限宽度的中间后果图 (DAG):
    - 每一层子节点可以有 1~N 个父节点 (来自上一层)。
    """

    def __init__(
        self,
        generator: LLMConsequenceGenerator,
        max_depth: int = 3,
        max_branch: int = 4,
        verbose: bool = True,   # 是否打印日志
    ):
        """
        参数:
        - generator: LLMConsequenceGenerator 实例
        - max_depth: 中间层最大深度 (根为 0)
        - max_branch: 控制“平均每个父节点”生成的子节点数,
                      每一层最多生成的子节点总数约为 max_branch * len(parent_layer)
        - verbose: 是否打印按层的进度信息
        """
        self.generator = generator
        self.max_depth = max_depth
        self.max_branch = max_branch
        self.verbose = verbose

    def _new_node_id(self) -> str:
        return str(uuid.uuid4())

    def build_tree(
        self,
        decision_description: str,
        context: str = "",
        root_text: Optional[str] = None,
    ) -> ConsequenceTree:
        """
        从决策出发, 构建仅包含 node_type="intermediate" 的中间后果 DAG.
        之后你可以再往 tree 里加特征节点 & feature edges.
        """
        tree = ConsequenceTree()

        # 创建根节点 (depth=0)
        root = Node(
            id=self._new_node_id(),
            text=root_text.strip() if root_text else decision_description.strip(),
            depth=0,
            node_type="intermediate",
            domain="decision",
            timescale="short_term",
            polarity="mixed",
            confidence="high",
            source="root",
        )
        tree.add_node(root)

        current_layer: List[Node] = [root]
        current_depth = 0

        while current_layer and current_depth < self.max_depth:
            # ====== 进程可视化: 打印当前层信息 ======
            if self.verbose:
                print(f"\n[Layer {current_depth}] 父节点数 = {len(current_layer)}")
                for idx, n in enumerate(current_layer):
                    preview = n.text.replace("\n", " ")[:50]
                    print(f"  - P{idx}: {preview}...")

            # 本层最多生成的子节点总数
            max_children_total = self.max_branch * max(1, len(current_layer))
            if self.verbose:
                print(f"[Layer {current_depth}] 调用 LLM 生成下一层, max_children_total = {max_children_total}")

            children_specs = self.generator.generate_layer(
                decision_description=decision_description,
                context=context,
                parent_nodes=current_layer,
                max_children_total=max_children_total,
            )

            if not children_specs:
                if self.verbose:
                    print(f"[Layer {current_depth}] 未生成任何子节点, 提前停止。")
                break

            if self.verbose:
                print(f"[Layer {current_depth}] 实际生成子节点规格数 = {len(children_specs)}")

            next_layer: List[Node] = []
            next_depth = current_depth + 1

            if self.verbose:
                print(f"[Layer {current_depth} -> {next_depth}] 开始落地子节点到图中...")

            kept_count = 0  # 统计会被继续扩展的节点数

            for spec in children_specs:
                text = (spec.get("text") or "").strip()
                if not text:
                    continue

                # 解析 parent_indices, 映射成 parent_ids
                raw_indices = spec.get("parent_indices", [])
                if not isinstance(raw_indices, list):
                    continue

                # 去重 + 过滤非法索引
                valid_indices = sorted(
                    {i for i in raw_indices if isinstance(i, int) and 0 <= i < len(current_layer)}
                )
                if not valid_indices:
                    # 至少需要一个合法父节点
                    continue

                parent_ids = [current_layer[i].id for i in valid_indices]

                node = Node(
                    id=self._new_node_id(),
                    text=text,
                    depth=next_depth,
                    node_type="intermediate",
                    domain=spec.get("domain", "unspecified"),
                    timescale=spec.get("timescale", "unspecified"),
                    polarity=spec.get("polarity", "unspecified"),
                    confidence=spec.get("confidence", "unspecified"),
                    source="model",
                    parents=parent_ids,
                )
                tree.add_node(node)

                # 为每个父节点创建边
                relation_type = spec.get("relation_type", "causal")
                for pid in parent_ids:
                    edge = Edge(
                        parent_id=pid,
                        child_id=node.id,
                        relation_type=relation_type,
                    )
                    tree.add_edge(edge)

                # 简单定性剪枝: 只对 high/medium 可信度的节点继续向下扩展
                if next_depth < self.max_depth and node.confidence in ("high", "medium"):
                    next_layer.append(node)
                    kept_count += 1

            if self.verbose:
                print(f"[Layer {current_depth} -> {next_depth}] 新建节点数 = {len(children_specs)}, 其中 {kept_count} 个将继续扩展。")

            # 进入下一层
            current_layer = next_layer
            current_depth = next_depth

        return tree


# ========================= 4. 特征层 & FeatureLinker =========================

@dataclass
class Feature:
    """与数据集中某一列一一对应的“社会指标”描述 (定性层)"""
    id: str           # 对应数据集中的列名
    label: str        # 人类可读的名字
    description: str  # 简短解释, 会给 LLM 看


# 示例特征; 实际使用时你按自己的数据集补全
FEATURES: List[Feature] = [
    Feature(
        id="healthcare_spending_share",
        label="医保支出占 GDP 比例",
        description="政府在医疗保健上的总支出占国内生产总值的比例。"
    ),
    Feature(
        id="public_debt_gdp_ratio",
        label="政府债务占 GDP 比例",
        description="中央及地方政府债务余额占国内生产总值的综合比例。"
    ),
    Feature(
        id="elderly_poverty_rate",
        label="老年人贫困率",
        description="65 岁及以上人口中生活在贫困线以下的比例。"
    ),
    Feature(
        id="gini_index",
        label="基尼系数",
        description="居民收入分配不平等程度的指标。"
    ),
]


def add_feature_nodes_to_tree(tree: ConsequenceTree) -> Dict[str, str]:
    """
    把 FEATURES 中的每个 Feature 加入树, 作为 node_type="feature" 的节点。
    返回 feature_id -> node_id 的映射。
    """
    feature_id_to_node_id: Dict[str, str] = {}
    for feat in FEATURES:
        nid = f"feat::{feat.id}"
        node = Node(
            id=nid,
            text=f"特征: {feat.label}",
            depth=999,                 # 特征层统一给一个特殊大 depth
            node_type="feature",
            domain="indicator",
            timescale="unspecified",
            polarity="unspecified",
            confidence="high",
            source="feature",
            parents=[],
        )
        tree.add_node(node)
        feature_id_to_node_id[feat.id] = nid
    return feature_id_to_node_id


FEATURE_LINKER_SYSTEM_PROMPT = """
你是一个政策指标映射助手。
给定一个“定性后果描述”和一组社会指标，请判断该后果会直接影响哪些指标，以及影响方向。
你只输出 JSON。
"""

# 注意：JSON 示例中的 { } 必须用 {{ }} 转义，避免 .format() 把它当占位符
FEATURE_LINKER_USER_TEMPLATE = """
[后果节点]
{text}

[可用社会指标]
{features_block}

要求:
- 从这些指标中选出本后果最直接相关的 0~K 个, 不要超过 {max_links} 个。
- 对每个选中的指标, 给出:
  - feature_id: 指标的 id
  - direction: "increase" / "decrease" / "ambiguous"
- 不要凭空发明新的指标。
- 如果没有任何指标直接相关, 则返回空列表。

输出格式示例:
{{
  "links": [
    {{
      "feature_id": "healthcare_spending_share",
      "direction": "increase"
    }}
  ]
}}
"""


class FeatureLinker:
    """
    用 DeepSeek 判断: 一个中间后果节点, 直接作用于哪些特征 (FEATURES), 以及大致方向。
    """

    def __init__(self, model_name: str = "deepseek-reasoner", temperature: float = 0.2):
        self.client = make_deepseek_client()
        self.model_name = model_name
        self.temperature = temperature

    def _build_features_block(self, features: List[Feature]) -> str:
        lines = []
        for f in features:
            lines.append(
                f"- id: {f.id}\n  label: {f.label}\n  description: {f.description}"
            )
        return "\n".join(lines)

    def link_node_to_features(
        self,
        node: Node,
        features: List[Feature],
        max_links: int = 3,
    ) -> List[Dict]:
        """
        返回若干 {feature_id, direction} 字典, 至多 max_links 个。
        """
        user_prompt = FEATURE_LINKER_USER_TEMPLATE.format(
            text=node.text.strip(),
            features_block=self._build_features_block(features),
            max_links=max_links,
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FEATURE_LINKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print("Feature linkage JSON 解析失败:", content)
            return []

        links = data.get("links", [])
        if not isinstance(links, list):
            return []
        return links[:max_links]


def attach_features(
    tree: ConsequenceTree,
    feature_linker: FeatureLinker,
    feature_id_to_node_id: Dict[str, str],
    max_links_per_node: int = 3,
    verbose: bool = True,   # 这里也加可视化日志
) -> None:
    """
    对树中所有 node_type="intermediate" 的节点, 用 FeatureLinker
    判断它直接影响哪些特征, 然后添加 edge: intermediate -> feature_node。
    """
    intermediate_nodes = [
        n for n in tree.nodes.values()
        if n.node_type == "intermediate"
    ]

    total = len(intermediate_nodes)
    if verbose:
        print(f"\n[Feature Mapping] 共有 {total} 个中间节点需要做特征映射。")

    for idx, node in enumerate(intermediate_nodes, start=1):
        if verbose:
            preview = node.text.replace("\n", " ")[:40]
            print(f"[Feature Mapping] ({idx}/{total}) 节点: {preview}...")

        links = feature_linker.link_node_to_features(
            node, FEATURES, max_links=max_links_per_node
        )

        if verbose:
            print(f"  -> LLM 返回 links 数 = {len(links)}")

        for link in links:
            fid = link.get("feature_id")
            direction = link.get("direction", "ambiguous")
            if not fid or fid not in feature_id_to_node_id:
                continue
            feature_node_id = feature_id_to_node_id[fid]
            edge = Edge(
                parent_id=node.id,
                child_id=feature_node_id,
                relation_type=f"feature_{direction}",  # 如 feature_increase
            )
            tree.add_edge(edge)
            if verbose:
                print(f"     * 连接到特征 {fid} ({direction})")


# ========================= 5. 打印工具 =========================

def pretty_print_tree(tree: ConsequenceTree, root_id: str, indent: int = 0) -> None:
    """
    简单打印从某个节点出发的后果图。
    注意: 因为支持多父, 这个打印是 DFS 风格, 可能会重复打印同一节点。
    """
    node = tree.nodes[root_id]
    prefix = "  " * indent
    label = f"[{node.node_type}/{node.domain}/{node.timescale}/{node.polarity}/{node.confidence}]"
    print(f"{prefix}- {node.text} {label}")

    children = tree.get_children(root_id)
    for child in children:
        pretty_print_tree(tree, child.id, indent + 1)


def print_feature_upstreams(tree: ConsequenceTree) -> None:
    """
    打印每个特征节点的直接上游中间后果节点 (一跳)。
    可以扩展成多跳回溯路径。
    """
    print("\n=== 每个特征节点的直接上游后果 ===")
    for feat in FEATURES:
        fid = f"feat::{feat.id}"
        if fid not in tree.nodes:
            continue
        parents = tree.get_parents(fid)
        print(f"* 特征: {feat.label} (id={feat.id})")
        if not parents:
            print("  - 暂无直接上游后果节点")
        else:
            for p in parents:
                print(f"  - 上游后果: {p.text} [{p.domain}/{p.timescale}/{p.polarity}/{p.confidence}]")


# ========================= 6. 示例主程序 =========================

if __name__ == "__main__":
    # ====== 示例: 日本全国现金补贴决策 ======
    decision_description = (
        "中央政府在通胀压力和需求低迷的背景下，向全国居民一次性发放现金补贴，"
        "金额按人头统一，不根据收入差异调整。"
    )

    context = (
        "国家：日本。特点：人口老龄化严重，医保与养老支出占财政比重较高，"
        "长期维持宽松货币政策与高政府债务水平，地方财政普遍紧张。"
    )

    # 1) 构造中间后果 DAG (支持多父继承, DeepSeek 思考模式)
    generator = LLMConsequenceGenerator(
        model_name="deepseek-reasoner",
        temperature=0.4,  # 稍收敛
    )
    builder = ConsequenceTreeBuilder(
        generator=generator,
        max_depth=3,   # 中间后果层数
        max_branch=4,  # 每层最多约 max_branch * 当前层节点数 个子节点
        verbose=True,  # 打印构图过程
    )
    tree = builder.build_tree(
        decision_description=decision_description,
        context=context,
        root_text="政府实施全国性一次性现金补贴政策",
    )

    # 2) 加入固定特征节点
    feature_id_to_node_id = add_feature_nodes_to_tree(tree)

    # 3) 用 DeepSeek 做特征映射 + 打印映射过程
    feature_linker = FeatureLinker(
        model_name="deepseek-reasoner",
        temperature=0.2,  # 这里更保守一些
    )
    attach_features(
        tree,
        feature_linker,
        feature_id_to_node_id,
        max_links_per_node=3,
        verbose=True,
    )

    # 4) 打印图结构概要
    print("\n=== 图结构概要 ===")
    total_nodes = len(tree.nodes)
    total_edges = len(tree.edges)
    intermediate_count = sum(1 for n in tree.nodes.values() if n.node_type == "intermediate")
    feature_count = sum(1 for n in tree.nodes.values() if n.node_type == "feature")
    print(f"总节点数: {total_nodes}")
    print(f"总边数:   {total_edges}")
    print(f"  - 中间节点: {intermediate_count}")
    print(f"  - 特征节点: {feature_count}")

    # 5) 打印从 root 出发的结构 (注意多父会导致重复打印)
    root_ids = [nid for nid, n in tree.nodes.items() if n.source == "root"]
    if root_ids:
        print("\n=== 从根节点出发的定性后果图 (DFS 展开) ===")
        pretty_print_tree(tree, root_ids[0])
    else:
        print("未找到根节点。")

    # 6) 打印每个特征的直接上游后果节点
    print_feature_upstreams(tree)

"use client";

import Link from "next/link";
import { useState } from "react";
import { ChevronRight } from "lucide-react";

interface TreeNode {
  name: string;
  href: string;
  count: number;
  children?: TreeNode[];
}

const TREE_DATA: TreeNode[] = [
  {
    name: "AI / 人工智能",
    href: "/learning/ai",
    count: 1,
    children: [
      { name: "Transformer 架构", href: "/learning/ai/transformer-architecture", count: 0 },
      { name: "Attention 机制", href: "/learning/ai", count: 0 },
      { name: "大语言模型", href: "/learning/ai", count: 0 },
    ],
  },
  {
    name: "Machine Learning",
    href: "/learning/ml",
    count: 0,
    children: [
      { name: "监督学习", href: "/learning/ml", count: 0 },
      { name: "无监督学习", href: "/learning/ml", count: 0 },
      { name: "强化学习", href: "/learning/ml", count: 0 },
    ],
  },
  {
    name: "Deep Learning",
    href: "/learning/ml",
    count: 0,
    children: [
      { name: "CNN", href: "/learning/ml", count: 0 },
      { name: "RNN / LSTM", href: "/learning/ml", count: 0 },
      { name: "GAN", href: "/learning/ml", count: 0 },
    ],
  },
  {
    name: "Python",
    href: "/learning/python",
    count: 0,
    children: [
      { name: "NumPy", href: "/learning/python", count: 0 },
      { name: "PyTorch", href: "/learning/python", count: 0 },
    ],
  },
  {
    name: "C++",
    href: "/learning/python",
    count: 0,
    children: [],
  },
  {
    name: "Rust",
    href: "/learning/python",
    count: 0,
    children: [],
  },
  {
    name: "Math / 数学",
    href: "/learning/math",
    count: 0,
    children: [
      { name: "线性代数", href: "/learning/math", count: 0 },
      { name: "概率论", href: "/learning/math", count: 0 },
      { name: "微积分", href: "/learning/math", count: 0 },
    ],
  },
  {
    name: "Algorithms",
    href: "/learning/math",
    count: 0,
    children: [],
  },
  {
    name: "System Design",
    href: "/learning/math",
    count: 0,
    children: [],
  },
];

function TreeNodeItem({ node, depth = 0 }: { node: TreeNode; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div>
      <div
        className="flex items-center gap-2 py-1.5 group"
        style={{ paddingLeft: `${depth * 20}px` }}
      >
        {hasChildren && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-0.5 rounded text-text-tertiary hover:text-accent transition-colors"
            aria-label={expanded ? "收起" : "展开"}
          >
            <ChevronRight
              size={14}
              className={`transition-transform ${expanded ? "rotate-90" : ""}`}
            />
          </button>
        )}
        {!hasChildren && <span className="w-5" />}
        <Link
          href={node.href}
          className="text-sm text-text-secondary hover:text-accent transition-colors font-medium"
        >
          {node.name}
        </Link>
        {node.count > 0 && (
          <span className="text-[10px] text-text-tertiary bg-tag-bg px-1.5 py-0.5 rounded-full">
            {node.count}
          </span>
        )}
      </div>
      {hasChildren && expanded && (
        <div>
          {node.children!.map((child, i) => (
            <TreeNodeItem key={i} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function KnowledgeTree() {
  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="font-display text-lg font-semibold mb-4">知识地图</h2>
      <div className="space-y-0.5">
        {TREE_DATA.map((node, i) => (
          <TreeNodeItem key={i} node={node} />
        ))}
      </div>
    </div>
  );
}

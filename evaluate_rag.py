from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from config import AppPaths, ensure_dirs
from core_rag import RAGIndex, QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL, QWEN_TIMEOUT

# =========================
# 1. 数据结构
# =========================


@dataclass
class EvalSample:
    id: int
    question: str
    gold_answer: str
    gold_evidence: str
    source_doc: str
    category: str


# =========================
# 2. 读取测试集
# =========================


def load_json_dataset(path: str) -> List[EvalSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        EvalSample(
            id=item["id"],
            question=item["question"],
            gold_answer=item["gold_answer"],
            gold_evidence=item["gold_evidence"],
            source_doc=item["source_doc"],
            category=item.get("category", ""),
        )
        for item in data
    ]


def load_csv_dataset(path: str) -> List[EvalSample]:
    samples: List[EvalSample] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                EvalSample(
                    id=int(row["id"]),
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    gold_evidence=row["gold_evidence"],
                    source_doc=row["source_doc"],
                    category=row.get("category", ""),
                )
            )
    return samples


# =========================
# 3. 项目初始化
# =========================


def create_rag() -> RAGIndex:
    """创建与当前项目一致的 RAGIndex 实例。"""
    ensure_dirs()
    paths = AppPaths()
    return RAGIndex(paths)


# =========================
# 4. 你的 RAG 系统接口
# =========================


def run_rag(rag: RAGIndex, question: str) -> Dict[str, Any]:
    """调用当前项目的 RAG 检索 + 生成逻辑。"""
    start = time.time()
    result = rag.answer(question)
    latency = time.time() - start

    contexts = result.get("contexts", []) or []
    retrieved_context = "\n\n".join(
        [
            f"[文档{i}]\n标题路径：{ctx.get('title_path', '')}\n来源：{ctx.get('source_file', '')}\n内容：{ctx.get('content', '')}"
            for i, ctx in enumerate(contexts, start=1)
        ]
    )

    return {
        "answer": result.get("answer", ""),
        "source": result.get("source", ""),
        "contexts": contexts,
        "retrieved_context": retrieved_context,
        "latency": latency,
    }


# =========================
# 5. LLM-as-judge 调用
# =========================


def _extract_json_object(text: str) -> dict[str, Any]:
    """尽量从模型输出中提取 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"无法从输出中解析 JSON: {text[:500]}")


def judge_answer(
    question: str,
    gold_answer: str,
    gold_evidence: str,
    retrieved_context: str,
    model_answer: str,
) -> Dict[str, Any]:
    """使用千问 OpenAI-compatible 接口做 LLM-as-judge。"""
    if not QWEN_API_KEY:
        # 没有 judge 模型时，脚本仍可运行，返回一个可识别的降级结果
        return {
            "correctness": 0,
            "faithfulness": 0,
            "completeness": 0,
            "relevance": 0,
            "citation_support": 0,
            "overall": 0,
            "judgement": "fail",
            "reason": "未配置 QWEN_API_KEY，已跳过 LLM-as-judge。",
        }

    prompt = f"""
你是一名严格的 RAG 评测裁判。请只根据给定信息评分，不要依赖外部知识。

[Question]
{question}

[Gold Answer]
{gold_answer}

[Gold Evidence]
{gold_evidence}

[Retrieved Context]
{retrieved_context}

[Model Answer]
{model_answer}

请严格输出 JSON，格式如下：
{{
  "correctness": 0-5,
  "faithfulness": 0-5,
  "completeness": 0-5,
  "relevance": 0-5,
  "citation_support": 0-5,
  "overall": 0-5,
  "judgement": "pass|partial|fail",
  "reason": "中文简述"
}}
""".strip()

    import requests

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": os.getenv("QWEN_JUDGE_MODEL", QWEN_MODEL),
        "messages": [
            {"role": "system", "content": "你是一个严格、稳定、只输出 JSON 的评测器。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    base_url = QWEN_BASE_URL.rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=QWEN_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(f"LLM-as-judge 请求失败: {resp.status_code} {resp.text[:1000]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)

    # 字段兜底，避免模型偶尔漏字段导致后续统计失败
    defaults = {
        "correctness": 0,
        "faithfulness": 0,
        "completeness": 0,
        "relevance": 0,
        "citation_support": 0,
        "overall": 0,
        "judgement": "fail",
        "reason": "",
    }
    defaults.update(parsed)
    return defaults


# =========================
# 6. 批量评测
# =========================


def evaluate_dataset(samples: List[EvalSample], rag: RAGIndex) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for sample in samples:
        print(f"Evaluating sample {sample.id} ...")

        rag_result = run_rag(rag, sample.question)
        answer = rag_result.get("answer", "")
        retrieved_context = rag_result.get("retrieved_context", "")
        latency = rag_result.get("latency", 0.0)

        judge_result = judge_answer(
            question=sample.question,
            gold_answer=sample.gold_answer,
            gold_evidence=sample.gold_evidence,
            retrieved_context=retrieved_context,
            model_answer=answer,
        )

        result = {
            "id": sample.id,
            "category": sample.category,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "gold_evidence": sample.gold_evidence,
            "source_doc": sample.source_doc,
            "model_answer": answer,
            "retrieved_context": retrieved_context,
            "source": rag_result.get("source", ""),
            "contexts": json.dumps(rag_result.get("contexts", []), ensure_ascii=False),
            "latency": latency,
            **judge_result,
        }
        results.append(result)

    return results


# =========================
# 7. 统计汇总
# =========================


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        return sum(float(r.get(key, 0) or 0) for r in results) / n

    pass_count = sum(1 for r in results if r.get("judgement") == "pass")
    partial_count = sum(1 for r in results if r.get("judgement") == "partial")
    fail_count = sum(1 for r in results if r.get("judgement") == "fail")

    return {
        "count": n,
        "pass_rate": pass_count / n,
        "partial_rate": partial_count / n,
        "fail_rate": fail_count / n,
        "avg_correctness": avg("correctness"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_completeness": avg("completeness"),
        "avg_relevance": avg("relevance"),
        "avg_citation_support": avg("citation_support"),
        "avg_overall": avg("overall"),
        "avg_latency": avg("latency"),
    }


# =========================
# 8. 保存结果
# =========================


def save_results_json(results: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



def save_summary_json(summary: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



def save_results_csv(results: List[Dict[str, Any]], path: str):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# =========================
# 9. 主程序入口
# =========================


def main():
    dataset_path = os.getenv("EVAL_DATASET_PATH", "eval_set.json")
    rebuild_before_eval = os.getenv("EVAL_REBUILD_INDEX", "0") == "1"
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", "eval_outputs"))
    output_dir.mkdir(exist_ok=True)

    if dataset_path.endswith(".json"):
        samples = load_json_dataset(dataset_path)
    elif dataset_path.endswith(".csv"):
        samples = load_csv_dataset(dataset_path)
    else:
        raise ValueError("只支持 .json 或 .csv 测试集文件")

    rag = create_rag()

    if rebuild_before_eval:
        print("正在重建索引...")
        count = rag.build(rag.paths.doc_dir)
        print(f"索引重建完成，共 {count} 个切片。")

    results = evaluate_dataset(samples, rag)
    summary = summarize_results(results)

    save_results_json(results, str(output_dir / "detailed_results.json"))
    save_results_csv(results, str(output_dir / "detailed_results.csv"))
    save_summary_json(summary, str(output_dir / "summary.json"))

    print("评测完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

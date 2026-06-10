from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class PathsConfig:
    question_file: str = "./eval_data/questions.json"
    results_file: str = "./eval_data/results_benchmark_final.xlsx"
    evaluated_file: str = "./eval_data/results_evaluated_scientific.xlsx"
    runs_root: str = "./eval_data/runs"
    eval_docs_dir: str = "./processed_txt/eval_docs"
    rag_kb_dir: str = "./processed_txt/rag_kb"


@dataclass
class ModelConfig:
    name: str
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0


@dataclass
class EvaluationConfig:
    judge_model: str = "openai/gpt-4o-mini"
    judge_prompt_version: str = "trap_v2"
    judge_throttle_seconds: float = 4.0
    judge_max_retries: int = 5


@dataclass
class RunConfig:
    question_file: str = "./eval_data/questions.json"
    results_file: str = "./eval_data/results_benchmark_final.xlsx"
    methods: List[str] = field(
        default_factory=lambda: ["Zero-Shot", "Advanced-Prompt", "RAG"]
    )
    models: List[str] = field(default_factory=lambda: ["google/gemini-2.0-flash-exp:free"])
    num_runs: int = 5
    seed: int = 42
    temperature: float = 0.0
    use_rag: bool = True
    max_tokens: int = 2048
    throttle_seconds: float = 0.5
    max_retries: int = 2
    model_temperatures: Dict[str, float] = field(default_factory=dict)
    default_max_words: Optional[int] = None
    auto_reindex: bool = True
    rag_chunk_size: int = 400
    rag_chunk_overlap: int = 70
    max_rag_context_chars: int = 6000
    max_prompt_chars: int = 14000
    slow_call_seconds: float = 10.0
    max_consecutive_api_fails: int = 10
    parallel_workers: int = 1
    prompt_zero_shot_path: str = "./prompts/zero_shot.txt"
    prompt_advanced_path: str = "./prompts/advanced_prompt.txt"
    prompt_rag_path: str = "./prompts/rag_prompt.txt"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class QuestionItem:
    id: str
    question: str
    category: str = ""
    task_type_raw: str = ""
    reference_answer: str = ""
    atomic_facts: List[str] = field(default_factory=list)
    answer_type: str = ""
    expected_trap_behavior: str = ""
    max_words: str = ""
    relevant_docs: str = ""
    difficulty: str = ""


@dataclass
class BenchmarkResultRow:
    run_uuid: str
    timestamp_utc: str
    config_hash: str
    run_number: int
    model: str
    method: str
    question_id: str
    question: str
    answer_type: str
    expected_trap_behavior: str
    llm_answer: str
    reference_answer: str
    atomic_facts: str
    document_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_sec: float
    retrieval_sources: str
    error: str = ""


@dataclass
class EvaluationRow:
    likert_score: float
    judge_reason: str
    judge_model: str
    judge_prompt_version: str
    confidence_mean: float = float("nan")
    confidence_found_flag: int = 0

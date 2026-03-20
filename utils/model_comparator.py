"""
模型对比工具 - 加载多个模型检查点，生成对比结果，计算 perplexity 等指标。
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationResult:
    """单个模型的生成结果。"""

    model_name: str
    prompt: str
    output: str
    generation_time: float  # 秒
    perplexity: Optional[float] = None
    num_tokens: int = 0


@dataclass
class ComparisonResult:
    """多模型对比结果。"""

    prompt: str
    results: List[GenerationResult] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)  # model_name -> score


class ModelComparator:
    """
    模型对比器：加载多个阶段的模型检查点，生成并对比输出。

    支持的模型阶段：base / SFT / DPO / RL 等。
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}  # name -> (model, tokenizer)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.comparison_history: List[ComparisonResult] = []

    def load_model(
        self,
        name: str,
        model_path: str,
        model_class: Optional[str] = None,
    ) -> None:
        """
        加载一个模型检查点。

        Args:
            name: 模型标识名，如 "base"、"sft"、"dpo"。
            model_path: 模型路径（HuggingFace 模型名或本地路径）。
            model_class: 可选，自定义模型类名。
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                model = model.to(self.device)
            model.eval()

            # 确保 tokenizer 有 pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.models[name] = (model, tokenizer)
        except Exception as e:
            raise RuntimeError(f"加载模型 '{name}' 失败 ({model_path}): {e}") from e

    def unload_model(self, name: str) -> None:
        """卸载指定模型，释放显存/内存。"""
        if name in self.models:
            del self.models[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def list_models(self) -> List[str]:
        """返回已加载模型的名称列表。"""
        return list(self.models.keys())

    @torch.no_grad()
    def generate(
        self,
        name: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """
        使用指定模型生成文本。

        Args:
            name: 模型名称。
            prompt: 输入提示。
            max_new_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus sampling 参数。

        Returns:
            GenerationResult 包含输出文本、耗时、perplexity 等。
        """
        if name not in self.models:
            raise ValueError(f"模型 '{name}' 尚未加载，请先调用 load_model。")

        model, tokenizer = self.models[name]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.time() - start

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 计算 perplexity
        ppl = self._compute_perplexity(model, tokenizer, prompt + output_text)

        return GenerationResult(
            model_name=name,
            prompt=prompt,
            output=output_text,
            generation_time=elapsed,
            perplexity=ppl,
            num_tokens=len(generated_ids),
        )

    def generate_comparison(
        self,
        prompt: str,
        model_names: Optional[List[str]] = None,
        **generate_kwargs,
    ) -> ComparisonResult:
        """
        对多个模型生成结果进行对比。

        Args:
            prompt: 输入提示。
            model_names: 要对比的模型名称列表，默认使用全部已加载模型。
            **generate_kwargs: 传递给 generate 的额外参数。

        Returns:
            ComparisonResult 包含各模型的生成结果。
        """
        names = model_names or list(self.models.keys())
        comparison = ComparisonResult(prompt=prompt)

        for name in names:
            if name not in self.models:
                continue
            result = self.generate(name, prompt, **generate_kwargs)
            comparison.results.append(result)

        self.comparison_history.append(comparison)
        return comparison

    def save_comparison(self, comparison: ComparisonResult, path: str) -> None:
        """将对比结果保存为 JSON 文件。"""
        data = {
            "prompt": comparison.prompt,
            "results": [
                {
                    "model_name": r.model_name,
                    "output": r.output,
                    "generation_time": r.generation_time,
                    "perplexity": r.perplexity,
                    "num_tokens": r.num_tokens,
                }
                for r in comparison.results
            ],
            "scores": comparison.scores,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_perplexity(self, model, tokenizer, text: str) -> Optional[float]:
        """计算文本的 perplexity。"""
        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            return math.exp(loss)
        except Exception:
            return None

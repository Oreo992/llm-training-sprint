"""
实验跟踪工具类 - 自动记录训练指标、模型输出，并支持实验对比。
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 实验数据默认保存目录
DEFAULT_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


class ExperimentTracker:
    """实验跟踪器，将训练过程中的指标和文本记录到 JSON 文件。"""

    def __init__(
        self,
        experiment_name: str,
        experiments_dir: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        初始化实验跟踪器。

        Args:
            experiment_name: 实验名称，用于创建子目录和文件名。
            experiments_dir: 实验数据保存根目录，默认为项目下的 experiments/。
            tags: 可选的标签字典，用于标记实验（如 day、stage 等）。
        """
        self.experiment_name = experiment_name
        self.dir = Path(experiments_dir) if experiments_dir else DEFAULT_EXPERIMENTS_DIR
        self.experiment_dir = self.dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 实验元数据
        self.metadata: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "tags": tags or {},
        }

        # 指标存储：name -> [(step, value, timestamp), ...]
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        # 文本存储：name -> [(text, timestamp), ...]
        self.texts: Dict[str, List[Dict[str, Any]]] = {}

        # 如果已有数据则加载
        self._load_existing()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        记录一个数值指标。

        Args:
            name: 指标名称，如 "loss"、"accuracy"。
            value: 指标值。
            step: 训练步数（可选，不传则自动递增）。
        """
        if name not in self.metrics:
            self.metrics[name] = []

        if step is None:
            step = len(self.metrics[name])

        self.metrics[name].append(
            {"step": step, "value": value, "timestamp": time.time()}
        )
        self._save()

    def log_text(self, name: str, text: str) -> None:
        """
        记录文本内容（如模型输出、prompt 等）。

        Args:
            name: 文本条目名称。
            text: 文本内容。
        """
        if name not in self.texts:
            self.texts[name] = []

        self.texts[name].append({"text": text, "timestamp": time.time()})
        self._save()

    def get_metric(self, name: str) -> List[Dict[str, Any]]:
        """获取指定指标的全部记录。"""
        return self.metrics.get(name, [])

    def get_latest_metric(self, name: str) -> Optional[float]:
        """获取指定指标的最新值。"""
        records = self.metrics.get(name, [])
        return records[-1]["value"] if records else None

    def summary(self) -> Dict[str, Any]:
        """返回实验摘要（每个指标的最新值）。"""
        result: Dict[str, Any] = {"experiment_name": self.experiment_name}
        for name, records in self.metrics.items():
            if records:
                result[name] = {
                    "latest": records[-1]["value"],
                    "min": min(r["value"] for r in records),
                    "max": max(r["value"] for r in records),
                    "steps": len(records),
                }
        return result

    # ------------------------------------------------------------------
    # 静态 / 类方法
    # ------------------------------------------------------------------

    @staticmethod
    def list_experiments(experiments_dir: Optional[str] = None) -> List[str]:
        """列出所有实验名称。"""
        root = Path(experiments_dir) if experiments_dir else DEFAULT_EXPERIMENTS_DIR
        if not root.exists():
            return []
        return sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and (d / "metrics.json").exists()
        )

    @staticmethod
    def load_experiment(
        experiment_name: str, experiments_dir: Optional[str] = None
    ) -> "ExperimentTracker":
        """加载一个已有的实验。"""
        return ExperimentTracker(experiment_name, experiments_dir)

    @staticmethod
    def compare_experiments(
        exp_name_1: str,
        exp_name_2: str,
        experiments_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对比两个实验，生成对比报告。

        返回字典包含：共有指标的最终值差异、各自独有指标等。
        """
        t1 = ExperimentTracker.load_experiment(exp_name_1, experiments_dir)
        t2 = ExperimentTracker.load_experiment(exp_name_2, experiments_dir)

        all_metrics = set(t1.metrics.keys()) | set(t2.metrics.keys())
        comparison: Dict[str, Any] = {
            "experiment_1": exp_name_1,
            "experiment_2": exp_name_2,
            "metrics": {},
        }

        for m in sorted(all_metrics):
            v1 = t1.get_latest_metric(m)
            v2 = t2.get_latest_metric(m)
            comparison["metrics"][m] = {
                exp_name_1: v1,
                exp_name_2: v2,
                "diff": (v2 - v1) if (v1 is not None and v2 is not None) else None,
            }

        return comparison

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """将当前数据持久化到 JSON 文件。"""
        metrics_path = self.experiment_dir / "metrics.json"
        data = {
            "metadata": self.metadata,
            "metrics": self.metrics,
            "texts": self.texts,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_existing(self) -> None:
        """如果已有 JSON 文件则加载。"""
        metrics_path = self.experiment_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = data.get("metadata", self.metadata)
            self.metrics = data.get("metrics", {})
            self.texts = data.get("texts", {})

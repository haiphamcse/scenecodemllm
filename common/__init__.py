"""Shared infrastructure for spatial-reasoning finetuning experiments.

Modules
-------
argument         CLI dataclasses (ModelArguments, DataArguments, …)
data             Training & VSI-Bench eval dataset builders
collator         Qwen-VL chat-template collator with prompt masking
callbacks        VsibenchMetricsCallback (generation-based eval)
vsibench_metrics VSI-Bench scoring / aggregation (lmms-eval mirror)
"""

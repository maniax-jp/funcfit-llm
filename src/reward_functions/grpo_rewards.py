"""
GRPO用報酬関数の実装

ノートブックサンプル（notebooks/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb）に基づいた
5つの独立した報酬関数を提供します。
"""

import re
from typing import List, Optional, Dict, Any
import langid


# グローバル変数：特殊トークン
reasoning_start: Optional[str] = None
reasoning_end: Optional[str] = None
user_token: Optional[str] = None
assistant_token: Optional[str] = None


def setup_special_tokens(tokenizer) -> Dict[str, Optional[str]]:
    """
    トークナイザーから特殊トークンを自動検出して設定

    Args:
        tokenizer: Hugging Face tokenizer

    Returns:
        特殊トークンの辞書
    """
    global reasoning_start, reasoning_end, user_token, assistant_token

    for token in tokenizer.get_added_vocab().keys():
        if "think" in token and "/" in token:
            reasoning_end = token  # </think>
        elif "think" in token:
            reasoning_start = token  # <think>
        elif "user" in token:
            user_token = token
        elif "assistant" in token:
            assistant_token = token

    return {
        "reasoning_start": reasoning_start,
        "reasoning_end": reasoning_end,
        "user_token": user_token,
        "assistant_token": assistant_token,
    }


def get_lang(text: str) -> str:
    """
    langidを使用してテキストの言語を検出

    Args:
        text: 検出対象のテキスト

    Returns:
        言語コード（'ja', 'en', 'zh'など）。空の場合は'und'
    """
    if not text:
        return "und"
    lang, _ = langid.classify(text)
    return lang


def match_format_exactly(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
    """
    報酬関数1: フォーマットの正確な合致をチェック

    </think>タグの正確な出現（1回のみ）を評価。

    Args:
        completions: GRPOTrainerから渡される完成リスト
                    各要素は[{"role": "assistant", "content": "..."}]形式
        **kwargs: 追加のキーワード引数

    Returns:
        各完成に対する報酬スコアのリスト（+3.0 or 0.0）
    """
    scores = []
    for completion_item in completions:
        # 完成からテキストを抽出
        if isinstance(completion_item, list) and len(completion_item) > 0:
            content = completion_item[0].get("content", "")
        elif isinstance(completion_item, dict):
            content = completion_item.get("content", "")
        else:
            content = str(completion_item)

        # </think>タグが正確に1回出現する場合は+3.0点
        if reasoning_end and content.count(reasoning_end) == 1:
            scores.append(3.0)
        else:
            scores.append(0.0)

    return scores


def match_format_approximately(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
    """
    報酬関数2: おおよそのフォーマット合致をチェック

    <think>と</think>タグのカウントを評価。

    Args:
        completions: GRPOTrainerから渡される完成リスト
        **kwargs: 追加のキーワード引数

    Returns:
        各完成に対する報酬スコアのリスト（-1.0 ~ +0.5）
    """
    scores = []
    for completion_item in completions:
        # 完成からテキストを抽出
        if isinstance(completion_item, list) and len(completion_item) > 0:
            content = completion_item[0].get("content", "")
        elif isinstance(completion_item, dict):
            content = completion_item.get("content", "")
        else:
            content = str(completion_item)

        score = 0.0

        # <think>タグのチェック（1回であれば+0.5、そうでなければ-1.0）
        if reasoning_start:
            if content.count(reasoning_start) == 1:
                score += 0.5
            else:
                score -= 1.0

        # </think>タグのチェック（1回であれば+0.5、そうでなければ-1.0）
        if reasoning_end:
            if content.count(reasoning_end) == 1:
                score += 0.5
            else:
                score -= 1.0

        scores.append(score)

    return scores


def check_answer(completions: List[List[Dict[str, Any]]], prompts: List[Any], **kwargs) -> List[float]:
    """
    報酬関数3: 答えの正確性をチェック

    プロンプトから期待される答えを取得し、完成テキストから数値を抽出して比較。
    数値比が0.9～1.1の範囲内であれば高いスコアを付与。

    Args:
        completions: GRPOTrainerから渡される完成リスト
        prompts: プロンプトのリスト（データセットの"prompt"キーに対応）
        **kwargs: 追加のキーワード引数（"true_value"などを含む可能性）

    Returns:
        各完成に対する報酬スコアのリスト（-4.5 ~ +5.0）
    """
    scores = []

    for i, completion_item in enumerate(completions):
        # 完成からテキストを抽出
        if isinstance(completion_item, list) and len(completion_item) > 0:
            content = completion_item[0].get("content", "")
        elif isinstance(completion_item, dict):
            content = completion_item.get("content", "")
        else:
            content = str(completion_item)

        # 真値を取得（プロンプトまたはkwargsから）
        true_value = None
        if i < len(prompts):
            prompt_item = prompts[i]
            if isinstance(prompt_item, dict):
                true_value = prompt_item.get("true_value")

        if true_value is None:
            # 真値がない場合は中立スコア
            scores.append(0.0)
            continue

        # 完成テキストから数値を抽出
        numbers = re.findall(r"[-+]?\d*\.?\d+", content)
        if not numbers:
            # 数値が見つからない場合は低スコア
            scores.append(-4.5)
            continue

        # 最初の数値を使用
        try:
            predicted_value = float(numbers[0])
        except ValueError:
            scores.append(-4.5)
            continue

        # 真値との比較
        try:
            ratio = predicted_value / float(true_value)
            if 0.9 <= ratio <= 1.1:
                # 10%以内の誤差であれば+5.0点
                scores.append(5.0)
            elif 0.7 <= ratio <= 1.3:
                # 30%以内の誤差であれば+1.5点
                scores.append(1.5)
            else:
                # それ以外は-1.5点
                scores.append(-1.5)
        except (ZeroDivisionError, ValueError):
            scores.append(-4.5)

    return scores


def check_numbers(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
    """
    報酬関数4: テキストからの数値抽出を検証

    完成テキストに数値が含まれているかをチェック。

    Args:
        completions: GRPOTrainerから渡される完成リスト
        **kwargs: 追加のキーワード引数

    Returns:
        各完成に対する報酬スコアのリスト（-1.5 or +3.5）
    """
    scores = []

    for completion_item in completions:
        # 完成からテキストを抽出
        if isinstance(completion_item, list) and len(completion_item) > 0:
            content = completion_item[0].get("content", "")
        elif isinstance(completion_item, dict):
            content = completion_item.get("content", "")
        else:
            content = str(completion_item)

        # 数値が含まれているかチェック
        numbers = re.findall(r"[-+]?\d*\.?\d+", content)
        if numbers:
            scores.append(3.5)
        else:
            scores.append(-1.5)

    return scores


def format_and_language_reward_func(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
    """
    報酬関数5: フォーマットと言語の検出

    langidを使用して言語を検出し、日本語であれば高スコア、英語や中国語は低スコアを付与。

    Args:
        completions: GRPOTrainerから渡される完成リスト
        **kwargs: 追加のキーワード引数

    Returns:
        各完成に対する報酬スコアのリスト（-5.0 ~ +5.0）
    """
    scores = []

    for completion_item in completions:
        # 完成からテキストを抽出
        if isinstance(completion_item, list) and len(completion_item) > 0:
            content = completion_item[0].get("content", "")
        elif isinstance(completion_item, dict):
            content = completion_item.get("content", "")
        else:
            content = str(completion_item)

        # 言語を検出
        lang = get_lang(content)

        # 言語に応じてスコアを付与
        if lang == "ja":  # 日本語
            score = 5.0
        elif lang == "en":  # 英語
            score = -3.0
        elif lang == "zh":  # 中国語
            score = -3.0
        else:  # その他
            score = -5.0

        scores.append(score)

    return scores

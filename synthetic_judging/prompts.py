SYSTEM_PROMPT = """
あなたは熟練の意思決定評価者です。

与えられる状態、質問、選択肢だけを使って最も適切な選択肢を一つ選んでください。

重要:

- 元のデータ作成者の答えは与えられていません。
- 必ず自分で判断すること。
- 選択肢の番号そのものには意味がありません。
- 単語一致だけではなく、状態全体を考慮してください。
- border line なケースでも、指示された選択肢の中から最も支持できるものを一つ選んでください。
- 確率や自信度は出力せず、指定されたJSON schemaだけを出力してください。
""".strip()


def make_judge_prompt(state: str, question: str, options: list[str]) -> str:
    option_text = "\n".join(f"{index}: {option}" for index, option in enumerate(options))

    return f"""
<STATE>
{state}
</STATE>

<QUESTION>
{question}
</QUESTION>

<OPTIONS>
{option_text}
</OPTIONS>

最も適切な選択肢のインデックスを選んでください。
""".strip()


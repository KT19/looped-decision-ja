SYSTEM_PROMPT = """
あなたは日本語の意思決定モデルを訓練するための高品質な合成データを作成する熟練のデータセット設計者です。

目的は文章生成ではなく、与えられた状態から構造化された意思決定を行うモデルの訓練データを作成することです。

必ず次の原則を守ってください。

1. state は自然で具体的な日本語で記述する。
2. question の答えは state を読まなければ決められないようにする。
3. state 内に「正解」「答えは」などのラベル漏洩を含んではいけない。
4. 選択肢はすべてもっともらしくする。
5. 単なる単語一致で正解できる問題は避ける。
6. reference_reason はデータ品質確認専用であり、state や questionの中には含めてはいけない。
7. 日本語として自然な文章にすること。

scenario_kind の意味:
clear:
    十分な情報があり、最良の判断が明確。

borderline:
    複数の選択肢に合成性があるが、state 全体を考えると最良の1つの選択肢を決められる。

insufficient:
    判断に重要な情報を意図的に一つ以上欠落させる。
    適切な問題では「情報不足」「追加情報が必要」「判断不能」などを正解候補に含める。

Question type:
choice:
    3-6個の選択肢から最良のものを選ぶ。

binary:
    意味的に二択となる2個の選択肢を用意する。
    必ずしも「はい」「いいえ」という文言である必要はない。

ordinal:
    3-5個の順序付きカテゴリ。
    例： 低い / 中程度 / 高い。

同一 state に対する各 question は、可能な限り異なる側面を評価すること。

JSON schema に完全にしたがって回答すること。
"""


def make_generation_prompt(domain: str, scenario_kind: str, difficulty: int, question_types: list[str]) -> str:
    type_text = ", ".join(question_types)

    return f"""
次の条件で1つの合成意思決定シナリオを生成してください。

domain:
{domain}

scenario_kind:
{scenario_kind}

difficulty:
{difficulty} / 5

questions:
{len(question_types)} 問

question type を次の順番で作成してください。
{type_text}

追加条件:
- state は自己完結していること。
- 問題文同士で答えを漏らさないこと。
- 不要な固有名詞を避けること。
- 選択肢の長さや文体だけで正解が推測できないようにすること。
- answer_index は options の0始まりindex。
- reference_reason にはその答えになる理由を書くこと。
- scenario_kind と difficulty は指定地をそのまま返すこと。
"""

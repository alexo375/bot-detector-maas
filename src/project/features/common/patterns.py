import math
import re
from collections import Counter

RE_LETTERS = re.compile(r"[a-zA-Zа-яА-ЯёЁ]")
RE_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
RE_WORDS_BOUNDED = re.compile(r"\b\w+\b")
RE_CODE_PATTERN = re.compile(
    r"(def\s+\w+|import\s+\w+|class\s+\w+|const\s+\w+|function\s*\(|console\.log|print\(|SELECT\s+.*FROM|curl\s+)",
    re.IGNORECASE,
)
RE_ECHO_PATTERN = re.compile(
    r"(вы сказали|ты написал|как ты и просил|в предыдущем сообщении|цитирую)",
    re.IGNORECASE,
)
RE_MARKDOWN_PATTERN = re.compile(r"(^|\s)(#+\s|\*|\*\*|`|```)", re.MULTILINE)
RE_BRACKET_SMILE = re.compile(r"(?<![:\w])(\)+|\(+)(?![\w])")
RE_ELLIPSIS = re.compile(r"\.{2,}")
RE_PROBING_PATTERN = re.compile(
    r"(ты\s+бот|вы\s+бот|бот\s*\?|не\s+бот|докажи|скажи\s+\d+|повтори|сколько\s+будет|"
    r"нейросеть|искусственный\s+интеллект|языковая\s+модель|напиши\s+стих|напиши\s+код|"
    r"кто\s+ты|ты\s+человек|ты\s+реальный|проверка\s+на\s+бота)",
    re.IGNORECASE,
)

SPAM_TEMPLATES = (
    "я искусственный интеллект",
    "я языковая модель",
    "чем я могу помочь",
    "как языковая модель",
    "а бот бы смог",
    "ты откуда вообще",
    "представь, что ты бот",
    "saman saman",
)

SLANG_MARKERS = frozenset(
    {
        "хз",
        "ща",
        "щас",
        "че",
        "чо",
        "норм",
        "пон",
        "капец",
        "ппц",
        "жиза",
        "рофл",
        "кринж",
        "лан",
        "спс",
        "пж",
        "чел",
        "бро",
        "ага",
        "неа",
        "имба",
        "кек",
        "лол",
        "прив",
        "ок",
        "го",
        "рил",
        "кста",
        "краш",
        "мда",
        "угу",
    }
)


class TextPatternAnalyzer:
    """Анализатор текстовых паттернов и информационной энтропии реплик."""

    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Вычисление информационной энтропии Шеннона по символам текста."""
        if not text:
            return 0.0
        counts = Counter(text.lower())
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

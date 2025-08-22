import re
import emoji
import regex as re2
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ASCII_RE = re2.compile(r"[^\p{L}\p{N}\s]")  # keep only letters/numbers/spaces

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    x = text.lower()
    x = emoji.replace_emoji(x, replace="")        # remove emojis
    x = URL_RE.sub(" ", x)
    x = MENTION_RE.sub(" ", x)
    x = HASHTAG_RE.sub(" ", x)
    x = NON_ASCII_RE.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    tokens = [t for t in x.split() if t not in STOP or t in {"not","no"}]
    return " ".join(tokens)

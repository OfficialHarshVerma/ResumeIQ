
import re
import unicodedata


def clean_resume_text(text):

    if not isinstance(text, str):
        raise TypeError("Resume text must be a string.")

    if not text.strip():
        raise ValueError("Resume text cannot be empty.")

    # 1. Normalize Unicode characters (NFC).
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 3. Normalize page breaks and tabs.
    text = text.replace("\f", "\n")
    text = text.replace("\t", " ")

    # 4. Normalize non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # 5. Remove unexpected control characters.
    # Preserve newline; tabs and formfeeds were already handled.
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Cc"
        or character == "\n"
    )

    # 6. Collapse repeated horizontal whitespace.
    text = re.sub(r"[^\S\n]+", " ", text)

    # 7. Remove leading/trailing whitespace from each line.
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # 8. Limit consecutive blank lines to two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 9. Remove leading/trailing whitespace.
    text = text.strip()

    if not text:
        raise ValueError(
            "Resume text became empty after cleaning."
        )

    return text


from datetime import datetime
from dateutil.relativedelta import relativedelta

def parse_date(value, is_end=False):

    if not value:
        return None

    value = str(value).strip()

    if value.lower() in ("present", "current", "ongoing"):
        return datetime.now().replace(day=1) if is_end else None

    formats = [
        "%b %Y",       # Jun 2025
        "%B %Y",       # June 2025
        "%m/%Y",       # 06/2025
        "%Y-%m",       # 2025-06
        "%Y"           # 2024
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None

def get_experience_category(entry):

    employment_type = (entry.get("employment_type") or "").lower()
    job_title = (entry.get("job_title") or "").lower()

    if "intern" in employment_type or "intern" in job_title:
        return "internship"

    if any(word in employment_type for word in [
        "full-time", "full time",
        "part-time", "part time",
        "contract", "permanent"
    ]):
        return "professional"

    if any(word in job_title for word in [
        "engineer", "developer",
        "analyst", "manager",
        "consultant"
    ]):
        return "professional"

    return "unclassified"


def merge_and_count(intervals):
    """Merge overlapping employment periods to avoid double-counting."""

    if not intervals:
        return 0

    intervals.sort()

    merged = [list(intervals[0])]

    for start, end in intervals[1:]:

        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return sum(end - start + 1 for start, end in merged)


def calculate_experience(experience):

    intervals = {
        "professional": [],
        "internship": [],
        "unclassified": []
    }

    for entry in experience:

        start = parse_date(entry.get("start_date"))
        end = parse_date(entry.get("end_date"), is_end=True)

        if start is None or end is None or end < start:
            continue

        category = get_experience_category(entry)

        start_month = start.year * 12 + start.month
        end_month = end.year * 12 + end.month

        intervals[category].append((start_month, end_month))

    professional_months = merge_and_count(
        intervals["professional"]
    )

    internship_months = merge_and_count(
        intervals["internship"]
    )

    unclassified_months = merge_and_count(
        intervals["unclassified"]
    )
    
    candidate_level = (
        "Fresher" if professional_months == 0
        else "Junior" if professional_months <= 36
        else "Experienced"
    )

    return {
        "total_professional_experience_months": professional_months,
        "total_professional_experience_years": round(
            professional_months / 12, 2
        ),
        "total_internship_experience_months": internship_months,
        "total_internship_experience_years": round(
            internship_months / 12, 2
        ),
        "unclassified_experience_months": unclassified_months,
        "total_experience_months": professional_months,
        "total_experience_years": round(professional_months / 12, 2),
        "candidate_level": candidate_level
    }

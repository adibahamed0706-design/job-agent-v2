import os
import re
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS_CSV = OUTPUT_DIR / "weekly_jobs.csv"
REPORT_MD = OUTPUT_DIR / "weekly_jobs_report.md"
SEEN_IDS_FILE = OUTPUT_DIR / "seen_job_ids.json"


def load_config() -> Dict[str, Any]:
    raw = os.getenv("JOB_AGENT_CONFIG_JSON")
    if not raw:
        raise ValueError("Missing JOB_AGENT_CONFIG_JSON secret.")
    return json.loads(raw)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s/+.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_seen_ids() -> set:
    if not SEEN_IDS_FILE.exists():
        return set()
    try:
        return set(json.loads(SEEN_IDS_FILE.read_text()))
    except Exception:
        return set()


def save_seen_ids(ids_set: set) -> None:
    SEEN_IDS_FILE.write_text(json.dumps(sorted(ids_set), indent=2))


def stable_job_id(title: str, company: str, location: str, link: str) -> str:
    raw = f"{title}|{company}|{location}|{link}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def serpapi_search(query: str, api_key: str, num: int = 20) -> List[Dict[str, Any]]:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": query,
        "hl": "en",
        "api_key": api_key
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("jobs_results", []) or []
    return results[:num]


def parse_job(result: Dict[str, Any]) -> Dict[str, Any]:
    title = result.get("title", "")
    company = result.get("company_name", "")
    location = result.get("location", "")
    description = result.get("description", "")
    apply_options = result.get("apply_options", []) or []

    link = ""
    if apply_options and isinstance(apply_options, list):
        first = apply_options[0]
        link = first.get("link", "") or first.get("apply_link", "")

    posted_at = result.get("detected_extensions", {}).get("posted_at", "")
    schedule = result.get("detected_extensions", {}).get("schedule_type", "")
    via = result.get("via", "")

    job_id = stable_job_id(title, company, location, link)

    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "link": link,
        "posted_at": posted_at,
        "schedule_type": schedule,
        "via": via
    }


def build_candidate_text(profile: Dict[str, Any]) -> str:
    parts = [
        profile.get("headline", ""),
        profile.get("resume_summary", ""),
        " ".join(profile.get("core_skills", [])),
        " ".join(profile.get("target_titles", [])),
        " ".join(profile.get("must_have_terms", [])),
        " ".join(profile.get("nice_to_have_terms", []))
    ]
    return normalize_text(" ".join(parts))


def contains_any(text: str, terms: List[str]) -> List[str]:
    text_n = normalize_text(text)
    found = []
    for term in terms:
        t = normalize_text(term)
        if t and t in text_n:
            found.append(term)
    return found


def score_job(job: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    candidate_text = build_candidate_text(profile)

    job_text_raw = " ".join([
        job.get("title", ""),
        job.get("company", ""),
        job.get("location", ""),
        job.get("description", ""),
        job.get("schedule_type", "")
    ])
    job_text = normalize_text(job_text_raw)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    mat = vectorizer.fit_transform([candidate_text, job_text])
    cosine = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])

    title_hits = contains_any(job.get("title", ""), profile.get("target_titles", []))
    skill_hits = contains_any(job_text_raw, profile.get("core_skills", []))
    must_hits = contains_any(job_text_raw, profile.get("must_have_terms", []))
    nice_hits = contains_any(job_text_raw, profile.get("nice_to_have_terms", []))
    exclude_hits = contains_any(job_text_raw, profile.get("exclude_terms", []))
    location_hits = contains_any(job.get("location", ""), profile.get("preferred_locations", []))

    score = 0.0
    score += cosine * 55
    score += min(len(title_hits), 2) * 8
    score += min(len(skill_hits), 6) * 3.5
    score += min(len(must_hits), 4) * 4
    score += min(len(nice_hits), 4) * 1.5
    score += min(len(location_hits), 2) * 4

    title_lower = normalize_text(job.get("title", ""))
    if "senior" in title_lower or "lead" in title_lower or "manager" in title_lower:
        score += 4

    if "remote" in normalize_text(job.get("location", "")):
        score += 4

    if exclude_hits:
        score -= 20

    score = max(0, min(100, round(score, 1)))

    reasons = []
    if title_hits:
        reasons.append(f"title alignment: {', '.join(title_hits[:2])}")
    if skill_hits:
        reasons.append(f"skill overlap: {', '.join(skill_hits[:5])}")
    if must_hits:
        reasons.append(f"must-have matches: {', '.join(must_hits[:4])}")
    if location_hits:
        reasons.append(f"location fit: {', '.join(location_hits[:2])}")
    if not reasons:
        reasons.append("general semantic similarity to your background")

    gaps = []
    verdict = "Apply"
    if score < 58:
        verdict = "Stretch / Review"
    if exclude_hits or score < 45:
        verdict = "Skip"

    details = {
        "cosine_similarity": round(cosine, 3),
        "title_hits": title_hits,
        "skill_hits": skill_hits,
        "must_hits": must_hits,
        "nice_hits": nice_hits,
        "exclude_hits": exclude_hits,
        "location_hits": location_hits,
        "reasons": reasons,
        "gaps": gaps,
        "verdict": verdict
    }
    return score, details


def dedupe_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for job in jobs:
        key = (
            normalize_text(job.get("title", "")),
            normalize_text(job.get("company", "")),
            normalize_text(job.get("location", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


def fetch_jobs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("Missing SERPAPI_API_KEY secret.")

    queries = config["search_queries"]
    max_results_per_query = config["filters"].get("max_results_per_query", 20)

    all_jobs = []
    for query in queries:
        try:
            raw_results = serpapi_search(query, api_key, num=max_results_per_query)
            parsed = [parse_job(r) for r in raw_results]
            all_jobs.extend(parsed)
            print(f"Fetched {len(parsed)} jobs for query: {query}")
        except Exception as e:
            print(f"Failed query '{query}': {e}")

    all_jobs = dedupe_jobs(all_jobs)
    return all_jobs


def build_dataframe(scored_jobs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in scored_jobs:
        job = item["job"]
        details = item["details"]
        rows.append({
            "score": item["score"],
            "verdict": details["verdict"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "posted_at": job["posted_at"],
            "schedule_type": job["schedule_type"],
            "link": job["link"],
            "why_it_fits": " | ".join(details["reasons"]),
            "gaps": ", ".join(details["gaps"]) if details["gaps"] else "",
            "skill_hits": ", ".join(details["skill_hits"][:8]),
            "must_hits": ", ".join(details["must_hits"][:8]),
            "title_hits": ", ".join(details["title_hits"][:4]),
            "exclude_hits": ", ".join(details["exclude_hits"][:4]),
            "job_id": job["job_id"]
        })
    return pd.DataFrame(rows).sort_values(by=["score", "company"], ascending=[False, True])


def render_markdown(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# Weekly Job Recommendations")
    lines.append("")
    lines.append(f"Generated: **{today}**")
    lines.append("")
    lines.append(f"Target titles: {', '.join(config['candidate_profile']['target_titles'][:8])}")
    lines.append("")
    lines.append(f"Total recommendations: **{len(df)}**")
    lines.append("")

    top = df.head(20)
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(f"## {idx}. {row.title} — {row.company}")
        lines.append(f"- **Score:** {row.score}")
        lines.append(f"- **Verdict:** {row.verdict}")
        lines.append(f"- **Location:** {row.location}")
        lines.append(f"- **Posted:** {row.posted_at or 'N/A'}")
        lines.append(f"- **Why it fits:** {row.why_it_fits}")
        if row.gaps:
            lines.append(f"- **Possible gaps:** {row.gaps}")
        if row.link:
            lines.append(f"- **Apply link:** {row.link}")
        lines.append("")
    return "\n".join(lines)


def main():
    config = load_config()
    profile = config["candidate_profile"]
    minimum_score = config["filters"].get("minimum_score", 58)
    top_n_final = config["filters"].get("top_n_final", 20)

    seen_ids = load_seen_ids()
    jobs = fetch_jobs(config)

    new_jobs = [j for j in jobs if j["job_id"] not in seen_ids]

    scored_jobs = []
    for job in new_jobs:
        score, details = score_job(job, profile)
        if score >= minimum_score and details["verdict"] != "Skip":
            scored_jobs.append({
                "job": job,
                "score": score,
                "details": details
            })

    scored_jobs.sort(key=lambda x: x["score"], reverse=True)
    scored_jobs = scored_jobs[:top_n_final]

    if not scored_jobs:
        REPORT_MD.write_text("# Weekly Job Recommendations\n\nNo matching jobs found this week.\n")
        print("No matching jobs found.")
        return

    df = build_dataframe(scored_jobs)
    df.to_csv(JOBS_CSV, index=False)
    REPORT_MD.write_text(render_markdown(df, config), encoding="utf-8")

    for item in scored_jobs:
        seen_ids.add(item["job"]["job_id"])
    save_seen_ids(seen_ids)

    print(f"Saved {len(df)} recommended jobs.")
    print(f"CSV: {JOBS_CSV}")
    print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()

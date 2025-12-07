from __future__ import annotations

import io
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, date, timezone, timedelta
from typing import List, Optional, Tuple, Dict
import sqlite3
from io import BytesIO
from textwrap import wrap

import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# Optional: path to Echt logo PNG (or leave as None)
LOGO_PATH = None

# Optional: paths to Poppins fonts for PDF (if you upload them with the app)
POPPINS_REGULAR_TTF = None  # e.g. "Poppins-Regular.ttf"
POPPINS_BOLD_TTF = None     # e.g. "Poppins-Bold.ttf"

PRIORITY_ORDER: Dict[Optional[str], int] = {
    "urgent": 1,
    "high": 2,
    "normal": 3,
    "medium": 3,
    "low": 4,
    None: 5,
}

COMPLETED_STATUSES = {
    "complete",
    "completed",
    "done",
    "closed",
    "resolved",
}

DB_PATH = "monitor.db"
LAST_MINUTE_THRESHOLD_HOURS = 48.0  # what we consider "last-minute" reassignment


# =======================================
# CLICKUP API HELPERS
# =======================================

def clickup_get(url: str, params: Dict | None = None) -> Dict:
    """Low-level GET wrapper for ClickUp API."""
    headers = {"Authorization": CLICKUP_API_TOKEN}
    resp = requests.get(url, headers=headers, params=params or {})
    if resp.status_code != 200:
        raise RuntimeError(f"ClickUp API error {resp.status_code}: {resp.text}")
    return resp.json()


def get_spaces(team_id: str) -> List[Dict]:
    url = f"https://api.clickup.com/api/v2/team/{team_id}/space"
    data = clickup_get(url)
    return data.get("spaces", [])


def get_folders(space_id: str) -> List[Dict]:
    url = f"https://api.clickup.com/api/v2/space/{space_id}/folder"
    data = clickup_get(url)
    return data.get("folders", [])


def get_lists_in_folder(folder_id: str) -> List[Dict]:
    url = f"https://api.clickup.com/api/v2/folder/{folder_id}/list"
    data = clickup_get(url)
    return data.get("lists", [])


def get_folderless_lists(space_id: str) -> List[Dict]:
    """Lists directly under a space (no folder)."""
    url = f"https://api.clickup.com/api/v2/space/{space_id}/list"
    data = clickup_get(url)
    return data.get("lists", [])


def get_tasks_for_list(list_id: str) -> List[Dict]:
    """
    Fetch tasks for a given list, including closed & subtasks, with pagination.
    """
    all_tasks: List[Dict] = []
    page = 0

    while True:
        url = f"https://api.clickup.com/api/v2/list/{list_id}/task"
        params = {
            "subtasks": "true",
            "include_closed": "true",
            "page": page,
        }
        data = clickup_get(url, params=params)
        tasks = data.get("tasks", [])
        if not tasks:
            break
        all_tasks.extend(tasks)
        page += 1

    return all_tasks


def fetch_active_member_identifiers() -> Set[str]:
    """
    Returns a set of usernames/emails for *current* workspace members.

    If /team endpoint fails, returns empty set and we skip active-member filtering.
    """
    url = "https://api.clickup.com/api/v2/team"
    try:
        data = clickup_get(url)
    except RuntimeError:
        return set()

    active_ids: Set[str] = set()

    for team in data.get("teams", []):
        if str(team.get("id")) != str(TEAM_ID):
            continue

        for m in team.get("members", []):
            user = m.get("user", m) or {}
            status = str(user.get("status", "active")).lower()
            if status not in ("active", "invited", "pending"):
                continue

            username = user.get("username")
            email = user.get("email")

            if username:
                active_ids.add(username)
            if email:
                active_ids.add(email)

    return active_ids


# =======================================
# TASK → DATAFRAME
# =======================================

def tasks_to_rows(
    tasks: List[Dict],
    client_name: str,
    space_id: str,
    folder_name: Optional[str],
    list_name: str,
) -> List[Dict]:
    rows: List[Dict] = []
    for t in tasks:
        # Due date
        due_date: Optional[date] = None
        if t.get("due_date"):
            try:
                due_date = datetime.fromtimestamp(int(t["due_date"]) / 1000).date()
            except Exception:
                due_date = None

        # Priority
        priority_val = None
        if t.get("priority"):
            priority_val = t["priority"].get("priority")

        # Assignee (first assignee if exists)
        assignee_name = None
        if t.get("assignees"):
            first_assignee = t["assignees"][0]
            assignee_name = first_assignee.get("username") or first_assignee.get("email")

        rows.append(
            {
                "client": client_name,
                "space_id": space_id,
                "folder": folder_name,
                "list": list_name,
                "task_id": t.get("id"),
                "task_name": t.get("name"),
                "status": (t.get("status", {}) or {}).get("status"),
                "priority": priority_val,
                "assignee": assignee_name,
                "due_date": due_date,
                "url": t.get("url"),
            }
        )
    return rows


def fetch_all_tasks_for_team(team_id: str) -> pd.DataFrame:
    """Fetch all tasks across all spaces (clients) for the given team."""
    spaces = get_spaces(team_id)
    all_rows: List[Dict] = []

    for space in spaces:
        space_id = space.get("id")
        client_name = space.get("name")

        # Folders & lists in them
        folders = get_folders(space_id)
        for folder in folders:
            folder_name = folder.get("name")
            folder_id = folder.get("id")
            lists = get_lists_in_folder(folder_id)
            for lst in lists:
                list_id = lst.get("id")
                list_name = lst.get("name")
                tasks = get_tasks_for_list(list_id)
                all_rows.extend(
                    tasks_to_rows(
                        tasks=tasks,
                        client_name=client_name,
                        space_id=space_id,
                        folder_name=folder_name,
                        list_name=list_name,
                    )
                )

        # Lists directly under space (no folder)
        fl_lists = get_folderless_lists(space_id)
        for lst in fl_lists:
            list_id = lst.get("id")
            list_name = lst.get("name")
            tasks = get_tasks_for_list(list_id)
            all_rows.extend(
                tasks_to_rows(
                    tasks=tasks,
                    client_name=client_name,
                    space_id=space_id,
                    folder_name=None,
                    list_name=list_name,
                )
            )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


# =======================================
# PRIORITY / DUE DATE / STATUS
# =======================================

def add_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    if "priority" not in df.columns:
        df["priority"] = None

    df["priority_lower"] = df["priority"].astype(str).str.lower()
    df["priority_score"] = (
        df["priority_lower"]
        .map(PRIORITY_ORDER)
        .fillna(PRIORITY_ORDER[None])
        .astype(int)
    )
    return df


def add_due_info(df: pd.DataFrame) -> pd.DataFrame:
    today = date.today()
    if "due_date" not in df.columns:
        df["due_date"] = None

    def _days_to_due(d: Optional[date]) -> Optional[int]:
        if pd.isna(d) or d is None:
            return None
        return (d - today).days

    df["days_to_due"] = df["due_date"].apply(_days_to_due)
    df["overdue"] = df["days_to_due"].apply(
        lambda x: True if x is not None and x < 0 else False
    )
    return df


def add_completion_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["status_lower"] = df["status"].astype(str).str.lower()
    df["is_completed"] = df["status_lower"].isin(COMPLETED_STATUSES)
    return df


def label_meeting_bucket(row: pd.Series) -> str:
    prio_label = (
        row["priority"].capitalize()
        if isinstance(row["priority"], str) and row["priority"]
        else "No priority"
    )

    if row["overdue"]:
        return f"Overdue – {prio_label}"
    if row["days_to_due"] is None:
        return f"No due date – {prio_label}"
    if row["days_to_due"] <= 3:
        return f"Due in ≤3 days – {prio_label}"
    if row["days_to_due"] <= 7:
        return f"Due in ≤7 days – {prio_label}"
    return f"Due later – {prio_label}"


def add_meeting_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df["meeting_bucket"] = df.apply(label_meeting_bucket, axis=1)
    return df


def filter_meeting_items(df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
    """
    Keep overdue tasks + tasks due within `days_ahead`.
    """
    mask = (
        (df["overdue"] == True)
        | (
            df["days_to_due"].notnull()
            & (df["days_to_due"] <= days_ahead)
        )
    )
    return df[mask]


# =======================================
# DUE DATE CHANGE TRACKING (SQLITE)
# =======================================

def init_due_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS due_history (
                task_id TEXT PRIMARY KEY,
                last_due_date TEXT,
                change_count INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()


def update_due_change_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Persist last seen due_date per task and count how many times it has changed.
    """
    init_due_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        for _, row in df.iterrows():
            tid = row["task_id"]
            due = (
                row["due_date"].isoformat()
                if pd.notnull(row["due_date"])
                else None
            )

            cur.execute(
                "SELECT last_due_date, change_count FROM due_history WHERE task_id = ?",
                (tid,),
            )
            existing = cur.fetchone()

            if existing is None:
                cur.execute(
                    """
                    INSERT INTO due_history (task_id, last_due_date, change_count)
                    VALUES (?, ?, 0)
                    """,
                    (tid, due),
                )
            else:
                last_due, count = existing
                if last_due is not None and due is not None and last_due != due:
                    count += 1
                cur.execute(
                    """
                    UPDATE due_history
                    SET last_due_date = ?, change_count = ?
                    WHERE task_id = ?
                    """,
                    (due, count, tid),
                )

        conn.commit()
        metrics = pd.read_sql_query(
            "SELECT task_id, change_count FROM due_history", conn
        )

    df = df.merge(metrics, on="task_id", how="left")
    df["change_count"] = df["change_count"].fillna(0).astype(int)
    return df


# =======================================
# ASSIGNMENT / REASSIGNMENT TRACKING
# =======================================

def init_assignment_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignment_state (
                task_id TEXT PRIMARY KEY,
                assignee TEXT,
                due_date TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignment_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                from_assignee TEXT,
                to_assignee TEXT,
                event_ts TEXT,
                due_date TEXT,
                hours_to_due REAL
            )
            """
        )
        conn.commit()


def update_assignment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track when a task's assignee changes.
    Insert an event with from_assignee, to_assignee, and hours_to_due.
    Also return df with extra 'reassign_count' per task.
    """
    init_assignment_db()
    now_iso = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        for _, row in df.iterrows():
            tid = row["task_id"]
            assignee = row["assignee"]
            if not assignee:
                continue

            due_str = row["due_date"].isoformat() if pd.notnull(row["due_date"]) else None

            cur.execute(
                "SELECT assignee, due_date FROM assignment_state WHERE task_id = ?",
                (tid,),
            )
            existing = cur.fetchone()

            if existing is None:
                cur.execute(
                    "INSERT INTO assignment_state (task_id, assignee, due_date) VALUES (?, ?, ?)",
                    (tid, assignee, due_str),
                )
            else:
                prev_assignee, prev_due = existing

                if prev_assignee != assignee:
                    hours_to_due = None
                    if pd.notnull(row["due_date"]):
                        due_dt = datetime.combine(
                            row["due_date"],
                            datetime.min.time(),
                            tzinfo=timezone.utc,
                        )
                        hours_to_due = (
                            due_dt - datetime.now(timezone.utc)
                        ).total_seconds() / 3600.0

                    cur.execute(
                        """
                        INSERT INTO assignment_events (
                            task_id, from_assignee, to_assignee, event_ts, due_date, hours_to_due
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (tid, prev_assignee, assignee, now_iso, due_str, hours_to_due),
                    )

                    cur.execute(
                        "UPDATE assignment_state SET assignee = ?, due_date = ? WHERE task_id = ?",
                        (assignee, due_str, tid),
                    )
                elif prev_due != due_str:
                    cur.execute(
                        "UPDATE assignment_state SET due_date = ? WHERE task_id = ?",
                        (due_str, tid),
                    )

        conn.commit()

        reassign_df = pd.read_sql_query(
            "SELECT task_id, COUNT(*) AS reassign_count FROM assignment_events GROUP BY task_id",
            conn,
        )

    df = df.merge(reassign_df, on="task_id", how="left")
    df["reassign_count"] = df["reassign_count"].fillna(0).astype(int)
    return df


def get_reassignment_offenders(
    threshold_hours: float = LAST_MINUTE_THRESHOLD_HOURS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns throwers & receivers DataFrames."""
    init_assignment_db()
    with sqlite3.connect(DB_PATH) as conn:
        throwers = pd.read_sql_query(
            """
            SELECT
                from_assignee AS assignee,
                COUNT(*) AS reassign_events,
                SUM(CASE WHEN hours_to_due IS NOT NULL AND hours_to_due <= ? THEN 1 ELSE 0 END)
                    AS last_minute_events
            FROM assignment_events
            WHERE from_assignee IS NOT NULL
            GROUP BY from_assignee
            """,
            conn,
            params=(threshold_hours,),
        )

        receivers = pd.read_sql_query(
            """
            SELECT
                to_assignee AS assignee,
                COUNT(*) AS tasks_received,
                SUM(CASE WHEN hours_to_due IS NOT NULL AND hours_to_due <= ? THEN 1 ELSE 0 END)
                    AS last_minute_received
            FROM assignment_events
            WHERE to_assignee IS NOT NULL
            GROUP BY to_assignee
            """,
            conn,
            params=(threshold_hours,),
        )

    return throwers, receivers


# =======================================
# COMMENT METRICS (QUESTION-BASED)
# =======================================

def get_task_comments(task_id: str) -> List[Dict]:
    url = f"https://api.clickup.com/api/v2/task/{task_id}/comment"
    data = clickup_get(url)
    return data.get("comments", [])


def add_comment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each task, find the most recent comment containing '?'
    and check if the assignee has replied afterwards.
    Also store the question date for reporting.
    """
    records: List[Dict] = []

    for _, row in df.iterrows():
        tid = row["task_id"]
        assignee = row["assignee"]
        comments = get_task_comments(tid)
        if not comments:
            continue

        # newest first; find most recent question
        question_idx = None
        for idx, c in enumerate(comments):
            text = (c.get("comment_text") or c.get("comment") or "").strip()
            if "?" in text:
                question_idx = idx
                break

        latest_question_author = None
        latest_question_text = None
        latest_question_age_days = None
        latest_question_date = None
        pending = False

        if question_idx is not None:
            q = comments[question_idx]
            q_user_data = q.get("user") or {}
            latest_question_author = q_user_data.get("username") or q_user_data.get("email")
            latest_question_text = (q.get("comment_text") or q.get("comment") or "").strip()

            try:
                q_ts = int(q.get("date", 0)) / 1000.0
                q_dt = datetime.fromtimestamp(q_ts, tz=timezone.utc)
                now = datetime.now(timezone.utc)
                latest_question_age_days = (now - q_dt).total_seconds() / 86400
                latest_question_date = q_dt.date().isoformat()
            except Exception:
                latest_question_age_days = None
                latest_question_date = None

            newer_comments = comments[:question_idx]
            assignee_replied_later = False
            if assignee:
                for c in newer_comments:
                    cu = (c.get("user") or {})
                    cu_name = cu.get("username") or cu.get("email")
                    if cu_name == assignee:
                        assignee_replied_later = True
                        break

            pending = bool(assignee) and not assignee_replied_later
        else:
            pending = False

        records.append(
            {
                "task_id": tid,
                "latest_comment_author": latest_question_author,
                "latest_comment_text": latest_question_text,
                "latest_comment_age_days": latest_question_age_days,
                "latest_comment_date": latest_question_date,
                "comment_pending_for_assignee": pending,
            }
        )

    if not records:
        return df

    cm = pd.DataFrame(records)
    df = df.merge(cm, on="task_id", how="left")
    return df


# =======================================
# PER-PERSON SUMMARY / LEADERBOARD
# =======================================

def build_people_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-assignee metrics used for scorecards and PDF."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "assignee",
                "tasks_total",
                "overdue_tasks",
                "change_total",
                "unanswered_count",
                "avg_unanswered_age",
                "overdue_rate",
                "change_rate",
                "unanswered_rate",
                "score",
            ]
        )

    base = (
        df.groupby("assignee", dropna=True)
        .agg(
            tasks_total=("task_id", "nunique"),
            overdue_tasks=("overdue", "sum"),
            change_total=("change_count", "sum"),
        )
        .reset_index()
    )

    # unanswered questions
    if (
        "comment_pending_for_assignee" in df.columns
        and "latest_comment_age_days" in df.columns
    ):
        pending = df[
            (df["comment_pending_for_assignee"] == True)
            & df["assignee"].notnull()
        ]
        if not pending.empty:
            unanswered = (
                pending.groupby("assignee")
                .agg(
                    unanswered_count=("task_id", "nunique"),
                    avg_unanswered_age=("latest_comment_age_days", "mean"),
                )
                .reset_index()
            )
            base = base.merge(unanswered, on="assignee", how="left")
        else:
            base["unanswered_count"] = 0
            base["avg_unanswered_age"] = None
    else:
        base["unanswered_count"] = 0
        base["avg_unanswered_age"] = None

    base["unanswered_count"] = base["unanswered_count"].fillna(0).astype(int)

    base["overdue_rate"] = base.apply(
        lambda r: (r["overdue_tasks"] / r["tasks_total"]) if r["tasks_total"] > 0 else 0.0,
        axis=1,
    )
    base["change_rate"] = base.apply(
        lambda r: (r["change_total"] / r["tasks_total"]) if r["tasks_total"] > 0 else 0.0,
        axis=1,
    )
    base["unanswered_rate"] = base.apply(
        lambda r: (r["unanswered_count"] / r["tasks_total"]) if r["tasks_total"] > 0 else 0.0,
        axis=1,
    )

    # higher score = worse (for leaderboard)
    base["score"] = (
        base["overdue_rate"] * 100.0
        + base["change_rate"] * 10.0
        + base["unanswered_rate"] * 5.0
    )

    return base


def auto_feedback_for_person(row: pd.Series) -> str:
    name = row["assignee"]
    tasks = row["tasks_total"]
    overdue = row["overdue_tasks"]
    changes = row["change_total"]
    unanswered = row["unanswered_count"]
    overdue_rate = row["overdue_rate"]

    if tasks == 0:
        return f"{name} has no tasks in this reporting period."

    if overdue == 0 and unanswered == 0 and changes <= 1:
        return (
            f"{name} maintained a clean track record this period – "
            "no overdue tasks, almost no due-date changes and all questions responded to."
        )

    parts = []

    if overdue == 0:
        parts.append("no overdue tasks")
    elif overdue_rate < 0.1:
        parts.append(f"only {overdue} overdue task(s)")
    else:
        parts.append(f"{overdue} overdue task(s) that need attention")

    if changes == 0:
        parts.append("no due-date changes")
    elif changes <= 3:
        parts.append(f"{changes} due-date change(s)")
    else:
        parts.append(f"{changes} due-date changes (review planning)")

    if unanswered == 0:
        parts.append("all ClickUp questions cleared")
    else:
        parts.append(f"{unanswered} unanswered question(s) in comments")

    feedback = f"For this period, {name} handled {tasks} task(s) with " + ", ".join(parts) + "."
    return feedback


# =======================================
# EXPORT HELPERS
# =======================================

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return an XLSX as raw bytes for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="The Monitor")
    output.seek(0)
    return output.getvalue()
# =======================================
# PDF REPORT GENERATION (DESIGN-FOCUSED)
# =======================================

def _rate_color(rate: float) -> "colors.Color":
    from reportlab.lib import colors
    rate = max(0.0, min(1.0, float(rate)))
    if rate == 0:
        return colors.HexColor("#16a34a")  # green
    if rate <= 0.25:
        return colors.HexColor("#22c55e")  # soft green
    if rate <= 0.5:
        return colors.HexColor("#facc15")  # amber
    if rate <= 0.75:
        return colors.HexColor("#fb923c")  # orange
    return colors.HexColor("#dc2626")      # red


def _truncate(text: str, max_len: int = 110) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def _status_breakdown_for_person(df_view: pd.DataFrame, assignee: str) -> pd.DataFrame:
    person = df_view[df_view["assignee"] == assignee].copy()
    if person.empty:
        return pd.DataFrame(columns=["status", "count"])

    if "status" not in person.columns:
        person["status"] = "Unknown"

    return (
        person.groupby("status")
        .agg(count=("task_id", "nunique"))
        .reset_index()
        .sort_values("count", ascending=False)
    )


def _compute_top_and_bottom(people_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top 3 'best' and bottom 3 'worst' performers."""
    if people_summary.empty:
        return (
            pd.DataFrame(columns=people_summary.columns),
            pd.DataFrame(columns=people_summary.columns),
        )

    df = people_summary.copy()

    # "Goodness" score: the lower the bad rates, the higher the good score
    df["good_score"] = (
        (1.0 - df["overdue_rate"]) * 0.5
        + (1.0 - df["unanswered_rate"]) * 0.3
        + (1.0 - df["change_rate"]) * 0.2
    )

    df_non_zero = df[df["tasks_total"] > 0].copy()
    if df_non_zero.empty:
        return (
            pd.DataFrame(columns=people_summary.columns),
            pd.DataFrame(columns=people_summary.columns),
        )

    top3 = df_non_zero.sort_values("good_score", ascending=False).head(3)
    bottom3 = df_non_zero.sort_values("score", ascending=False).head(3)  # "worst" by bad score

    return top3, bottom3


def build_pdf_report(
    df_view: pd.DataFrame,
    people_summary: pd.DataFrame,
    person_notes: Dict[str, str],
    motivation_text: str,
    timeframe_label: str,
    generated_label: str,
    unanswered_df: Optional[pd.DataFrame] = None,
    active_members: Optional[Set[str]] = None,
) -> bytes:
    """
    Build a multi-page PDF report with a designed layout:

    - Strong cover page in Echt style
    - Team overview + zero-task people
    - Unanswered questions (List • Comment • Date)
    - Per-assignee cards with health bars & neat separators
    - Top 3 vs Bottom 3 comparison page
    - Busiest person page (status breakdown)
    - Beautiful leaderboard page with crown for top performer
    - Final motivational message
    """
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
        Flowable,
        HRFlowable,
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    # Optional: register Poppins fonts if paths are provided
    if POPPINS_REGULAR_TTF or POPPINS_BOLD_TTF:
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            if POPPINS_REGULAR_TTF:
                pdfmetrics.registerFont(TTFont("Poppins", POPPINS_REGULAR_TTF))
            if POPPINS_BOLD_TTF:
                pdfmetrics.registerFont(TTFont("Poppins-Bold", POPPINS_BOLD_TTF))
        except Exception:
            # Fallback silently to default fonts if registration fails
            pass

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    # Base font name depending on whether Poppins is registered
    base_font = "Poppins" if "Poppins" in styles["Normal"].fontName or POPPINS_REGULAR_TTF else "Helvetica"
    base_bold = "Poppins-Bold" if POPPINS_BOLD_TTF else "Helvetica-Bold"

    styles.add(
        ParagraphStyle(
            name="EchtTitle",
            parent=styles["Heading1"],
            fontName=base_bold,
            fontSize=24,
            leading=28,
            textColor=colors.white,
            alignment=1,  # center
        )
    )
    styles.add(
        ParagraphStyle(
            name="EchtSubtitle",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#e5e7eb"),
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EchtSection",
            parent=styles["Heading2"],
            fontName=base_bold,
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=6,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Muted",
            parent=styles["BodyText"],
            fontName=base_font,
            textColor=colors.HexColor("#6b7280"),
            fontSize=8.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9,
            leading=11,
        )
    )

    teal = colors.HexColor("#19b3b1")
    teal_light = colors.HexColor("#67e8f9")
    dark_bg = colors.HexColor("#020617")
    dark_teal = colors.HexColor("#0b4b63")

    Story: List = []

    # ---------------------------------------
    # COVER PAGE
    # ---------------------------------------

    class CoverBanner(Flowable):
        def __init__(self, height=70 * mm):
            Flowable.__init__(self)
            self.height = height

        def draw(self):
            c = self.canv
            w = doc.width
            h = self.height
            # Background block
            c.saveState()
            c.setFillColor(dark_bg)
            c.rect(0, 0, w, h, fill=1, stroke=0)
            # Accent diagonal band
            c.setFillColor(teal)
            c.rect(0, 0, w * 0.65, h * 0.55, fill=1, stroke=0)
            c.setFillColor(dark_teal)
            c.rect(w * 0.35, h * 0.25, w * 0.65, h * 0.4, fill=1, stroke=0)
            c.restoreState()

    Story.append(CoverBanner())
    Story.append(Spacer(1, 10))

    Story.append(Paragraph("Echt • The Monitor", styles["EchtTitle"]))
    Story.append(Spacer(1, 4))
    Story.append(
        Paragraph(
            "Team Delivery & Discipline Report",
            styles["EchtSubtitle"],
        )
    )
    Story.append(Spacer(1, 10))

    Story.append(
        Paragraph(
            f"{timeframe_label}<br/>{generated_label}",
            styles["EchtSubtitle"],
        )
    )
    Story.append(Spacer(1, 12))

    # Logo on the cover
    if LOGO_PATH:
        try:
            img = Image(LOGO_PATH, width=32 * mm, height=32 * mm)
            img.hAlign = "CENTER"
            Story.append(img)
            Story.append(Spacer(1, 8))
        except Exception:
            pass

    Story.append(
        Paragraph(
            "This report highlights how tasks are being delivered, "
            "who is staying disciplined with timelines and communication, "
            "and where support or process tightening is needed.",
            styles["Small"],
        )
    )
    Story.append(PageBreak())

    # ---------------------------------------
    # TEAM OVERVIEW (with zero-task people)
    # ---------------------------------------

    Story.append(Paragraph("Team overview", styles["EchtSection"]))

    total_tasks = len(df_view)
    overdue_tasks = int(df_view["overdue"].sum())
    due_3 = int(
        (
            (df_view["days_to_due"].notnull())
            & (df_view["days_to_due"] >= 0)
            & (df_view["days_to_due"] <= 3)
        ).sum()
    )
    urgent = int(df_view["priority_lower"].eq("urgent").sum())
    total_changes = int(df_view["change_count"].sum())

    overview_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Tasks in scope", total_tasks, "Overdue tasks", overdue_tasks],
        ["Due in ≤3 days", due_3, "Urgent tasks", urgent],
        ["Total due-date changes", total_changes, "", ""],
    ]

    overview_tbl = Table(
        overview_data,
        colWidths=[55 * mm, 25 * mm, 55 * mm, 25 * mm],
    )
    overview_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), dark_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), base_bold),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTNAME", (0, 1), (-1, -1), base_font),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
            ]
        )
    )
    Story.append(overview_tbl)
    Story.append(Spacer(1, 10))

    # People with no tasks this period (from active_members)
    if active_members:
        active_set = set(active_members)
        with_tasks = set(df_view["assignee"].dropna().unique().tolist())
        zero_task_people = sorted(list(active_set - with_tasks))
    else:
        zero_task_people = []

    if zero_task_people:
        Story.append(
            Paragraph(
                "Team members without tasks in this reporting period:",
                styles["Small"],
            )
        )
        Story.append(
            Paragraph(
                ", ".join(zero_task_people),
                styles["Muted"],
            )
        )
        Story.append(Spacer(1, 6))

    # Single big progress bar
    Story.append(Paragraph("Progress towards 0 overdue tasks", styles["Small"]))
    Story.append(Spacer(1, 3))

    class BigProgressBar(Flowable):
        def __init__(self, width, height, overdue, total):
            Flowable.__init__(self)
            self.width = width
            self.height = height
            self.overdue = overdue
            self.total = max(total, 1)

        def draw(self):
            c = self.canv
            radius = self.height / 2.0
            c.setFillColor(colors.HexColor("#111827"))
            c.roundRect(0, 0, self.width, self.height, radius, stroke=0, fill=1)
            fraction_good = (self.total - self.overdue) / self.total
            good_width = max(0, fraction_good) * self.width
            c.setFillColor(teal)
            c.roundRect(0, 0, good_width, self.height, radius, stroke=0, fill=1)

    Story.append(BigProgressBar(doc.width, 8 * mm, overdue_tasks, total_tasks))
    Story.append(
        Paragraph(
            f"{overdue_tasks} overdue out of {total_tasks} task(s).",
            styles["Muted"],
        )
    )
    Story.append(PageBreak())

    # ---------------------------------------
    # UNANSWERED QUESTIONS PAGE
    # ---------------------------------------

    Story.append(Paragraph("Unanswered questions in ClickUp comments", styles["EchtSection"]))
    Story.append(
        Paragraph(
            "Only comments containing a question mark ('?') are treated as questions. "
            "Below are lists where questions are still waiting for a response from the assignee.",
            styles["Small"],
        )
    )
    Story.append(Spacer(1, 6))

    if unanswered_df is not None and not unanswered_df.empty:
        # We only show: assignee, list, question, date
        tmp = unanswered_df.copy()
        cols_needed = ["assignee", "list", "latest_comment_text", "latest_comment_date"]
        for c in cols_needed:
            if c not in tmp.columns:
                tmp[c] = None

        table_data = [["Assignee", "List", "Question", "Date"]]
        for _, row in tmp.sort_values("latest_comment_age_days", ascending=False).iterrows():
            assignee = str(row.get("assignee", "") or "")
            lst_name = str(row.get("list", "") or "")
            question = _truncate(row.get("latest_comment_text", ""), 100)
            q_date = str(row.get("latest_comment_date", "") or "")
            table_data.append([assignee, lst_name, question, q_date])

        q_tbl = Table(
            table_data,
            colWidths=[30 * mm, 35 * mm, 80 * mm, 25 * mm],
            repeatRows=1,
        )
        q_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), dark_bg),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), base_bold),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ]
            )
        )
        Story.append(q_tbl)
    else:
        Story.append(Paragraph("No unanswered questions for this period.", styles["Small"]))

    Story.append(PageBreak())

    # ---------------------------------------
    # PER-ASSIGNEE CARDS (MULTI-PER-PAGE)
    # ---------------------------------------

    Story.append(Paragraph("Per-assignee performance", styles["EchtSection"]))
    Story.append(
        Paragraph(
            "Each card summarises overdue tasks, unanswered questions, and due-date changes "
            "for a team member in this period.",
            styles["Small"],
        )
    )
    Story.append(Spacer(1, 4))

    per_df = people_summary.sort_values("assignee").copy()

    class HealthBar(Flowable):
        def __init__(self, width, height, rate):
            Flowable.__init__(self)
            self.width = width
            self.height = height
            self.rate = max(0.0, min(1.0, float(rate)))

        def draw(self):
            c = self.canv
            radius = self.height / 2.0
            c.setFillColor(colors.HexColor("#e5e7eb"))
            c.roundRect(0, 0, self.width, self.height, radius, stroke=0, fill=1)
            c.setFillColor(_rate_color(self.rate))
            c.roundRect(0, 0, self.width * self.rate, self.height, radius, stroke=0, fill=1)

    for idx, row in per_df.iterrows():
        a = row["assignee"]
        tasks = max(int(row["tasks_total"]), 0)
        overdue = int(row["overdue_tasks"])
        changes = int(row["change_total"])
        unanswered = int(row["unanswered_count"])
        overdue_rate = float(row["overdue_rate"])
        unanswered_rate = float(row["unanswered_rate"])

        note = person_notes.get(a, auto_feedback_for_person(row))

        Story.append(Paragraph(f"<b>{a}</b>", styles["Small"]))
        Story.append(
            Paragraph(
                f"{tasks} task(s) in this reporting period.",
                styles["Muted"],
            )
        )
        Story.append(Spacer(1, 2))

        # Overdue health bar
        Story.append(Paragraph("Overdue tasks", styles["Small"]))
        Story.append(HealthBar(doc.width, 5 * mm, overdue_rate))
        Story.append(
            Paragraph(
                f"{overdue} overdue task(s) • {overdue_rate:.0%} of their workload.",
                styles["Muted"],
            )
        )
        Story.append(Spacer(1, 2))

        # Unanswered health bar
        Story.append(Paragraph("Unanswered questions", styles["Small"]))
        Story.append(HealthBar(doc.width, 5 * mm, unanswered_rate))
        Story.append(
            Paragraph(
                f"{unanswered} unanswered question(s) • {unanswered_rate:.0%} of their tasks.",
                styles["Muted"],
            )
        )
        Story.append(Spacer(1, 2))

        Story.append(
            Paragraph(
                f"Due-date changes: <b>{changes}</b> in this period.",
                styles["Small"],
            )
        )
        Story.append(Spacer(1, 3))

        # Short list of this person's overdue tasks (with blue hyperlinks)
        person_overdue = df_view[
            (df_view["assignee"] == a) & (df_view["overdue"] == True)
        ].copy()

        if not person_overdue.empty:
            tbl_data = [["Client", "Task", "Due", "Days overdue", "Link"]]
            for _, trow in person_overdue.sort_values("due_date").iterrows():
                client = str(trow.get("client", "") or "")
                tname = _truncate(trow.get("task_name", ""), 50)
                due_val = trow.get("due_date", "")
                if isinstance(due_val, (pd.Timestamp, date)):
                    due_str = due_val.strftime("%Y-%m-%d")
                else:
                    due_str = str(due_val or "")
                if pd.notnull(trow["days_to_due"]) and trow["days_to_due"] < 0:
                    days_over = -int(trow["days_to_due"])
                else:
                    days_over = ""
                url = str(trow.get("url", "") or "")
                if url:
                    link_html = f'<link href="{url}" color="blue">Open</link>'
                else:
                    link_html = ""
                tbl_data.append([client, tname, due_str, f"{days_over}", link_html])

            od_tbl = Table(
                tbl_data,
                colWidths=[32 * mm, 60 * mm, 23 * mm, 23 * mm, 25 * mm],
            )
            od_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), dark_bg),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), base_bold),
                        ("FONTSIZE", (0, 0), (-1, 0), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
                        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#e5e7eb")),
                    ]
                )
            )
            Story.append(od_tbl)
            Story.append(Spacer(1, 3))

        # Manager note / summary
        Story.append(Paragraph("Summary & feedback", styles["Small"]))
        Story.append(Paragraph(_truncate(note, 400), styles["Small"]))
        Story.append(Spacer(1, 4))

        # Neat separator bar
        Story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor("#d1d5db"),
                spaceBefore=4,
                spaceAfter=8,
            )
        )

    # ---------------------------------------
    # TOP 3 VS BOTTOM 3 PAGE
    # ---------------------------------------

    Story.append(PageBreak())
    Story.append(Paragraph("Top 3 vs Bottom 3", styles["EchtSection"]))
    Story.append(
        Paragraph(
            "Top performers are those who keep work on time, avoid unnecessary due-date changes, "
            "and clear questions quickly. Bottom performers have the highest combination of overdue work, "
            "changes, and unanswered questions.",
            styles["Small"],
        )
    )
    Story.append(Spacer(1, 6))

    top3, bottom3 = _compute_top_and_bottom(people_summary)

    def _summary_table(df: pd.DataFrame, highlight_best: bool) -> Table:
        rows = [["Rank", "Assignee", "Overdue %", "Unanswered %", "Due-date changes"]]
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            overdue_pct = f"{r.get('overdue_rate', 0.0):.0%}"
            unanswered_pct = f"{r.get('unanswered_rate', 0.0):.0%}"
            changes = int(r.get("change_total", 0))
            rows.append(
                [
                    i,
                    str(r["assignee"]),
                    overdue_pct,
                    unanswered_pct,
                    f"{changes}",
                ]
            )

        tbl = Table(rows, colWidths=[12 * mm, 45 * mm, 30 * mm, 35 * mm, 35 * mm])
        if highlight_best:
            header_bg = colors.HexColor("#022c22")
        else:
            header_bg = colors.HexColor("#450a0a")

        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), base_bold),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ]
            )
        )
        return tbl

    col_left, col_right = [], []
    if not top3.empty:
        col_left.append(Paragraph("Top 3 (on-time & responsive)", styles["Small"]))
        col_left.append(_summary_table(top3, highlight_best=True))
    if not bottom3.empty:
        col_right.append(Paragraph("Bottom 3 (overdue & unresponsive)", styles["Small"]))
        col_right.append(_summary_table(bottom3, highlight_best=False))

    if col_left or col_right:
        # Use a small 2-column layout via a table of flowables
        wrapper = Table(
            [
                [col_left[0] if col_left else "", col_right[0] if col_right else ""],
                [col_left[1] if len(col_left) > 1 else "", col_right[1] if len(col_right) > 1 else ""],
            ],
            colWidths=[(doc.width / 2) - 5 * mm, (doc.width / 2) - 5 * mm],
        )
        wrapper.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        Story.append(wrapper)
    else:
        Story.append(Paragraph("Not enough data to compute top and bottom performers.", styles["Small"]))

    # ---------------------------------------
    # BUSIEST PERSON PAGE
    # ---------------------------------------

    if not people_summary.empty:
        busiest_row = people_summary.sort_values("tasks_total", ascending=False).iloc[0]
        busiest_name = busiest_row["assignee"]
        busiest_count = int(busiest_row["tasks_total"])

        Story.append(PageBreak())
        Story.append(Paragraph("Busiest assignee", styles["EchtSection"]))
        Story.append(
            Paragraph(
                f"{busiest_name} handled the highest number of tasks in this period.",
                styles["Small"],
            )
        )
        Story.append(Spacer(1, 4))

        Story.append(
            Paragraph(
                f"Total tasks: <b>{busiest_count}</b><br/>"
                f"Overdue tasks: <b>{int(busiest_row['overdue_tasks'])}</b><br/>"
                f"Due-date changes: <b>{int(busiest_row['change_total'])}</b><br/>"
                f"Unanswered questions: <b>{int(busiest_row['unanswered_count'])}</b>",
                styles["Small"],
            )
        )
        Story.append(Spacer(1, 8))

        # Status breakdown table
        status_df = _status_breakdown_for_person(df_view, busiest_name)
        if not status_df.empty:
            s_rows = [["Status", "Task count"]]
            for _, r in status_df.iterrows():
                s_rows.append([str(r["status"]), int(r["count"])])

            s_tbl = Table(
                s_rows,
                colWidths=[80 * mm, 30 * mm],
            )
            s_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), dark_bg),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), base_bold),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                    ]
                )
            )
            Story.append(Paragraph("Status breakdown", styles["Small"]))
            Story.append(s_tbl)
        else:
            Story.append(Paragraph("No detailed status breakdown available.", styles["Small"]))

    # ---------------------------------------
    # LEADERBOARD PAGE
    # ---------------------------------------

    Story.append(PageBreak())
    Story.append(Paragraph("Leaderboard", styles["EchtSection"]))
    Story.append(
        Paragraph(
            "Ranked from strongest overall discipline (top) to weakest (bottom). "
            "The top performer is crowned.",
            styles["Small"],
        )
    )
    Story.append(Spacer(1, 4))

    lb = people_summary.copy()
    if not lb.empty:
        lb["good_score"] = (
            (1.0 - lb["overdue_rate"]) * 0.5
            + (1.0 - lb["unanswered_rate"]) * 0.3
            + (1.0 - lb["change_rate"]) * 0.2
        )
        lb = lb.sort_values("good_score", ascending=False).reset_index(drop=True)

        rows = [["", "Assignee", "Score", "Overdue %", "Unanswered %", "Changes"]]
        for i, (_, r) in enumerate(lb.iterrows(), start=1):
            crown = "👑" if i == 1 else ""
            name = f"{crown} {r['assignee']}".strip()
            rows.append(
                [
                    i,
                    name,
                    f"{r['good_score']:.2f}",
                    f"{r['overdue_rate']:.0%}",
                    f"{r['unanswered_rate']:.0%}",
                    int(r["change_total"]),
                ]
            )

        lb_tbl = Table(
            rows,
            colWidths=[10 * mm, 55 * mm, 25 * mm, 25 * mm, 30 * mm, 25 * mm],
            repeatRows=1,
        )
        lb_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), dark_bg),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), base_bold),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ]
            )
        )
        Story.append(lb_tbl)
    else:
        Story.append(Paragraph("No data to build a leaderboard.", styles["Small"]))

    # ---------------------------------------
    # FINAL MESSAGE PAGE
    # ---------------------------------------

    Story.append(PageBreak())
    Story.append(Paragraph("Message from management", styles["EchtSection"]))

    if not motivation_text.strip():
        motivation_text = (
            "Use this page to add a short motivational note for the team before exporting. "
            "For example: appreciation for strong performers, clarity on expectations, "
            "and one or two key focus areas for the next period."
        )

    Story.append(Paragraph(_truncate(motivation_text, 1500), styles["Small"]))

    doc.build(Story)
    buffer.seek(0)
    return buffer.read()
# =======================================
# STREAMLIT APP – THE MONITOR
# =======================================

st.set_page_config(
    page_title="The Monitor",
    layout="wide",
)

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    :root {
        --echt-teal: #19b3b1;
        --echt-dark: #020617;
        --echt-bg: #020617;
    }

    .main {
        background-color: var(--echt-bg);
    }

    h1, h2, h3, h4, h5 {
        color: #ffffff;
    }

    .scorecard {
        background: #020617;
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.5);
    }
    .scorecard.bad1 { border-left: 6px solid #dc2626; }
    .scorecard.bad2 { border-left: 6px solid #fb923c; }
    .scorecard.bad3 { border-left: 6px solid #facc15; }
    .scorecard.bad4 { border-left: 6px solid #4ade80; }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ The Monitor")
st.caption(
    "Echt Ops dashboard to track overdue work, due-date changes, reassignments, unanswered questions, "
    "and generate beautiful PDF reports for the team."
)

# ---- Sidebar configuration ----
st.sidebar.header("🔐 ClickUp Connection")
st.sidebar.success("Using ClickUp API token and Team ID from the config section in the code.")

days_ahead = st.sidebar.slider(
    "Include tasks due within next X days",
    min_value=1,
    max_value=60,
    value=7,
)

include_comments = st.sidebar.checkbox(
    "Include comment metrics (slower, more API calls)",
    value=False,
    help="Only marks a question as pending if the latest comment with '?' has no reply from the assignee.",
)

task_scope = st.sidebar.selectbox(
    "Task scope",
    ["Open tasks only", "Include completed as well"],
    help="Use 'Include completed' for historical offender analysis.",
)

if "df_meeting" not in st.session_state:
    st.session_state["df_meeting"] = None
if "active_ids" not in st.session_state:
    st.session_state["active_ids"] = None
if "people_notes_df" not in st.session_state:
    st.session_state["people_notes_df"] = None
if "motivation_note" not in st.session_state:
    st.session_state["motivation_note"] = ""

fetch_button = st.sidebar.button("Fetch / Refresh tasks")


@st.cache_data(show_spinner=True)
def fetch_and_prepare(
    team_id: str,
    days_ahead: int,
    include_comments: bool,
) -> Tuple[pd.DataFrame, Set[str]]:
    """Fetch all tasks from ClickUp and enrich with metrics."""
    df_raw = fetch_all_tasks_for_team(team_id)
    if df_raw.empty:
        return df_raw, set()

    active_ids = fetch_active_member_identifiers()

    df = df_raw.copy()
    df = add_priority_score(df)
    df = add_due_info(df)
    df = add_completion_flag(df)
    df = add_meeting_bucket(df)
    df = filter_meeting_items(df, days_ahead=days_ahead)
    df = update_due_change_metrics(df)
    df = update_assignment_metrics(df)

    if include_comments:
        df = add_comment_metrics(df)

    # Filter to active members only (if we managed to fetch them)
    if active_ids:
        df = df[df["assignee"].isna() | df["assignee"].isin(active_ids)]

    return df, active_ids


if fetch_button:
    try:
        with st.spinner("Fetching tasks from ClickUp…"):
            df_meeting, active_ids = fetch_and_prepare(
                TEAM_ID,
                days_ahead=days_ahead,
                include_comments=include_comments,
            )
        st.session_state["df_meeting"] = df_meeting
        st.session_state["active_ids"] = active_ids
        st.session_state["people_notes_df"] = None  # reset notes when refreshing
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        raise

df_meeting = st.session_state["df_meeting"]
active_ids = st.session_state["active_ids"]

if df_meeting is None or df_meeting.empty:
    st.info("Click **Fetch / Refresh tasks** in the sidebar to load data from ClickUp.")
    st.stop()

# ---- Apply scope (open vs completed) ----
if task_scope == "Open tasks only":
    df_scope = df_meeting[~df_meeting["is_completed"]].copy()
else:
    df_scope = df_meeting.copy()

st.success(f"Loaded {len(df_scope)} tasks in scope with current filters from ClickUp.")

# ---- Filter controls ----
clients = sorted(df_scope["client"].dropna().unique().tolist())

if active_ids:
    assignees_all = [
        a for a in df_scope["assignee"].dropna().unique().tolist()
        if a in active_ids
    ]
else:
    assignees_all = df_scope["assignee"].dropna().unique().tolist()

assignees = sorted(assignees_all)
buckets = sorted(df_scope["meeting_bucket"].dropna().unique().tolist())

with st.container():
    fc1, fc2, fc3, fc4 = st.columns([1.4, 1.4, 1.1, 1.1])
    with fc1:
        selected_clients = st.multiselect(
            "Client (Space)",
            options=clients,
            default=clients,
        )
    with fc2:
        selected_assignees = st.multiselect(
            "Assignee",
            options=assignees,
            default=assignees,
        )
    with fc3:
        selected_buckets = st.multiselect(
            "Urgency bucket",
            options=buckets,
            default=buckets,
        )
    with fc4:
        max_changes = int(df_scope["change_count"].max()) if "change_count" in df_scope.columns else 0
        if max_changes > 0:
            min_changes = st.slider(
                "Min due-date changes",
                min_value=0,
                max_value=max_changes,
                value=0,
            )
        else:
            min_changes = 0
            st.caption("No due-date changes yet – filter disabled.")

df_view = df_scope.copy()
if selected_clients:
    df_view = df_view[df_view["client"].isin(selected_clients)]
if selected_assignees:
    df_view = df_view[
        df_view["assignee"].isin(selected_assignees)
        | df_view["assignee"].isna()
    ]
if selected_buckets:
    df_view = df_view[df_view["meeting_bucket"].isin(selected_buckets)]
if min_changes > 0 and "change_count" in df_view.columns:
    df_view = df_view[df_view["change_count"] >= min_changes]

if df_view.empty:
    st.warning("No tasks remain after applying the filters.")
    st.stop()

# ---- Top metric strip ----
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown('<div class="metric-label">TASKS IN VIEW</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(df_view)}</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-label">OVERDUE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{int(df_view["overdue"].sum())}</div>', unsafe_allow_html=True)
with m3:
    due3 = int(
        (
            (df_view["days_to_due"].notnull())
            & (df_view["days_to_due"] <= 3)
            & (df_view["days_to_due"] >= 0)
        ).sum()
    )
    st.markdown('<div class="metric-label">DUE IN ≤3 DAYS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{due3}</div>', unsafe_allow_html=True)
with m4:
    urgent_count = int(df_view["priority_lower"].eq("urgent").sum())
    st.markdown('<div class="metric-label">URGENT TASKS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{urgent_count}</div>', unsafe_allow_html=True)
with m5:
    st.markdown('<div class="metric-label">DUE-DATE CHANGES</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{int(df_view["change_count"].sum())}</div>', unsafe_allow_html=True)

st.markdown("---")

# ---- Tabs ----
tab_overview, tab_offenders, tab_charts, tab_comments, tab_pdf = st.tabs(
    ["Overview table", "Offenders & reassignments", "Charts", "Comments & questions", "PDF report"]
)

# =======================================
# TAB 1: OVERVIEW TABLE
# =======================================

with tab_overview:
    st.subheader("Tasks to discuss in the Monday meeting")

    display_cols = [
        "client",
        "list",
        "task_name",
        "assignee",
        "priority",
        "meeting_bucket",
        "due_date",
        "status",
        "is_completed",
        "change_count",
        "reassign_count",
        "url",
    ]
    if include_comments:
        display_cols.extend(
            [
                "latest_comment_author",
                "latest_comment_age_days",
                "comment_pending_for_assignee",
                "latest_comment_text",
            ]
        )
    display_cols = [c for c in display_cols if c in df_view.columns]

    df_display = df_view[display_cols].reset_index(drop=True)
    st.dataframe(df_display, use_container_width=True)

    excel_data = to_excel_bytes(df_display)
    st.download_button(
        label="⬇️ Download XLSX report",
        data=excel_data,
        file_name="the_monitor_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        label="⬇️ Download XLS (legacy) report",
        data=excel_data,
        file_name="the_monitor_report.xls",
        mime="application/vnd.ms-excel",
    )

# =======================================
# TAB 2: OFFENDERS & REASSIGNMENTS
# =======================================

with tab_offenders:
    st.subheader("Due-date change offenders")

    if "change_count" not in df_scope.columns or df_scope.empty:
        st.info("No due-date change data yet. This will build up over time as The Monitor runs.")
    else:
        offenders = (
            df_scope.groupby("assignee", dropna=True)
            .agg(
                total_changes=("change_count", "sum"),
                tasks=("task_id", "nunique"),
                overdue_tasks=("overdue", "sum"),
            )
            .reset_index()
        )
        offenders = offenders[offenders["assignee"].notnull()]
        offenders = offenders.sort_values(
            by=["total_changes", "overdue_tasks", "tasks"],
            ascending=[False, False, False],
        )

        st.caption("Sorted by total due-date changes, then overdue tasks, then total tasks.")

        top_n = offenders.head(10)
        for idx, row in top_n.iterrows():
            rank = top_n.index.get_loc(idx) + 1
            if rank == 1:
                cls = "bad1"
            elif rank <= 3:
                cls = "bad2"
            elif rank <= 5:
                cls = "bad3"
            else:
                cls = "bad4"

            st.markdown(
                f"""
                <div class="scorecard {cls}">
                    <div style="font-size:0.8rem; opacity:0.7;">#{rank} • {row['assignee']}</div>
                    <div style="font-size:1.25rem; font-weight:600;">{int(row['total_changes'])} due-date changes</div>
                    <div style="font-size:0.8rem; opacity:0.85;">
                        {int(row['tasks'])} tasks in scope • {int(row['overdue_tasks'])} currently overdue
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            person_tasks = df_scope[df_scope["assignee"] == row["assignee"]]
            with st.expander(f"View tasks for {row['assignee']} ({len(person_tasks)} in scope)"):
                if person_tasks.empty:
                    st.write("No tasks for this assignee under the current filters.")
                else:
                    cols = [
                        "client",
                        "list",
                        "task_name",
                        "due_date",
                        "overdue",
                        "change_count",
                        "reassign_count",
                        "url",
                    ]
                    cols = [c for c in cols if c in person_tasks.columns]
                    st.dataframe(
                        person_tasks[cols].reset_index(drop=True),
                        use_container_width=True,
                    )

    st.markdown("### Reassignment behaviour (who throws / who gets dumped on)")

    throwers, receivers = get_reassignment_offenders()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**People who REASSIGN tasks (throw work away)**")
        if throwers.empty:
            st.info("No reassignment events logged yet.")
        else:
            throwers = throwers.sort_values(
                by=["last_minute_events", "reassign_events"],
                ascending=[False, False],
            )
            top_t = throwers.head(10)
            for idx, row in top_t.iterrows():
                rank = top_t.index.get_loc(idx) + 1
                if rank == 1:
                    cls = "bad1"
                elif rank <= 3:
                    cls = "bad2"
                elif rank <= 5:
                    cls = "bad3"
                else:
                    cls = "bad4"
                st.markdown(
                    f"""
                    <div class="scorecard {cls}">
                        <div style="font-size:0.8rem; opacity:0.7;">#{rank} • {row['assignee']}</div>
                        <div style="font-size:1.1rem; font-weight:600;">{int(row['reassign_events'])} reassignments</div>
                        <div style="font-size:0.8rem; opacity:0.85;">
                            {int(row['last_minute_events'])} last-minute
                            (≤ {int(LAST_MINUTE_THRESHOLD_HOURS)} hours to due date)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with c2:
        st.markdown("**People who RECEIVE last-minute tasks (get dumped on)**")
        if receivers.empty:
            st.info("No reassignment events logged yet.")
        else:
            receivers = receivers.sort_values(
                by=["last_minute_received", "tasks_received"],
                ascending=[False, False],
            )
            top_r = receivers.head(10)
            for idx, row in top_r.iterrows():
                rank = top_r.index.get_loc(idx) + 1
                if rank == 1:
                    cls = "bad1"
                elif rank <= 3:
                    cls = "bad2"
                elif rank <= 5:
                    cls = "bad3"
                else:
                    cls = "bad4"
                st.markdown(
                    f"""
                    <div class="scorecard {cls}">
                        <div style="font-size:0.8rem; opacity:0.7;">#{rank} • {row['assignee']}</div>
                        <div style="font-size:1.1rem; font-weight:600;">{int(row['tasks_received'])} tasks received</div>
                        <div style="font-size:0.8rem; opacity:0.85;">
                            {int(row['last_minute_received'])} last-minute
                            (≤ {int(LAST_MINUTE_THRESHOLD_HOURS)} hours to due date)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# =======================================
# TAB 3: CHARTS
# =======================================

with tab_charts:
    st.subheader("Charts for decision-making")

    if df_scope.empty:
        st.info("No data available for charts.")
    else:
        chart_type = st.selectbox(
            "Chart type",
            ["Overdue tasks by assignee", "Due-date changes by assignee"],
        )
        chart_style = st.selectbox(
            "Chart style",
            ["Bar", "Pie / Donut", "Line"],
        )

        if chart_type == "Overdue tasks by assignee":
            data = (
                df_scope[df_scope["overdue"] == True]
                .groupby("assignee", dropna=True)["task_id"]
                .count()
                .reset_index(name="value")
            )
            y_label = "Overdue tasks"
            title = "Overdue tasks by assignee"
        else:
            if "change_count" not in df_scope.columns:
                st.info("No due-date change data yet.")
                data = pd.DataFrame(columns=["assignee", "value"])
                y_label = "Due-date changes"
                title = "Due-date changes by assignee"
            else:
                data = (
                    df_scope.groupby("assignee", dropna=True)["change_count"]
                    .sum()
                    .reset_index(name="value")
                )
                y_label = "Due-date changes"
                title = "Due-date changes by assignee"

        if data.empty:
            st.info("No data for this chart selection.")
        else:
            if chart_style == "Bar":
                fig = px.bar(
                    data,
                    x="assignee",
                    y="value",
                    title=title,
                    color="value",
                    color_continuous_scale=["#19b3b1", "#0b4b63"],
                )
                fig.update_layout(
                    xaxis_title="Assignee",
                    yaxis_title=y_label,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
            elif chart_style == "Pie / Donut":
                fig = px.pie(
                    data,
                    names="assignee",
                    values="value",
                    title=title,
                    hole=0.4,
                    color_discrete_sequence=["#19b3b1", "#0b4b63", "#67e8f9", "#0f172a"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
            else:
                fig = px.line(
                    data,
                    x="assignee",
                    y="value",
                    title=title,
                )
                fig.update_traces(line=dict(width=3))
                fig.update_layout(
                    xaxis_title="Assignee",
                    yaxis_title=y_label,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

            st.plotly_chart(fig, use_container_width=True)

# =======================================
# TAB 4: COMMENTS & QUESTIONS
# =======================================

with tab_comments:
    st.subheader("Unanswered questions in ClickUp comments")

    if not include_comments:
        st.info("Turn on **Include comment metrics** in the sidebar and refresh to see unanswered questions.")
    else:
        if "comment_pending_for_assignee" not in df_scope.columns:
            st.info("No comment metrics available yet.")
        else:
            pending = df_scope[
                (df_scope["comment_pending_for_assignee"] == True)
                & df_scope["latest_comment_text"].notnull()
            ].copy()

            if pending.empty:
                st.success("No outstanding questions (with '?') waiting for assignee replies in this scope.")
            else:
                st.caption(
                    "Only comments containing a '?' are considered questions. "
                    "Below you can see who has the most unanswered questions and how old they are."
                )

                summary = (
                    pending[pending["assignee"].notnull()]
                    .groupby("assignee")
                    .agg(
                        unanswered_count=("task_id", "nunique"),
                        avg_age_days=("latest_comment_age_days", "mean"),
                    )
                    .reset_index()
                )

                if not summary.empty:
                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown("**Unanswered questions by assignee**")
                        fig1 = px.bar(
                            summary.sort_values("unanswered_count", ascending=False),
                            x="assignee",
                            y="unanswered_count",
                            title="Number of unanswered questions",
                            color="unanswered_count",
                            color_continuous_scale=["#19b3b1", "#0b4b63"],
                        )
                        fig1.update_layout(
                            xaxis_title="Assignee",
                            yaxis_title="Unanswered questions",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                    with c2:
                        st.markdown("**Average age of unanswered questions (days)**")
                        fig2 = px.bar(
                            summary.sort_values("avg_age_days", ascending=False),
                            x="assignee",
                            y="avg_age_days",
                            title="Average age of unanswered questions",
                            color="avg_age_days",
                            color_continuous_scale=["#19b3b1", "#0b4b63"],
                        )
                        fig2.update_layout(
                            xaxis_title="Assignee",
                            yaxis_title="Average age (days)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                cols = [
                    "client",
                    "list",
                    "task_name",
                    "assignee",
                    "latest_comment_author",
                    "latest_comment_age_days",
                    "latest_comment_text",
                    "latest_comment_date",
                    "url",
                ]
                cols = [c for c in cols if c in pending.columns]
                st.dataframe(
                    pending[cols].reset_index(drop=True),
                    use_container_width=True,
                )

# =======================================
# TAB 5: PDF REPORT
# =======================================

with tab_pdf:
    st.subheader("PDF report – preview & export")

    # Timeframe for the report
    if df_view["due_date"].notnull().any():
        start_date = df_view["due_date"].min()
        end_date = df_view["due_date"].max()
        timeframe_label = f"Reporting period: {start_date:%d %b %Y} – {end_date:%d %b %Y}"
    else:
        timeframe_label = "Reporting period: based on current filters (no due dates available)"

    generated_label = f"Generated on: {datetime.now():%d %b %Y, %H:%M}"

    st.info(f"{timeframe_label} | {generated_label}")

    people_summary = build_people_summary(df_view)

    if people_summary.empty:
        st.warning("No assignees found to include in the PDF report.")
    else:
        people_summary["auto_feedback"] = people_summary.apply(auto_feedback_for_person, axis=1)

        # Build / sync editable manager notes
        if st.session_state["people_notes_df"] is None:
            editable = people_summary[
                [
                    "assignee",
                    "tasks_total",
                    "overdue_tasks",
                    "change_total",
                    "unanswered_count",
                    "auto_feedback",
                ]
            ].copy()
            editable["manager_note"] = editable["auto_feedback"]
            st.session_state["people_notes_df"] = editable
        else:
            existing = st.session_state["people_notes_df"]
            merged = people_summary[
                [
                    "assignee",
                    "tasks_total",
                    "overdue_tasks",
                    "change_total",
                    "unanswered_count",
                    "auto_feedback",
                ]
            ].merge(
                existing[["assignee", "manager_note"]],
                on="assignee",
                how="left",
            )
            merged["manager_note"] = merged["manager_note"].fillna(merged["auto_feedback"])
            st.session_state["people_notes_df"] = merged

        st.markdown("### Per-person feedback (for the PDF)")
        st.caption("Edit the manager notes if you want to customise feedback before generating the PDF.")

        edited = st.data_editor(
            st.session_state["people_notes_df"],
            hide_index=True,
            use_container_width=True,
            column_config={
                "auto_feedback": st.column_config.Column("Auto feedback (read only)", disabled=True),
                "manager_note": st.column_config.Column("Manager note (editable)"),
            },
        )
        st.session_state["people_notes_df"] = edited

        st.markdown("### Final motivational message")
        st.session_state["motivation_note"] = st.text_area(
            "Message to include on the last page:",
            value=st.session_state["motivation_note"],
            height=140,
            placeholder="Great work team – here’s what we’re doing well and what to focus on next…",
        )

        # unanswered_df for PDF (list & comment level)
        if include_comments and "comment_pending_for_assignee" in df_scope.columns:
            unanswered_df = df_scope[
                (df_scope["comment_pending_for_assignee"] == True)
                & df_scope["latest_comment_text"].notnull()
            ].copy()
        else:
            unanswered_df = None

        # Active member identifiers for "no tasks" listing
        active_member_set: Optional[Set[str]] = None
        if active_ids:
            active_member_set = set(active_ids)

        if st.button("Generate PDF report"):
            notes_df = st.session_state["people_notes_df"]
            person_notes = {
                row["assignee"]: row["manager_note"]
                for _, row in notes_df.iterrows()
            }

            pdf_bytes = build_pdf_report(
                df_view=df_view,
                people_summary=people_summary,
                person_notes=person_notes,
                motivation_text=st.session_state["motivation_note"],
                timeframe_label=timeframe_label,
                generated_label=generated_label,
                unanswered_df=unanswered_df,
                active_members=active_member_set,
            )

            st.download_button(
                label="📄 Download team PDF report",
                data=pdf_bytes,
                file_name="the_monitor_team_report.pdf",
                mime="application/pdf",
            )

"""
storage.py — Persist and deduplicate researcher news items.

Uses the ResearcherItem and ResearcherBrief tables from database.py.
All writes go through DB_WRITE_LOCK for SQLite safety.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone

from database import (
    DB_WRITE_LOCK, SessionLocal,
    ResearcherItem, ResearcherBrief,
)
from researcher.sources import NewsItem

logger = logging.getLogger(__name__)


def _headline_hash(headline: str) -> str:
    return hashlib.sha256(headline.strip().lower().encode()).hexdigest()[:16]


def ingest_items(items: list[NewsItem], game_date: str) -> int:
    """Store news items, deduplicating on (source, headline_hash).

    Returns count of NEW items inserted (skips duplicates).
    """
    inserted = 0
    with DB_WRITE_LOCK:
        session = SessionLocal()
        try:
            for item in items:
                h = _headline_hash(item.headline)
                exists = (
                    session.query(ResearcherItem.id)
                    .filter_by(source=item.source, headline_hash=h)
                    .first()
                )
                if exists:
                    continue

                row = ResearcherItem(
                    headline=item.headline,
                    headline_hash=h,
                    source=item.source,
                    url=item.url,
                    timestamp=item.timestamp,
                    body=item.body[:2000] if item.body else None,
                    players=json.dumps(item.players) if item.players else None,
                    teams=json.dumps(item.teams) if item.teams else None,
                    category=item.category,
                    game_date=game_date,
                )
                session.add(row)
                inserted += 1

            session.commit()
        except Exception:
            session.rollback()
            logger.exception("[researcher] ingest_items failed")
            raise
        finally:
            session.close()

    logger.info("[researcher] Ingested %d new items (of %d total)", inserted, len(items))
    return inserted


def get_items_for_date(game_date: str) -> list[dict]:
    """Retrieve all researcher items for a given date, newest first."""
    session = SessionLocal()
    try:
        rows = (
            session.query(ResearcherItem)
            .filter_by(game_date=game_date)
            .order_by(ResearcherItem.timestamp.desc())
            .all()
        )
        return [
            {
                "headline": r.headline,
                "source": r.source,
                "url": r.url,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "body": r.body,
                "players": json.loads(r.players) if r.players else [],
                "teams": json.loads(r.teams) if r.teams else [],
                "category": r.category,
            }
            for r in rows
        ]
    finally:
        session.close()


def save_brief(game_date: str, brief_type: str, content: str,
               model_feed: str = None, item_count: int = 0,
               token_cost: int = 0) -> None:
    """Save a synthesized brief (upserts on game_date + brief_type)."""
    with DB_WRITE_LOCK:
        session = SessionLocal()
        try:
            existing = (
                session.query(ResearcherBrief)
                .filter_by(game_date=game_date, brief_type=brief_type)
                .first()
            )
            if existing:
                existing.content = content
                existing.model_feed = model_feed
                existing.item_count = item_count
                existing.token_cost = token_cost
                existing.created_at = datetime.utcnow()
            else:
                session.add(ResearcherBrief(
                    game_date=game_date,
                    brief_type=brief_type,
                    content=content,
                    model_feed=model_feed,
                    item_count=item_count,
                    token_cost=token_cost,
                ))
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("[researcher] save_brief failed")
            raise
        finally:
            session.close()


def get_brief(game_date: str, brief_type: str = "daily") -> dict | None:
    """Retrieve the latest brief for a date and type."""
    session = SessionLocal()
    try:
        row = (
            session.query(ResearcherBrief)
            .filter_by(game_date=game_date, brief_type=brief_type)
            .first()
        )
        if not row:
            return None
        return {
            "game_date": row.game_date,
            "brief_type": row.brief_type,
            "content": row.content,
            "model_feed": row.model_feed,
            "item_count": row.item_count,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    finally:
        session.close()


def get_item_counts_by_source(game_date: str) -> dict[str, int]:
    """Count items per source for a given date."""
    session = SessionLocal()
    try:
        from sqlalchemy import func
        rows = (
            session.query(ResearcherItem.source, func.count(ResearcherItem.id))
            .filter_by(game_date=game_date)
            .group_by(ResearcherItem.source)
            .all()
        )
        return {source: count for source, count in rows}
    finally:
        session.close()

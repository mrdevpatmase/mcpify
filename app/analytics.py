import json
import os
from typing import Any, Dict

_client = None


def _get_client():
    """Lazy singleton, same pattern as app/db.py's engine - only actually
    builds credentials/client on first real use, so importing this module
    (and running the test suite) never needs GA configured."""
    global _client
    if _client is not None:
        return _client

    creds_json = os.getenv("GOOGLE_ANALYTICS_CREDENTIALS_JSON")
    if not creds_json:
        raise RuntimeError(
            "GOOGLE_ANALYTICS_CREDENTIALS_JSON is not set - paste the full service "
            "account JSON key (Viewer access on the GA4 property) as this env var."
        )

    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.oauth2 import service_account

    info = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/analytics.readonly"]
    )
    _client = BetaAnalyticsDataClient(credentials=credentials)
    return _client


def get_ga_summary(days: int = 30) -> Dict[str, Any]:
    """
    Runs two GA4 Data API reports over the trailing `days` days: overall
    totals (active users, page views, sessions) and a per-page breakdown
    (top 10 by views). Synchronous (the google-analytics-data client has
    no asyncio support) - callers should run this in a thread
    (asyncio.to_thread) rather than call it directly from an async
    handler.
    """
    from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, OrderBy, RunReportRequest

    property_id = os.getenv("GA_PROPERTY_ID")
    if not property_id:
        raise RuntimeError("GA_PROPERTY_ID is not set - the numeric GA4 property id (Admin > Property Settings).")

    client = _get_client()
    date_range = DateRange(start_date=f"{days}daysAgo", end_date="today")

    totals_response = client.run_report(
        RunReportRequest(
            property=f"properties/{property_id}",
            date_ranges=[date_range],
            metrics=[Metric(name="activeUsers"), Metric(name="screenPageViews"), Metric(name="sessions")],
        )
    )
    totals_row = totals_response.rows[0] if totals_response.rows else None
    active_users = int(totals_row.metric_values[0].value) if totals_row else 0
    page_views = int(totals_row.metric_values[1].value) if totals_row else 0
    sessions = int(totals_row.metric_values[2].value) if totals_row else 0

    pages_response = client.run_report(
        RunReportRequest(
            property=f"properties/{property_id}",
            date_ranges=[date_range],
            dimensions=[Dimension(name="pagePath")],
            metrics=[Metric(name="screenPageViews")],
            order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)],
            limit=10,
        )
    )
    top_pages = [
        {"page": row.dimension_values[0].value, "views": int(row.metric_values[0].value)}
        for row in pages_response.rows
    ]

    return {
        "period_days": days,
        "active_users": active_users,
        "page_views": page_views,
        "sessions": sessions,
        "top_pages": top_pages,
    }

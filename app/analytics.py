import json
import os
from typing import Any, Dict


def _get_client():
    """
    Builds credentials/client fresh on every call - NOT cached as a
    module-level singleton, deliberately: a client built from a
    wrong-but-structurally-complete credential (e.g. a stale/revoked
    refresh token) would otherwise get cached on the first failing call
    and keep failing identically for the rest of the process's life,
    surviving even after the env var is fixed, until a restart/redeploy.
    Rebuilding here is cheap - it only wraps credentials, no network
    call happens until run_report() actually executes - and this is a
    low-traffic, rate-limited admin-only endpoint, so there's no real
    cost to paying it on every call.

    Two auth paths, tried in this order:
    1. OAuth refresh token (GA_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN) - the
       one actually in use here, because this Google Cloud org enforces
       iam.disableServiceAccountKeyCreation, which blocks the usual
       service-account-JSON approach entirely (key creation itself fails,
       org-policy-admin-only to lift). A refresh token isn't a service
       account key, so it isn't affected by that constraint - obtained
       once via a local InstalledAppFlow consent run, then reusable
       indefinitely (refresh tokens don't expire unless revoked).
    2. Service account JSON (GOOGLE_ANALYTICS_CREDENTIALS_JSON) - kept as
       an alternative for any deployment where service account keys
       aren't blocked by org policy; simpler if it's available.
    """
    from google.analytics.data_v1beta import BetaAnalyticsDataClient

    client_id = os.getenv("GA_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GA_OAUTH_CLIENT_SECRET")
    refresh_token = os.getenv("GA_OAUTH_REFRESH_TOKEN")
    if client_id and client_secret and refresh_token:
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/analytics.readonly"],
        )
        return BetaAnalyticsDataClient(credentials=credentials)

    creds_json = os.getenv("GOOGLE_ANALYTICS_CREDENTIALS_JSON")
    if creds_json:
        from google.oauth2 import service_account

        info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/analytics.readonly"]
        )
        return BetaAnalyticsDataClient(credentials=credentials)

    raise RuntimeError(
        "No GA credentials configured - set GA_OAUTH_CLIENT_ID/GA_OAUTH_CLIENT_SECRET/"
        "GA_OAUTH_REFRESH_TOKEN, or GOOGLE_ANALYTICS_CREDENTIALS_JSON."
    )


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

"""
IBVAP local alert dispatch.

Default behavior is in-memory and local. The optional webhook is disabled unless
an operator explicitly configures it. Do not send alerts to unapproved endpoints.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict
from typing import Callable, Deque, Dict, Optional

from activity_engine import ActivityAlert


class AlertDispatcher:
    def __init__(self, max_items: int = 500, webhook_url: str = ""):
        self.history: Deque[Dict[str, object]] = deque(maxlen=max_items)
        self.webhook_url = webhook_url.strip()
        self.last_sent: Dict[str, float] = {}

    def dispatch(self, alert: ActivityAlert, ui_sink: Optional[Callable[[Dict[str, object]], None]] = None) -> Dict[str, object]:
        payload = alert.as_dict()
        payload["received_at"] = time.time()
        self.history.appendleft(payload)
        if ui_sink:
            ui_sink(payload)
        if self.webhook_url:
            self._post_webhook(payload)
        return payload

    def _post_webhook(self, payload: Dict[str, object]) -> None:
        # Optional integration, intentionally isolated from the detector.
        # Use a short timeout and never let notification failure stop inference.
        try:
            import requests
            requests.post(self.webhook_url, json=payload, timeout=2.0)
        except Exception:
            pass

    def export_json(self) -> str:
        return json.dumps(list(self.history), indent=2, default=str)

    def clear(self) -> None:
        self.history.clear()


def streamlit_alert_sink(payload: Dict[str, object]) -> None:
    """Render a notification in the existing Streamlit session."""
    import streamlit as st
    level = str(payload.get("level", "AMBER"))
    text = f"{payload.get('kind')} · {payload.get('camera_id')} · {payload.get('reason')}"
    if level == "RED":
        st.error(text)
    elif level == "AMBER":
        st.warning(text)
    else:
        st.info(text)


# engine/feature_store.py

import redis
import time
import threading
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import REDIS_HOST, REDIS_PORT, REDIS_DB


class RealTimeFeatureStore:
    """
    Tracks transaction timestamps and computes velocity metrics.

    Uses Redis when available and an in-memory fallback otherwise.
    The in-memory fallback lasts only until the API process stops.
    """

    def __init__(self):
        self.client = None
        self._lock = threading.Lock()
        self._memory_history = defaultdict(list)

        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            self.client = client
            print("[+] Connected to Redis Feature Store successfully.")

        except (redis.RedisError, OSError) as exc:
            print(
                "[!] Redis unavailable. Using in-memory transaction "
                f"history for this API process. Details: {exc}"
            )

    def update_and_get_velocity(
        self,
        card_id: str,
        current_timestamp: float,
    ) -> dict:
        """
        Record the current transaction and return counts, including
        the current transaction, within the last minute and hour.
        """

        now = float(current_timestamp)
        key = f"card_txns:{card_id}"

        if self.client is not None:
            try:
                # Use a unique member for every request, even when
                # multiple transactions share the same timestamp.
                member = f"{now:.9f}:{time.time_ns()}"

                self.client.zadd(key, {member: now})

                day_ago = now - 86400
                self.client.zremrangebyscore(key, "-inf", day_ago)

                count_1m = self.client.zcount(key, now - 60, now)
                count_1h = self.client.zcount(key, now - 3600, now)

                self.client.expire(key, 86400)

                return {
                    "velocity_1m": int(count_1m),
                    "velocity_1h": int(count_1h),
                    "history_backend": "redis",
                }

            except redis.RedisError as exc:
                print(
                    "[!] Redis operation failed. Using in-memory "
                    f"history for this request. Details: {exc}"
                )

        # In-memory fallback: process-local, not shared across workers.
        with self._lock:
            history = self._memory_history[card_id]

            # Retain recent timestamps and add this transaction.
            history[:] = [
                ts for ts in history
                if now - 86400 <= ts <= now
            ]
            history.append(now)

            count_1m = sum(1 for ts in history if now - 60 <= ts <= now)
            count_1h = sum(1 for ts in history if now - 3600 <= ts <= now)

            return {
                "velocity_1m": int(count_1m),
                "velocity_1h": int(count_1h),
                "history_backend": "memory",
            }


if __name__ == "__main__":
    store = RealTimeFeatureStore()
    card_id = "CARD_TEST"
    now = time.time()

    for offset in (0, 10, 20, 30):
        metrics = store.update_and_get_velocity(
            card_id=card_id,
            current_timestamp=now + offset,
        )
        print(f"Offset={offset}s: {metrics}")


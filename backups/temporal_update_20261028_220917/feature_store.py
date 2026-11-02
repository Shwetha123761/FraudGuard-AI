# engine/feature_store.py
import redis
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import REDIS_HOST, REDIS_PORT, REDIS_DB

class RealTimeFeatureStore:
    """
    Manages real-time feature computation and caching using Redis 
    for fast transaction velocity lookups.
    """
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST, 
                port=REDIS_PORT, 
                db=REDIS_DB, 
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            print("[+] Connected to Redis Feature Store successfully.")
        except redis.ConnectionError:
            print("[!] Warning: Could not connect to Redis server. Falling back to mock memory storage mode.")
            self.client = None

    def update_and_get_velocity(self, card_id: str, current_timestamp: float) -> dict:
        """
        Updates transaction history for a card and computes sliding-window velocities:
        - 1-minute velocity
        - 1-hour velocity
        """
        if not self.client:
            # Fallback mock metrics if Redis is not running locally yet
            return {"velocity_1m": 1, "velocity_1h": 1}

        key = f"card_txns:{card_id}"
        
        # Add current transaction timestamp to a Redis sorted set
        self.client.zadd(key, {str(current_timestamp): current_timestamp})
        
        # Clean up history older than 24 hours to save memory
        day_ago = current_timestamp - 86400
        self.client.zremrangebyscore(key, 0, day_ago)
        
        # Count transactions in sliding windows
        now = current_timestamp
        min_ago = now - 60
        hour_ago = now - 3600

        count_1m = self.client.zcount(key, min_ago, now)
        count_1h = self.client.zcount(key, hour_ago, now)

        # Set expiration on the key (expires after 24h of inactivity)
        self.client.expire(key, 86400)

        return {
            "velocity_1m": int(count_1m),
            "velocity_1h": int(count_1h)
        }

if __name__ == "__main__":
    store = RealTimeFeatureStore()
    # Test feature generation simulation
    mock_card = "CARD_9999"
    current_time = time.time()
    metrics = store.update_and_get_velocity(mock_card, current_time)
    print(f"[*] Test Velocity Metrics for {mock_card}: {metrics}")
import redis
import json
import yaml

def get_redis_client():
    # Load from config, defaults
    return redis.Redis(host='localhost', port=6379, db=0)

def cache_set(client, key, value, ttl=3600):
    client.setex(key, ttl, json.dumps(value))

def cache_get(client, key):
    val = client.get(key)
    if val:
        return json.loads(val)
    return None

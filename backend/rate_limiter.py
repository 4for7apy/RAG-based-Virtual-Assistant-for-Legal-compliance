"""
Rate limiter decorator - Basic placeholder
"""

def rate_limit(requests_per_minute: int = 60):
    """Rate limiting decorator - placeholder"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

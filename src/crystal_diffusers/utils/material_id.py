import time
import uuid
from typing import Optional


def get_cd_id(ts: Optional[float] = None, use_utc: bool = True) -> str:
    """
    Generate IDs like 'cd-YYMMDD-abc12'.

    Args:
        ts: Optional POSIX timestamp; if None, uses now().
        use_utc: If True, date part is in UTC; else uses local time.

    Returns:
        str: Identifier, e.g. 'cd-251023-k8h2q'.
    """
    if ts is None:
        ts = time.time()

    tm = time.gmtime(ts) if use_utc else time.localtime(ts)
    yymmdd = time.strftime("%y%m%d", tm)

    tail = uuid.uuid4().hex
    return f"cd-{yymmdd}-{tail}"

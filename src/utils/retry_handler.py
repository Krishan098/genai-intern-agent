import asyncio
import logging
from typing import Callable, Any
logger=logging.getLogger(__name__)
from src.utils.config import settings
def retry_with_exponential_backoff(
    func:Callable,
    *args,
    max_retries:int=settings.MAX_RETRIES,
    base_delay:float=settings.RETRY_DELAY,
    **kwargs
)->Any:
    for attempt in range(max_retries+1):
        try:
            if asyncio.iscoroutinefunction(func):
                return func(*args,**kwargs)
            else:
                return func(*args,**kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Function failed after {max_retries+1} attempts:{e}")
                raise e
            delay=base_delay*(2**attempt)
            logger.warning(f"Attempt {attempt+1} failed:{e}. Retrying in {delay} seconds ...")
            asyncio.sleep(delay)
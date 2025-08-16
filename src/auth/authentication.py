from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from src.utils.config import settings
security=HTTPBearer()
def verify_api_key(credentials:HTTPAuthorizationCredentials=Security(security)):
    if credentials.credentials!=settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,deetail="Invalid API Key")
    return credentials.credentials
    
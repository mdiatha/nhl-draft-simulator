"""Thin async client wrapping the NHL web API."""
import httpx

from app.config import settings


class NHLClient:
    def __init__(self):
        self._base = settings.NHL_API_BASE_URL

    async def get(self, path: str, **params) -> dict:
        url = f"{self._base}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_draft(self, year: int) -> dict:
        return await self.get(f"/draft/{year}")

    async def get_prospects(self, year: int) -> dict:
        return await self.get(f"/prospects/{year}")

    async def get_teams(self) -> dict:
        return await self.get("/standings/now")

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .. import __project__, __version__

app = FastAPI()


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/healthz")
async def healthz():
    return {"message": f"Running '{__project__}' ver: '{__version__}'"}

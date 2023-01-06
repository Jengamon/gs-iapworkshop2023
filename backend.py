from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import Request

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class Point(BaseModel):
    x: float
    y: float

class LinearRegression(BaseModel):
    points: list[Point]
    degree: int | None = None

@app.post("/lr")
async def linear_regression(data: LinearRegression):
    print(data)
    return {}
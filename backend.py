from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import Request
import jax.numpy as jnp
import jax.random as jrand
from jax import jit, grad, vmap, value_and_grad

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
    
    def convert(self) -> jnp.ndarray:
        arr = jnp.zeros((len(self.points), 2))
        for (i, point) in enumerate(self.points):
            arr = arr.at[i, 0].set(point.x / 300)
            arr = arr.at[i, 1].set(point.y / 300)
        return arr

key = jrand.PRNGKey(0)

def predict(W, b, inputs):
    return jnp.dot(inputs, W.T) + b

# @jit
def loss(W, b, points):
    xs = points[:, 0]
    ys = points[:, 1]
    pows = jnp.arange(0, W.shape[1]).reshape(W.shape[1], 1) + 1
    x_pow = (xs[None, :] ** pows).reshape(xs.shape[0], -1)
    print(x_pow.shape, x_pow)
    preds = predict(W, b, x_pow)
    return jnp.sum((preds - ys) ** 2)

@app.post("/lr")
async def linear_regression(data: LinearRegression):
    global key

    points = data.convert()

    # Degree determines size of theta
    skey, Wkey, bkey = jrand.split(key, 3)
    key = skey
    W = jrand.normal(Wkey, (1, data.degree))
    b = jrand.normal(bkey, ())

    # print(value_and_grad(loss)(jnp.array([[1.0]]), 0.0, points))
    err = jnp.inf
    for _ in range(0, 100):
        val, (Wgrad, bgrad) = value_and_grad(loss, (0, 1))(W, b, points)
        W = W - 0.25/data.degree * Wgrad
        b = b - 0.25/data.degree * bgrad
        print(Wgrad, bgrad, err)

    points = []
    pows = jnp.arange(0, W.shape[1]) + 1
    for i in jnp.linspace(0, 1, 300):
        x_pow = i ** pows
        y = predict(W, b, x_pow)
        points.append(Point(x = i * 300, y = y * 300))

    return points
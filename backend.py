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

def predict(W, inputs):
    return jnp.dot(inputs, W.T)

@jit
def loss(W, points, lamb):
    xs = points[:, 0]
    ys = points[:, 1]
    pows = jnp.arange(0, W.shape[1]).reshape(W.shape[1], 1)
    x_pow = (xs[None, :] ** pows).T
    # print(x_pow.shape, x_pow)
    preds = predict(W, x_pow)
    # print(ys.reshape(-1, 1).shape, preds.shape)
    return jnp.sum((preds - ys.reshape(-1, 1)) ** 2) + lamb * jnp.squeeze(jnp.dot(W, W.T))

@app.post("/lr")
async def linear_regression(data: LinearRegression):
    global key

    points = data.convert()

    # Degree determines size of theta
    skey, Wkey = jrand.split(key)
    key = skey
    W = jrand.normal(Wkey, (1, data.degree + 1))

    # print(value_and_grad(loss)(jnp.array([[1.0]]), 0.0, points))
    err = jnp.inf
    iters = 100_000
    counter = 0
    vg = value_and_grad(loss)
    while err > 0.00001:
        val, Wgrad = vg(W, points, 0)
        W = W - 0.05 * Wgrad
        err = jnp.dot(Wgrad, Wgrad.T) ** (1/2)
        counter += 1
        if counter > iters:
            break
        # print("gradup", Wgrad, err)

    # print(W)
    points = []
    pows = jnp.arange(0, W.shape[1])
    for i in jnp.linspace(0, 1, 300):
        x_pow = i ** pows
        y = predict(W, x_pow)
        points.append(Point(x = i * 300, y = y * 300))

    return points
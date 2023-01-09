const canvas = document.getElementById("canvas");
const degree = document.getElementById("degree");
const ctx = canvas.getContext("2d");

const points = [];
let bgpoints = [];

function createPoint(x, y) {
    points.push({ x, y });
}

function clearPoints() {
    points = [];
}

function debounce(func, timeout = 300) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            func.apply(this, args);
        }, timeout);
    };
}

function drawData() {
    ctx.clearRect(0, 0, 300, 300);

    for (let i = 0; i < bgpoints.length - 1; i++) {
        ctx.moveTo(bgpoints[i].x, bgpoints[i].y);
        ctx.lineTo(bgpoints[i + 1].x, bgpoints[i + 1].y);
        ctx.stroke();
    }

    for (let i = 0; i < points.length; i++) {
        ctx.beginPath();
        ctx.arc(points[i].x, points[i].y, 3, 0, 2 * Math.PI);
        ctx.fill();
    }
}

function _requestData() {
    ctx.fillText("Loading...", 0, 300);
    // Request a linear regression, and then display it
    const body = {
        degree: degree.value,
        points: points.map((point) => {
            return { x: point.x, y: point.y };
        }),
    };
    console.log(JSON.stringify(body));
    const response = fetch("/lr", {
        method: "post",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });

    response
        .then((resp) => resp.json())
        .then((payload) => {
            bgpoints = payload;
            drawData();
        })
        .catch((err) => console.error(err));
}

const requestData = debounce(() => _requestData());

degree.addEventListener("change", requestData);
canvas.addEventListener("click", (ev) => {
    console.log(ev.offsetX - 150, ev.offsetY - 150);
    createPoint(ev.offsetX, ev.offsetY);
    requestData();
});

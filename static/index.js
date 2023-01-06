const canvas = document.getElementById("canvas");
const degree = document.getElementById("degree");
const ctx = canvas.getContext("2d");

const points = [];

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
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  }
}

function drawData(payload) {
  console.log(payload);
}

function _requestData() {
  // Request a linear regression, and then display it
  const body = {
    "degree": degree.value,
    "points": points,
  };
  console.log(JSON.stringify(body))
  const response = fetch("/lr", {
    method: "post",
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body)
  });

  response.then(resp => resp.json()).then(drawData)
    .catch(err => console.error(err));
}

const requestData = debounce(() => _requestData());

degree.addEventListener('change', requestData);
canvas.addEventListener('click', (ev) => {
  console.log(ev.offsetX - 150, ev.offsetY - 150);
});
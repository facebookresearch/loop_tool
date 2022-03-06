import * as lt from "./loop_tool.mjs";
import * as util from "./main.mjs";
window.lt = lt;

let data_canvas = null;
let data_ctx = null;
let cur_hash = null;
let opt_hash = null;
let loop_editor = null;

const blank_template = `const [h, w, c] = lt.symbols("h w c");
const V = lt.tensor(video_data.height,
                    video_data.width,
                    4).to(h, w, c);
V.set(video_data.data);

await display(V);`;

const color_template = `// manipulate colors

const [h, w, c] = lt.symbols("h w c");
const V = lt.tensor(video_data.height, video_data.width, 4).to(h, w, c);
V.set(video_data.data);

// brighten but clip to max value of 240
const B = V.mul(1.2).min(240);

// make 40% more red
const redden = lt.tensor(4).to(c);
redden.set([1.4, 1, 1, 1]);
const R = B.mul(redden);

// contrast
const C = R.sub(15).div(230).mul(255);

await display(C);`;

const edge_template = `// Edge detection kernel
// https://en.wikipedia.org/wiki/Kernel_(image_processing)

const [h, w, c] = lt.symbols("h w c");
const V = lt.tensor(video_data.height,
                    video_data.width,
                    4).to(h, w, c);
V.set(video_data.data);

const [kh, kw] = lt.symbols("kh kw");
const W = lt.tensor(3, 3).to(kh, kw);
W.set([0, -1,  0,
      -1,  4, -1,
       0, -1,  0]);

const X = lt.convolve(V, W, [h, w], [kh, kw]);
const Y = X.sum(c).add(100);

const alpha = lt.tensor(4).to(c);
alpha.set([255, 255, 255, 0]);

const Z = Y.max(alpha);
await display(Z);`;

const sobel_template = `// Sobel operator

const [h, w, c] = lt.symbols("h w c");
const V = lt.tensor(video_data.height,
                    video_data.width,
                    4).to(h, w, c);
V.set(video_data.data);

const [kh, kw] = lt.symbols("kh kw");

const Gx = lt.tensor(3, 3).to(kh, kw);
Gx.set([ -1, 0, 1,
         -2, 0, 2,
         -1, 0, 1 ])

const Gy = lt.tensor(3, 3).to(kh, kw);
Gy.set([ 1,  2,  1,
         0,  0,  0,
        -1, -2, -1 ])

let Vx = lt.convolve(V, Gx, [h, w], [kh, kw])
					 .sum(c)
           .div(3);
let Vy = lt.convolve(V, Gy, [h, w], [kh, kw])
           .sum(c)
           .div(3);
Vx = Vx.to(...Vy.symbolic_shape);

const Vx2 = Vx.mul(Vx);
const Vy2 = Vy.mul(Vy);
const Y = Vx2.add(Vy2).sqrt();

const alpha = lt.tensor(4).to(c);
alpha.set([255, 255, 255, 0]);

const Z = Y.add(alpha);

await display(Z);`;

async function initWebcam() {
  const webcam_div = document.querySelector("#webcam");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: true,
    });
    let { width, height, facingMode } = stream.getTracks()[0].getSettings();
    console.log(stream.getTracks()[0].getSettings());
    if (data_canvas === null) {
      data_canvas = document.createElement("canvas");
      data_canvas.height = height;
      data_canvas.width = width;
      data_ctx = data_canvas.getContext("2d");
    }

    const vid_elem = document.createElement("video");
    vid_elem.setAttribute("muted", true);
    vid_elem.setAttribute("playsinline", true);
    vid_elem.srcObject = stream;
    vid_elem.addEventListener("play", function () {
      runLoop();
    });
    vid_elem.play();

    const display_canvas = document.querySelector("#output_canvas");
    display_canvas.height = height;
    display_canvas.width = width;

    webcam_div.appendChild(vid_elem);
  } catch (err) {
    webcam_div.textContent = err;
  }
}

function optimize(tensor) {
  if (tensor.hash !== opt_hash) {
    tensor.optimize();
    opt_hash = tensor.hash;
  }
}

async function display(tensor) {
  if (
    (cur_hash === null || cur_hash != tensor.hash) &&
    !document.querySelector("#opt").classList.contains("hidden")
  ) {
    cur_hash = tensor.hash;
    loop_editor = new util.Editor(document.getElementById("opt"), tensor);
    loop_editor.render();
  }
  const float_data = await tensor.data;
  const [h, w, c] = tensor.shape;
  const image_data = new ImageData(w, h);
  image_data.data.set(float_data);
  const canvas = document.querySelector("#output_canvas");
  const ctx = canvas.getContext("2d");
  ctx.putImageData(image_data, 0, 0);
}

async function loop() {
  if (!data_ctx) {
    return;
  }
  const video = document.querySelector("#webcam video");
  if (!video) {
    return;
  }

  data_ctx.drawImage(video, 0, 0, data_canvas.width, data_canvas.height);
  const video_data = data_ctx.getImageData(
    0,
    0,
    data_canvas.width,
    data_canvas.height
  );
}

async function runLoop() {
  while (true) {
    try {
      const t = performance.now();
      if (document.querySelector("#opt").classList.contains("hidden")) {
        lt.clear_heap();
        cur_hash = null;
      }
      await loop();
      const d = performance.now() - t;
      document.querySelector("#stats").textContent = `${Math.round(
        1e3 / d
      )} fps`;
    } catch (e) {
      if (Number.isInteger(e)) {
        e = lt.getExceptionMessage(e);
      }
      document.querySelector("#error_out").textContent = e;
    }
    await new Promise((r) => {
      requestAnimationFrame(r);
    });
  }
}

function updateLoop(editor) {
  const fn_def = `
loop = null;
loop = async function() {

if (!data_ctx) {
  return;
}
const video = document.querySelector("#webcam video");
if (!video) {
  return;
}

data_ctx.drawImage(video, 0, 0, data_canvas.width, data_canvas.height);
const video_data = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);

${editor.getValue()}

}
`;
  eval(fn_def);
}

function init() {
  let editor = CodeMirror.fromTextArea(document.querySelector("#codeeditor"), {
    lineNumbers: true,
    tabSize: 2,
    mode: "javascript",
  });
  if (window.location.hash) {
    const s = window.location.hash.slice(1);
    const c = LZString.decompressFromBase64(decodeURIComponent(s));
    editor.setValue(c);
  } else {
    editor.setValue(edge_template);
  }
  editor.on("change", function () {
    try {
      updateLoop(editor);
      window.location.hash = encodeURIComponent(
        LZString.compressToBase64(editor.getValue())
      );
      document.querySelector("#error_out").textContent = "";
    } catch (e) {
      document.querySelector("#error_out").textContent = e;
      console.log(e);
    }
  });

  document
    .querySelector('button[name="blank"]')
    .addEventListener("click", () => {
      editor.setValue(blank_template);
    });
  document
    .querySelector('button[name="color"]')
    .addEventListener("click", () => {
      editor.setValue(color_template);
    });
  document
    .querySelector('button[name="edge"]')
    .addEventListener("click", () => {
      editor.setValue(edge_template);
    });
  document
    .querySelector('button[name="sobel"]')
    .addEventListener("click", () => {
      editor.setValue(sobel_template);
    });

  updateLoop(editor);

  initWebcam();
  window.addEventListener("keydown", (e) => {
    if (editor.hasFocus()) {
      return;
    }
    if (!loop_editor) {
      return;
    }
    e.preventDefault();
    loop_editor.handle_keydown(e);
    loop_editor.render();
  });
}

export { init };

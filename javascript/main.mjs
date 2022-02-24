import * as lt from "./loop_tool.mjs";
window.lt = lt;

async function log(...args) {
  const str = args.join(" ");
  for (let s of str) {
    document.querySelector("#output").appendChild(document.createTextNode(s));
    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  }
  document.querySelector("#output").appendChild(document.createElement("br"));
}

class Editor {
  constructor(div, t, callback = null) {
    div.innerHTML = "";
    this.div = document.createElement("pre");
    div.appendChild(this.div);
    this.t = t;
    this.changed = true;
    this.flops = 0;
    this.benchspan = null;
    this.callback = callback;
    this.lt = t.loop_tree;
    this.colors = [
      "#56B6C2",
      "#E06C75",
      "#C678DD",
      "#BE5046",
      "#D19A66",
      "#E5C07B",
      "#61AFEF",
      "#98C379",
    ];
    this.pallette = {};
    this.highlight = this.lt.children(-1)[0];
    this.bench_loop();
  }

  handle_keydown(e) {
    if (e.code === "ArrowUp") {
      if (e.shiftKey) {
        const prev = this.lt.prev_ref(this.highlight);
        this.try_swap(this.highlight, prev);
        this.highlight = prev;
        return;
      }
      this.highlight = this.lt.prev_ref(this.highlight);
    }
    if (e.code === "ArrowDown") {
      if (e.shiftKey) {
        const next = this.lt.next_ref(this.highlight);
        this.try_swap(this.highlight, next);
        this.highlight = next;
        return;
      }
      this.highlight = this.lt.next_ref(this.highlight);
    }
    if (e.code === "Backspace") {
      if (!this.lt.is_loop(this.highlight)) {
        return;
      }
      try {
        this.update_tree(this.lt.merge(this.highlight));
      } catch (e) {
        throw lt.getExceptionMessage(e);
      }
    }
    if (e.code === "KeyU") {
      if (!this.lt.is_loop(this.highlight)) {
        return;
      }
      let annot = this.lt.annotation(this.highlight);
      if (annot === "unroll") {
        annot = "";
      } else {
        annot = "unroll";
      }
      try {
        this.update_tree(this.lt.annotate(this.highlight, annot));
      } catch (e) {
        throw lt.getExceptionMessage(e);
      }
    }
    if (e.code === "KeyS") {
      if (!this.lt.is_loop(this.highlight)) {
        return;
      }
      const s = Number.parseInt(prompt("split by what factor?"));
      if (!Number.isInteger(s)) {
        return;
      }
      try {
        this.update_tree(this.lt.split(this.highlight, s));
      } catch (e) {
        throw lt.getExceptionMessage(e);
      }
    }
  }

  update_tree(new_lt) {
    this.highlight = new_lt.map_ref(this.highlight, this.lt);
    this.lt.delete();
    this.lt = new_lt;
    this.t.set_loop_tree(this.lt);
    this.changed = true;
  }

  try_swap(refa, refb) {
    if (!this.lt.is_loop(refb) || !this.lt.is_loop(refa)) {
      return;
    }
    const loop = this.lt.loop(refa);
    const next_loop = this.lt.loop(refb);
    if (loop.v() != next_loop.v()) {
      try {
        this.update_tree(this.lt.swap(refa, refb));
      } catch (e) {
        throw lt.getExceptionMessage(e);
      }
    }
    loop.delete();
    next_loop.delete();
  }

  async bench_loop() {
    while (true) {
      if (this.changed && this.benchspan) {
        // || (Math.random() < 0.1)) {
        this.flops = Math.round((await this.t.benchmark()) / 1e7) / 1e2;
        this.benchspan.textContent = `${this.flops} gflops`;
        this.changed = false;
      }
      await new Promise((resolve) => {
        requestAnimationFrame(resolve);
      });
    }
  }

  render() {
    if (this.callback && this.changed) {
      this.callback();
    }
    let lines = [];
    const spanGen = (txt) => {
      const s = document.createElement("span");
      s.textContent = txt;
      return s;
    };
    const brGen = () => {
      return document.createElement("br");
    };

    this.benchspan = spanGen(`${this.flops} gflops`);
    this.benchspan.style.fontWeight = "bold";
    this.benchspan.style.color = "#d19a66";
    lines.push(this.benchspan);
    lines.push(spanGen(` achieved, (${this.t.flops} total flops in program)`));
    lines.push(brGen());

    let highlighted_var = -1;
    if (this.lt.is_loop(this.highlight)) {
      let loop = this.lt.loop(this.highlight);
      highlighted_var = loop.v();
      loop.delete();
    }

    const randomColor = (brightness) => {
      if (this.colors.length) {
        let c = this.colors[0];
        this.colors = this.colors.slice(1);
        return c;
      }
      function randomChannel(brightness) {
        const r = 255 - brightness;
        const n = 0 | (Math.random() * r + brightness);
        const s = n.toString(16);
        return s.length == 1 ? "0" + s : s;
      }
      return (
        "#" +
        randomChannel(brightness) +
        randomChannel(brightness) +
        randomChannel(brightness)
      );
    };

    const renderVar = (v) => {
      const s = spanGen(this.lt.var_name(v).split("_")[0]);
      if (v == highlighted_var) {
        s.style.fontWeight = "bold";
      }
      if (!(v in this.pallette)) {
        this.pallette[v] = randomColor(80);
      }
      s.style.color = this.pallette[v];
      return s;
    };
    const renderValue = (n) => {
      let out = [];
      out.push(spanGen("%" + n + "["));
      const vs = this.lt.node_vars(n);
      let set = false;
      for (let v of vs) {
        if (set) {
          out.push(spanGen(", "));
        } else {
          set = true;
        }
        out.push(renderVar(v));
      }
      out.push(spanGen("]"));
      return out;
    };
    const renderNode = (nr, sized) => {
      let out = [];
      let set = false;
      out.push(spanGen(this.lt.node_type(nr)));
      out.push(spanGen("("));
      for (let n of this.lt.node_inputs(nr)) {
        if (set) {
          out.push(spanGen(", "));
        } else {
          set = true;
        }
        out = out.concat(renderValue(n));
      }
      out.push(spanGen(")"));
      if (sized) {
        const elems = this.lt.node_size(nr);
        const s = spanGen(` [${elems} elem${elems > 1 ? "s" : ""}]`);
        out.push(s);
      }
      return out;
    };
    const renderLoop = (ref) => {
      let out = [];
      const loop = this.lt.loop(ref);
      const v = loop.v();
      out.push(spanGen("for "));
      out.push(renderVar(v));
      out.push(spanGen(" in " + loop.size()));
      const tail = loop.tail();
      if (tail) {
        out.push(spanGen(" r " + tail));
      }
      const annot = this.lt.annotation(ref);
      if (annot) {
        out.push(spanGen(" [" + annot + "]"));
      }
      loop.delete();
      return out;
    };

    for (let ref of this.lt.walk()) {
      let spans = [];
      spans.push(spanGen(" ".repeat(this.lt.depth(ref))));
      if (this.lt.is_loop(ref)) {
        spans = spans.concat(renderLoop(ref));
      } else {
        const n = this.lt.node(ref);
        spans = spans.concat(renderNode(n, this.highlight == ref));
      }
      let div = document.createElement("div");
      for (let s of spans) {
        div.appendChild(s);
      }
      if (ref === this.highlight) {
        div.style.background = "#eeeeee";
        div.style.color = "black";
      }
      lines.push(div);
    }
    this.div.innerHTML = "";
    for (let l of lines) {
      this.div.appendChild(l);
    }
  }
}

class Renderer {
  constructor(div, wabt, t) {
    this.div = div;
    this.div.innerHTML = "";
    const pre = document.createElement("pre");
    this.code_elem = document.createElement("code");
    pre.appendChild(this.code_elem);
    this.div.appendChild(pre);
    this.code_elem.classList.toggle("hljs");
    this.wabt = wabt;
    this.t = t;
    // dump wasm or C
    this.wasm_mode = true;
    this.div.addEventListener("click", () => {
      this.wasm_mode = !this.wasm_mode;
      this.render();
    });
  }

  async render() {
    let text = "";
    if (!this.wasm_mode) {
      text = this.t.code;
    } else {
      const mod = this.wabt.readWasm(this.t.wasm, {
        readDebugNames: true,
      });
      text = mod.toText({ foldExprs: false, inlineExport: false });
    }
    this.code_elem.textContent = text;
    this.code_elem.innerHTML = hljs.highlight(text, {
      language: this.wasm_mode ? "wasm" : "c",
    }).value;
  }
}

let wabt = null;
let editor = null;
let loop_editor = null;

function render(tensor) {
  let r = new Renderer(document.getElementById("display"), wabt, tensor);
  loop_editor = new Editor(document.getElementById("edit"), tensor, () => {
    r.render();
  });
  loop_editor.render();
}

async function setup() {
  wabt = WabtModule();
  editor = CodeMirror.fromTextArea(document.querySelector("#codeeditor"), {
    lineNumbers: true,
    tabSize: 2,
    mode: "javascript",
  });
  editor.on("change", function () {
    try {
      eval(editor.getValue());
    } catch (e) {}
  });

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

  eval(editor.getValue());
  await log("hi! this is the loop_tool.js demo :^}");
  await log("");
  await log("<<< on the left is an interactive loop optimizer");
  await log("    - use the arrow keys to move around");
  await log("    - hold shift while moving to drag loops");
  await log("    - hit 's' to split the loops");
  await log("    - hit 'u' to unroll them");
  await log("");
  await log("vvvv below is the live-generated wasm");
  await log("    - it's benchmarked in real time ('gflops achieved')");
  await log("    - SIMD is a work in progress");
  await log("");
  await log("&&&& bottom left is the frontend code (feel free to edit)");
  await log(
    "     - the repo is here: https://github.com/facebookresearch/loop_tool"
  );
}

export { setup }

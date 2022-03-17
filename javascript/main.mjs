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

  get_vars() {
    if (this.lt.is_loop(this.highlight)) {
      return [];
    }
    let out = [];
    const node = this.lt.node(this.highlight);
    out.push(this.lt.node_vars(node));
    const inputs = this.lt.node_inputs(node);
    for (let inp of inputs) {
      out.push(this.lt.node_vars(inp));
    }
    return out;
  }

  prev_var() {
    const vars = this.get_vars();
    if (!this.var_highlight) {
      return;
    }
    let {
      val_idx,
      var_idx
    } = this.var_highlight;
    if (var_idx == 0) {
      if (val_idx == 0) {
        return;
      }
      val_idx = val_idx - 1;
      var_idx = vars[val_idx].length - 1;
      this.var_highlight = {
        val_idx: val_idx,
        var_idx: var_idx
      };
    } else {
      this.var_highlight = {
        val_idx: val_idx,
        var_idx: var_idx - 1
      };
    }
  }

  swap_prev_var() {
    const vars = this.get_vars();
    if (!this.var_highlight) {
      return;
    }
    let {
      val_idx,
      var_idx
    } = this.var_highlight;
    if (var_idx == 0) {
      return;
    }
    let n = this.lt.node(this.highlight);
    if (val_idx != 0) {
      n = this.lt.node_inputs(n)[val_idx - 1];
    }
    const a = vars[val_idx][var_idx];
    const b = vars[val_idx][var_idx - 1];
    this.update_tree(this.lt.swap_vars(n, a, b));
  }

  next_var() {
    const vars = this.get_vars();
    if (!this.var_highlight) {
      return;
    }
    const {
      val_idx,
      var_idx
    } = this.var_highlight;
    if (vars[val_idx].length - 1 == var_idx) {
      if (vars.length - 1 == val_idx) {
        return;
      }
      this.var_highlight = {
        val_idx: val_idx + 1,
        var_idx: 0
      };
    } else {
      this.var_highlight = {
        val_idx: val_idx,
        var_idx: var_idx + 1
      };
    }
  }

  swap_next_var() {
    const vars = this.get_vars();
    if (!this.var_highlight) {
      return;
    }
    let {
      val_idx,
      var_idx
    } = this.var_highlight;
    if (var_idx == vars[val_idx].length - 1) {
      return;
    }
    let n = this.lt.node(this.highlight);
    if (val_idx != 0) {
      n = this.lt.node_inputs(n)[val_idx - 1];
    }
    const a = vars[val_idx][var_idx];
    const b = vars[val_idx][var_idx + 1];
    this.update_tree(this.lt.swap_vars(n, a, b));
  }

  handle_keydown(e) {
    console.log(e);
    if (e.code === "ArrowUp") {
      if (e.shiftKey) {
        const prev = this.lt.prev_ref(this.highlight);
        this.try_swap(this.highlight, prev);
        this.highlight = prev;
        return;
      } else if ((e.metaKey || e.ctrlKey) && !this.lt.is_loop(this.highlight)) {
        this.update_tree(this.lt.decrease_reuse(this.highlight));
        return;
      }
      this.highlight = this.lt.prev_ref(this.highlight);
      this.var_highlight = null;
    }
    if (e.code === "ArrowDown") {
      if (e.shiftKey) {
        const next = this.lt.next_ref(this.highlight);
        this.try_swap(this.highlight, next);
        this.highlight = next;
        return;
      } else if ((e.metaKey || e.ctrlKey) && !this.lt.is_loop(this.highlight)) {
        this.update_tree(this.lt.increase_reuse(this.highlight));
        return;
      }
      this.highlight = this.lt.next_ref(this.highlight);
      this.var_highlight = null;
    }
    if (e.code === "ArrowLeft") {
      if (!this.var_highlight) {
        return;
      }
      if (e.shiftKey) {
        this.swap_prev_var();
      }
      this.prev_var();
    }

    if (e.code === "ArrowRight") {
      if (!this.var_highlight) {
        return;
      }
      if (e.shiftKey) {
        this.swap_next_var();
      }
      this.next_var();
    }
    if (e.code == "Enter") {
      if (!this.lt.is_loop(this.highlight)) {
        const vars = this.get_vars();
        if (vars.length) {
          this.var_highlight = {
            val_idx: 0,
            var_idx: 0
          };
        }
      }
    }
    if (e.code === "Backspace") {
      if (!this.lt.is_loop(this.highlight)) {
        const n = this.lt.node(this.highlight);
        if (this.lt.node_type(n) == "copy") {
          this.update_tree(this.lt.delete_copy(this.highlight));
        }
        return;
      }
      try {
        this.update_tree(this.lt.merge(this.highlight));
      } catch (e) {
        throw lt.getExceptionMessage(e);
      }
    }
    if (e.code === "KeyV") {
      if (!this.lt.is_loop(this.highlight)) {
        return;
      }
      let annot = this.lt.annotation(this.highlight);
      if (annot === "vectorize") {
        annot = "";
      } else {
        annot = "vectorize";
      }
      try {
        this.update_tree(this.lt.annotate(this.highlight, annot));
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
    if (e.code === "KeyB") {
      this.changed = true;
      this.long_bench = true;
    }
    if (e.code === "KeyS") {
      if (!this.lt.is_loop(this.highlight)) {
        const s = Number.parseInt(prompt("split which input?"));
        if (!Number.isInteger(s)) {
          return;
        }
        try {
          this.update_tree(this.lt.copy_input(this.highlight, s));
        } catch (e) {
          throw lt.getExceptionMessage(e);
        }
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
    this.lt = new_lt;
    this.t.set_loop_tree(this.lt);
    this.changed = true;
  }

  try_swap(refa, refb) {
    try {
      if (!this.lt.is_loop(refb) && !this.lt.is_loop(refa)) {
        this.update_tree(this.lt.swap_nodes(refa, refb));
      } else if (!this.lt.is_loop(refa) && this.lt.is_loop(refb)) {
        if (this.lt.parent(refa) == refb) {
          // TODO siblings
          this.update_tree(this.lt.remove_loop(refa, refb));
        } else {
          this.update_tree(this.lt.add_loop(refa, refb));
        }
      } else if (this.lt.is_loop(refa) && this.lt.is_loop(refb)) {
        const loop = this.lt.loop(refa);
        const next_loop = this.lt.loop(refb);
        if (loop.v() != next_loop.v()) {
          this.update_tree(this.lt.swap(refa, refb));
        }
        loop.delete();
        next_loop.delete();
      }
    } catch (e) {
      throw lt.getExceptionMessage(e);
    }
  }

  async bench_loop() {
    while (true) {
      if (this.changed && this.benchspan) {
        let bench_ms = 50;
        let warmup_ms = 10;
        if (this.long_bench) {
          this.long_bench = false;
          bench_ms = 1000;
          warmup_ms = 500;
        }
        try {
          this.flops =
            Math.round((await this.t.benchmark(bench_ms, warmup_ms)) / 1e7) / 1e2;
          this.iters = Math.round((this.flops * 1e11) / this.t.flops) / 1e2;
          this.benchspan.textContent = `${this.flops} gflops | ${this.iters} iters/sec`;
          this.changed = false;
        } catch (e) {
          this.benchspan.textContent = '[[cannot run]]';
        }
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

    this.benchspan = spanGen(`${this.flops} gflops | ${this.iters} iters/sec`);
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

    const renderVar = (v, h) => {
      const s = spanGen(this.lt.var_name(v).split("_")[0]);
      if (v == highlighted_var) {
        s.style.fontWeight = "bold";
      }
      if (!(v in this.pallette)) {
        this.pallette[v] = randomColor(80);
      }
      s.style.color = this.pallette[v];
      if (h) {
        s.style.background = '#333333';
      }
      return s;
    };
    const renderValue = (n, hi) => {
      let out = [];
      out.push(spanGen("%" + n + "["));
      const vs = this.lt.node_vars(n);
      let set = false;
      let i = 0;
      for (let v of vs) {
        if (set) {
          out.push(spanGen(", "));
        } else {
          set = true;
        }
        out.push(renderVar(v, hi == i));
        i++;
      }
      out.push(spanGen("]"));
      return out;
    };
    const renderNode = (nr, highlighted) => {
      let out = [];
      let set = false;
      let hi = -1;
      const hi_idx = highlighted && this.var_highlight ? this.var_highlight.val_idx : -1;
      if (hi_idx == 0) {
        hi = this.var_highlight.var_idx;
      }
      out = out.concat(renderValue(nr, hi));
      out.push(spanGen(` = `));
      out.push(spanGen(this.lt.node_type(nr)));
      out.push(spanGen("("));
      let val_idx = 1;
      for (let n of this.lt.node_inputs(nr)) {
        if (set) {
          out.push(spanGen(", "));
        } else {
          set = true;
        }
        if (val_idx == hi_idx) {
          hi = this.var_highlight.var_idx;
        } else {
          hi = -1;
        }
        out = out.concat(renderValue(n, hi));
        val_idx++;
      }
      out.push(spanGen(")"));
      if (highlighted) {
        const elems = this.lt.node_size(nr);
        let attr = this.lt.node_attributes(nr);
        attr = attr.length ? ` (${attr})` : '';
        const s = spanGen(
          ` [${elems} elem${elems > 1 ? "s" : ""}${attr}]`
        );
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
    //this.div.addEventListener("click", () => {
    //  this.wasm_mode = !this.wasm_mode;
    //  this.render();
    //});
  }

  async render() {
    let text = "";
    if (!this.wasm_mode) {
      text = this.t.code;
    } else {
      const mod = this.wabt.readWasm(this.t.wasm, {
        readDebugNames: true,
        simd: true,
      });
      text = mod.toText({
        foldExprs: false,
        inlineExport: false
      });
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
  editor.on("change", function() {
    try {
      eval(editor.getValue());
    } catch (e) {
      console.log(e);
    }
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
  await log("vvvv below is an interactive loop optimizer");
  await log("    - use the arrow keys to move around");
  await log("    - hold shift while moving to drag loops");
  await log("    - hit 's' to split the loops");
  await log("    - hit 'u' to unroll them");
  await log("");
  await log(">>>> to the right is the frontend code (feel free to edit)");
  await log(
    "     - the repo is here: https://github.com/facebookresearch/loop_tool"
  );
  await log("");
  await log("vv>> bottom right is the live-generated wasm");
  await log("    - it's benchmarked in real time ('gflops achieved')");
  await log("    - SIMD is a work in progress");
}

export {
  setup,
  Editor,
  Renderer
};
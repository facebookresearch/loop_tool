import loop_tool_py as lt
import curses
from curses import wrapper
from curses.textpad import Textbox
import time

# use with vim [file] -c 'set updatetime=750 | set autoread | au CursorHold * checktime | call feedkeys("lh")'


def hex_to_color(h):
    r, g, b = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return curses.color_pair(int(16 + r / 48 * 36 + g / 48 * 6 + b / 48))


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)


def get_versions(loop):
    versions = []

    def f(r, depth):
        nonlocal versions
        if tree.is_loop(r) and (tree.loop(r) == loop):
            versions.append(r)

    tree.walk(f)
    return versions


def benchmark(tensor, limit_ms=100):
    start = time.time() * 1000
    iters = 1
    t = 0
    while (t - start) < limit_ms:
        for i in range(iters):
            tensor.force_recompute()
            tensor.resolve()
        t = time.time() * 1000
        iters *= 2
    return 1000 * (iters - 1) / (t - start)


def loop_version(tree, ref):
    if not tree.is_loop(ref):
        return None
    loop = tree.loop(ref)
    version = 0
    keep_scanning = True

    def f(r, depth):
        nonlocal keep_scanning
        nonlocal version
        if r == ref:
            keep_scanning = False
        if keep_scanning and tree.is_loop(r) and tree.loop(r) == loop:
            version += 1

    tree.walk(f)
    return (loop, version)


def highlight(tree, drag):
    assert drag
    highlighted = None
    version = 0

    def find_loop(ref, depth):
        nonlocal highlighted
        nonlocal version
        if (
            tree.is_loop(ref)
            and (tree.loop(ref) == drag[0] and version == drag[1])
            and highlighted == None
        ):
            highlighted = ref
        if tree.is_loop(ref) and (tree.loop(ref) == drag[0]):
            version += 1

    tree.walk(find_loop)
    assert highlighted != None, (
        f"found {version} versions but wanted {drag[1]}:\n" + tree.dump()
    )
    return highlighted


def gen_info(tree, highlighted, drag):
    s = ""
    if tree.is_loop(highlighted):
        if drag is not None:
            s += f"[dragging {tree.ir.dump_var(drag[0].var)} v{drag[1]}]"
    else:
        allocs = lt.Compiler(tree).allocations
        n = tree.ir_node(highlighted)
        if n in allocs:
            s += f"[size: {allocs[n].size}]"
        else:
            s += f"[allocs size {len(allocs)}]"
    return s


def prompt(stdscr, pad, s):
    rows, cols = stdscr.getmaxyx()
    pad.addstr(0, 0, s + " " * (cols - len(s) - 1))
    # , hex_to_color("00ff00"))
    stdscr.refresh()
    pad.refresh(0, 0, 0, 0, rows, cols)
    editwin = curses.newwin(1, 30, 0, len(s))
    box = Textbox(editwin, insert_mode=True)

    def validate(x):
        if x == 10:
            x = 7
        if x == 127:
            x = curses.KEY_BACKSPACE
        return x

    box.edit(validate)
    message = box.gather()
    split_size = 0
    try:
        split_size = int(message)
    except:
        pass
    return split_size


def ui_impl(stdscr, tensor, fn):
    tree = tensor.loop_tree
    trees = [tree]
    highlighted = tree.roots[0]
    drag = None
    rows, cols = stdscr.getmaxyx()
    stdscr.clear()
    curses.curs_set(0)
    tree_pad = curses.newpad(rows, cols)

    iters_sec = 0
    flops = 0
    reads = 0
    writes = 0

    def render(changed):
        nonlocal highlighted, iters_sec, flops, reads, writes
        highlighted = highlight(tree, drag) if drag else highlighted
        tree_pad.erase()
        i = 0
        info = gen_info(tree, highlighted, drag)
        tree_pad.addstr(i, 0, info)

        if changed:
            tensor.set(tree)
            trees.append(tree)
            if fn:
                with open(fn, "w") as f:
                    f.write(tensor.code)
            _ = benchmark(tensor, 10)  # warmup
            iters_sec = benchmark(tensor)
            flops = tree.flops()
        tree_pad.addstr(
            i,
            len(info) + 1,
            f"{flops * iters_sec / 1e9:.2f} GFlops, ({iters_sec:.2f} iters/sec, {flops} total flops)",
        )

        def _render_ref(ref):
            if tree.is_loop(ref):
                loop = tree.loop(ref)
                v = tree.ir.dump_var(loop.var)
                r = f" r {loop.tail}" if loop.tail else ""
                return f"for {v} in {loop.size}{r}"
            return tree.dump(ref)

        def _r(ref, depth):
            nonlocal i
            i += 1
            tree_pad.addstr(i, depth, _render_ref(ref))
            if ref == highlighted:
                tree_pad.chgat(i, 0, curses.A_REVERSE)

        tree.walk(_r)

        stdscr.refresh()
        tree_pad.refresh(0, 0, 0, 0, rows, cols)

    render(True)

    def update_tree(new_tree):
        nonlocal highlighted, tree, changed
        highlighted = new_tree.map_ref(highlighted, tree)
        tree = new_tree
        changed = True

    while True:
        key = stdscr.getkey()
        changed = False
        if key == "q":
            break
        elif key == "s":
            split_size = prompt(stdscr, tree_pad, "inner size? ")
            try:
                update_tree(tree.split(highlighted, split_size))
            except:
                pass
        elif key == "u" and len(trees) > 1:
            trees = trees[:-1]
            update_tree(trees[-1])
        elif key == "KEY_DOWN":
            if drag:
                update_tree(tree.try_swap(highlighted, tree.next_ref(highlighted)))
            else:
                n = tree.next_ref(highlighted)
                if n is not None:
                    highlighted = n
        elif key == "KEY_UP":
            if drag:
                update_tree(tree.try_swap(highlighted, tree.previous_ref(highlighted)))
            else:
                p = tree.previous_ref(highlighted)
                if p is not None:
                    highlighted = p
        elif key == "KEY_SR":  # up + shift
            update_tree(tree.try_swap(highlighted, tree.previous_ref(highlighted)))
        elif key == "KEY_SF":  # down + shift
            update_tree(tree.try_swap(highlighted, tree.next_ref(highlighted)))
            changed = True
        elif key in ("KEY_BACKSPACE", "\b", "\x7f"):
            update_tree(tree.merge(highlighted))
        elif key == "\n":
            key = "ENTER"
            drag = None if drag else loop_version(tree, highlighted)
        render(changed)
        if key == "u":
            trees = trees[:-1]
    return tree


def ui(T, path=""):
    T.set(wrapper(ui_impl, T, path))

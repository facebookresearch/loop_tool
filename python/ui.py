import loop_tool_py as lt
import curses
from curses import wrapper
import time

# use with vim [file] -c 'set updatetime=750 | set autoread | au CursorHold * checktime | call feedkeys("lh")'


def ui_impl(stdscr, tensor, fn):
    tree = tensor.loop_tree
    trees = [tree]
    highlighted = tree.roots[0]
    drag = None
    rows, cols = stdscr.getmaxyx()
    stdscr.clear()
    curses.curs_set(0)
    tree_pad = curses.newpad(rows, cols)

    def count_stats(tree):
        loop_sizes = []
        total_flops = 0
        total_reads = 0
        total_writes = 0

        def _r(ref, depth):
            nonlocal loop_sizes, total_flops, total_reads, total_writes
            if tree.is_loop(ref):
                loop_sizes = loop_sizes[:depth]
                loop = tree.loop(ref)
                loop_sizes.append(loop)
                return
            var_sizes = dict()
            for loop in loop_sizes[::-1]:
                if loop.var not in var_sizes:
                    var_sizes[loop.var] = 1
                var_sizes[loop.var] *= loop.size
                var_sizes[loop.var] += loop.tail
            iterations = 1
            for _, s in var_sizes.items():
                iterations *= s
            if "read" in tree.dump(ref):
                total_reads += iterations
            elif "view" in tree.dump(ref):
                total_reads += iterations
                total_writes += iterations
            elif "write" in tree.dump(ref):
                total_writes += iterations
            else:
                total_flops += iterations

        tree.walk(_r)
        return total_flops, total_reads, total_writes

    def benchmark(limit_ms=100):
        start = time.time() * 1000
        iters = 1
        t = 0
        while (t - start) < limit_ms:
            for i in range(iters):
                tensor.invalidate()
                tensor.resolve()
            t = time.time() * 1000
            iters *= 2
        return 1000 * (iters - 1) / (t - start)

    iters_sec = 0
    flops = 0
    reads = 0
    writes = 0

    def render(changed, info=""):
        nonlocal iters_sec, flops, reads, writes
        tree_pad.erase()
        i = 0
        tree_pad.addstr(i, 0, info)

        if changed:
            tensor.set(tree)
            trees.append(tree)
            if fn:
                with open(fn, "w") as f:
                    f.write(tensor.code)
            _ = benchmark(10)  # warmup
            iters_sec = benchmark()
            flops, reads, writes = count_stats(tree)
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

    def get_versions(loop):
        versions = []

        def f(r, depth):
            nonlocal versions
            if tree.is_loop(r) and (tree.loop(r) == loop):
                versions.append(r)

        tree.walk(f)
        return versions

    def rehighlight():
        nonlocal highlighted
        if not drag:
            return
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
            f"found {version} versions and wanted {drag[1]}:\n" + tree.dump()
        )

    def prev_ref(tree, ref):
        if ref == -1:
            return None
        sibs = tree.children(tree.parent(ref))
        idx = 0
        while sibs[idx] != ref:
            idx += 1
        idx -= 1
        if idx < 0:
            p = tree.parent(ref)
            if p == -1:
                return None
            return p
        n = sibs[idx]
        p = n
        while n != ref:
            p = n
            n = next_ref(tree, n)
        return p

    def next_ref(tree, ref, handle_children=True):
        if ref == -1:
            return None
        children = tree.children(ref)
        if len(children) and handle_children:
            return children[0]
        sibs = tree.children(tree.parent(ref))
        idx = 0
        while sibs[idx] != ref:
            idx += 1
        idx += 1
        if idx < len(sibs):
            return sibs[idx]
        return next_ref(tree, tree.parent(ref), False)

    def drag_inward(ref):
        nonlocal tree
        cs = tree.children(ref)
        for c in cs:
            if tree.is_loop(c):
                tree = lt.swap(tree, ref, c)
                rehighlight()
                return

    def drag_outward(ref):
        nonlocal tree
        nonlocal drag
        p = tree.parent(ref)
        v_before = get_versions(tree.loop(ref))
        if p != -1:
            loop = tree.loop(ref)
            tree = lt.swap(tree, ref, p)
            v_after = get_versions(loop)
            if len(v_after) < len(v_before):
                drag = (drag[0], max(0, drag[1] - 1))
            rehighlight()

    def loop_version(ref):
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

    def info(ref):
        s = ""
        if tree.is_loop(ref):
            if drag is not None:
                s += "[dragging]"
        else:
            allocs = lt.Compiler(tree).allocations
            n = tree.ir_node(ref)
            if n in allocs:
                s += f"[size: {allocs[n].size}]"
            else:
                s += f"[allocs size {len(allocs)}]"
        return s

    def prompt(s):
        nonlocal tree
        rows, cols = stdscr.getmaxyx()
        tree_pad.addstr(0, 0, s + " " * (cols - len(s) - 1))
        stdscr.refresh()
        tree_pad.refresh(0, 0, 0, 0, rows, cols)
        aggregate_s = ""
        split_size = 0
        while True:
            key = stdscr.getkey()
            tree_pad.addstr(0, len(s) + len(aggregate_s), key)
            aggregate_s += key
            if key == "":
                return
            elif key == "\n":
                try:
                    split_size = int(aggregate_s)
                except:
                    pass
                return split_size
            stdscr.refresh()
            tree_pad.refresh(0, 0, 0, 0, *stdscr.getmaxyx())

    while True:
        key = stdscr.getkey()
        changed = False
        if key == "q":
            break
        elif key == "s":
            split_size = prompt("inner size? ")
            try:
                tree = lt.split(tree, highlighted, split_size)
                changed = True
            except:
                pass
        elif key == "u" and len(trees) > 1:
            trees = trees[:-1]
            tree = trees[-1]
            changed = True
        elif key == "KEY_DOWN":
            if drag:
                try:
                    drag_inward(highlighted)
                    changed = True
                except:
                    pass
            else:
                n = next_ref(tree, highlighted)
                if n is not None:
                    highlighted = n
        elif key == "KEY_UP":
            if drag:
                try:
                    drag_outward(highlighted)
                    changed = True
                except:
                    pass
            else:
                p = prev_ref(tree, highlighted)
                if p is not None:
                    highlighted = p
        elif key == "\n":
            key = "ENTER"
            drag = None if drag else loop_version(highlighted)
            rehighlight()
        render(changed, info=info(highlighted))
        if key == "u":
            trees = trees[:-1]
    return tree


def ui(T, path=""):
    T.set(wrapper(ui_impl, T, path))

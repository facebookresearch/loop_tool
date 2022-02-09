import loop_tool_py as lt
import numpy as np
import curses
from curses import wrapper

# use with vim file -c 'set updatetime=750 | set autoread | au CursorHold * checktime | call feedkeys("lh")'

#
# print(C.loop_tree)
#
# c0 = C.loop_tree.roots[0]
# c1 = C.loop_tree.children(c0)[0]
#
# tree = lt.swap(C.loop_tree, c0, c1)
# C.set(tree)
#
# print(C.loop_tree)

# highlighted = tree.roots[0]

# def woot(ref, depth):
#    if ref == highlighted:
#        print(" " * depth, tree.dump(ref), "<<<")
#    else:
#        print(" " * depth, tree.dump(ref))
#
#
# tree.walk(woot)


def ui_impl(stdscr, tensor, fn):
    tree = tensor.loop_tree
    highlighted = tree.roots[0]
    drag = None
    rows, cols = stdscr.getmaxyx()
    stdscr.clear()
    curses.curs_set(0)
    tree_pad = curses.newpad(rows, cols)

    def render():
        tensor.set(tree)
        with open(fn, "w") as f:
            f.write(tensor.code)
        tree_pad.erase()
        i = 0

        def _r(ref, depth):
            nonlocal i
            i += 1
            tree_pad.addstr(i, depth, tree.dump(ref))
            if ref == highlighted:
                tree_pad.chgat(i, 0, curses.A_REVERSE)

        tree.walk(_r)

    render()
    stdscr.refresh()
    tree_pad.refresh(0, 0, 0, 0, rows, cols)

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
                s += f"size: {allocs[n].size}"
            else:
                s += f"allocs size {len(allocs)}"
        return s

    def prompt(s):
        nonlocal tree
        tree_pad.addstr(0, 0, s)
        stdscr.refresh()
        tree_pad.refresh(0, 0, 0, 0, rows, cols)
        aggregate_s = ""
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
                tree = lt.split(tree, highlighted, split_size)
                return
            stdscr.refresh()
            tree_pad.refresh(0, 0, 0, 0, rows, cols)

    while True:
        key = stdscr.getkey()
        if key == "q":
            return
        elif key == "s":
            prompt("inner size? ")
        elif key == "KEY_DOWN":
            if drag:
                drag_inward(highlighted)
            else:
                n = next_ref(tree, highlighted)
                if n is not None:
                    highlighted = n
        elif key == "KEY_UP":
            if drag:
                drag_outward(highlighted)
            else:
                p = prev_ref(tree, highlighted)
                if p is not None:
                    highlighted = p
        elif key == "\n":
            key = "ENTER"
            drag = None if drag else loop_version(highlighted)
            rehighlight()
        render()
        tree_pad.addstr(0, 0, info(highlighted))
        stdscr.refresh()
        tree_pad.refresh(0, 0, 0, 0, rows, cols)


def ui(T, path):
    wrapper(ui_impl, T, path)

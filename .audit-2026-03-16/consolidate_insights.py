#!/usr/bin/env python3
"""
Second-pass cleanup for insights_page.py.
Removes render methods that reference removed analyzer methods,
their tab blocks, and rebuilds the st.tabs() assignment.
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEDUP_MAP = BASE_DIR.parent / ".loki" / "memory" / "semantic" / "dedup_map.json"
INSIGHTS_FILE = BASE_DIR / "insights_page.py"


def main():
    # Load removed method names
    with open(DEDUP_MAP) as f:
        data = json.load(f)

    removed_methods = set()
    for cluster in data["clusters"]:
        methods = cluster["methods"]
        if len(methods) > 1:
            removed_methods.update(methods[1:])

    print(f"Removed methods from analyzer: {len(removed_methods)}")

    with open(INSIGHTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    original_count = len(lines)
    print(f"Insights page: {original_count} lines")

    # --- Step 1: Identify render methods to remove ---
    render_pattern = re.compile(r"^    def (_render_\w+)\(self")
    render_starts = []
    for i, line in enumerate(lines):
        m = render_pattern.match(line)
        if m:
            render_starts.append((i, m.group(1)))

    render_to_remove = set()
    render_ranges = {}  # name -> (start, end)

    for idx, (start, name) in enumerate(render_starts):
        end = render_starts[idx + 1][0] if idx + 1 < len(render_starts) else len(lines)
        render_ranges[name] = (start, end)

        # Check if this render method references any removed method
        block = "".join(lines[start:end])
        for method in removed_methods:
            if method in block:
                render_to_remove.add(name)
                break

    print(f"Render methods to remove: {len(render_to_remove)}")

    # --- Step 2: Find tab blocks to remove ---
    tab_pattern = re.compile(r"^(\s+)with (tab\d+):")
    render_call_pattern = re.compile(r"^\s+self\.(_render_\w+)\(")

    # Map each tab block to its render method and line range
    tab_info = []  # (tab_var, block_start, block_end, render_method)

    i = 0
    while i < len(lines):
        tb = tab_pattern.match(lines[i])
        if tb:
            indent = tb.group(1)
            tab_var = tb.group(2)
            block_start = i
            render_method = None
            block_end = i + 1

            # Scan for render call and find extent of this tab block
            for j in range(i + 1, min(i + 15, len(lines))):
                rc = render_call_pattern.match(lines[j])
                if rc:
                    render_method = rc.group(1)
                    block_end = j + 1
                    break
                # Next tab block or equal/lower indent means end
                if tab_pattern.match(lines[j]):
                    block_end = j
                    break
                block_end = j + 1

            # Extend to include trailing blank lines
            while block_end < len(lines) and lines[block_end].strip() == "":
                block_end += 1

            tab_info.append((tab_var, block_start, block_end, render_method))
            i = block_end
        else:
            i += 1

    # Determine which tabs to remove
    tabs_to_remove = set()
    tab_lines_to_remove = set()

    for tab_var, start, end, render_method in tab_info:
        if render_method and render_method in render_to_remove:
            tabs_to_remove.add(tab_var)
            for k in range(start, end):
                tab_lines_to_remove.add(k)

    print(f"Tab blocks to remove: {len(tabs_to_remove)}")

    # --- Step 3: Mark render method definition lines for removal ---
    render_lines_to_remove = set()
    for name in render_to_remove:
        if name in render_ranges:
            start, end = render_ranges[name]
            for k in range(start, end):
                render_lines_to_remove.add(k)

    all_lines_to_remove = tab_lines_to_remove | render_lines_to_remove
    print(f"Total lines to remove: {len(all_lines_to_remove)}")

    # --- Step 4: Rebuild st.tabs() assignment ---
    # Find the st.tabs block
    tabs_assign_start = None
    tabs_assign_end = None
    for i, line in enumerate(lines):
        if "= st.tabs([" in line:
            tabs_assign_start = i
        if tabs_assign_start is not None and tabs_assign_end is None:
            if "])" in line:
                tabs_assign_end = i + 1
                break

    # Parse existing tab variables
    all_tab_vars = []
    if tabs_assign_start is not None:
        assign_line = lines[tabs_assign_start]
        for vm in re.finditer(r"tab(\d+)", assign_line):
            all_tab_vars.append(f"tab{vm.group(1)}")

    # Parse existing labels
    all_labels = []
    if tabs_assign_start is not None and tabs_assign_end is not None:
        label_pattern = re.compile(r'"([^"]*)"')
        for j in range(tabs_assign_start, tabs_assign_end):
            for lm in label_pattern.finditer(lines[j]):
                all_labels.append(lm.group(1))

    print(f"Original tabs: {len(all_tab_vars)}, labels: {len(all_labels)}")

    # Build keep lists
    keep_indices = [i for i, tv in enumerate(all_tab_vars) if tv not in tabs_to_remove]
    kept_labels = [all_labels[i] for i in keep_indices if i < len(all_labels)]
    new_tab_count = len(keep_indices)
    new_tab_vars = [f"tab{i+1}" for i in range(new_tab_count)]

    print(f"Kept tabs: {new_tab_count}")

    # Build old->new mapping
    var_mapping = {}
    for new_idx, old_idx in enumerate(keep_indices):
        old_var = all_tab_vars[old_idx]
        new_var = new_tab_vars[new_idx]
        var_mapping[old_var] = new_var

    # Mark old st.tabs block for removal
    if tabs_assign_start is not None and tabs_assign_end is not None:
        for k in range(tabs_assign_start, tabs_assign_end):
            all_lines_to_remove.add(k)

    # Build new st.tabs block
    indent = "                    "
    new_tabs_block = []
    vars_str = ", ".join(new_tab_vars)
    new_tabs_block.append(f"{indent}{vars_str} = st.tabs([\n")
    for i, label in enumerate(kept_labels):
        comma = ","
        new_tabs_block.append(f'{indent}    "{label}"{comma}\n')
    new_tabs_block.append(f"{indent}])\n\n")

    # --- Step 5: Write new file ---
    new_lines = []
    insert_done = False
    for i, line in enumerate(lines):
        if i in all_lines_to_remove:
            if tabs_assign_start is not None and i == tabs_assign_start and not insert_done:
                new_lines.extend(new_tabs_block)
                insert_done = True
            continue

        # Rename tab variables in "with tabN:" lines
        tw = re.match(r"^(\s+)with (tab\d+):", line)
        if tw and tw.group(2) in var_mapping:
            old = tw.group(2)
            new = var_mapping[old]
            line = line.replace(f"with {old}:", f"with {new}:")

        new_lines.append(line)

    # Clean consecutive blank lines
    cleaned = []
    blank_count = 0
    for line in new_lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
        f.writelines(cleaned)

    print(f"\nResult: {original_count} -> {len(cleaned)} lines ({original_count - len(cleaned)} removed)")


if __name__ == "__main__":
    main()

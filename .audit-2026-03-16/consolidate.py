#!/usr/bin/env python3
"""
Consolidation script: Remove redundant phase methods from financial_analyzer.py
and insights_page.py based on dedup_map.json cluster analysis.

For each cluster, keeps the FIRST method listed and removes the rest.
Also removes corresponding dataclasses, UI tabs, and identifies test files for deletion.
"""
import json
import re
import os
import sys
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEDUP_MAP = BASE_DIR.parent / ".loki" / "memory" / "semantic" / "dedup_map.json"
ANALYZER_FILE = BASE_DIR / "financial_analyzer.py"
INSIGHTS_FILE = BASE_DIR / "insights_page.py"
TESTS_DIR = BASE_DIR / "tests"


def load_dedup_map():
    """Parse dedup_map.json and build keep/remove lists."""
    with open(DEDUP_MAP) as f:
        data = json.load(f)

    methods_to_keep = []
    methods_to_remove = []

    for cluster in data["clusters"]:
        methods = cluster["methods"]
        if len(methods) >= 1:
            methods_to_keep.append(methods[0])
        if len(methods) > 1:
            methods_to_remove.extend(methods[1:])

    return methods_to_keep, methods_to_remove


def find_method_boundaries(lines):
    """Find all 4-space-indented method definitions in CharlieAnalyzer.

    Returns dict: method_name -> (start_line_idx, end_line_idx, return_type)
    where end_line_idx is exclusive (first line of next section).
    """
    method_pattern = re.compile(r"^    def (\w+)\(self[^)]*\)\s*(?:->\s*(\w+))?\s*:")
    method_starts = []

    for i, line in enumerate(lines):
        m = method_pattern.match(line)
        if m:
            method_starts.append((i, m.group(1), m.group(2)))

    boundaries = {}
    for idx, (start, name, ret_type) in enumerate(method_starts):
        if idx + 1 < len(method_starts):
            end = method_starts[idx + 1][0]
        else:
            # Last method - find end of class or file
            end = len(lines)
            for j in range(start + 1, len(lines)):
                # Top-level code (non-indented, non-blank, non-comment)
                stripped = lines[j].strip()
                if stripped and not lines[j].startswith(" ") and not lines[j].startswith("\t"):
                    if not stripped.startswith("#"):
                        end = j
                        break

        # Trim trailing blank lines (include them in removal)
        boundaries[name] = (start, end, ret_type)

    return boundaries


def find_dataclass_boundaries(lines):
    """Find all @dataclass class definitions at module level (0-indent).

    Returns dict: class_name -> (start_line_idx, end_line_idx)
    where start includes the @dataclass decorator.
    """
    boundaries = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == "@dataclass":
            start = i
            # Find class name on next non-blank line
            class_name = None
            for j in range(i + 1, min(i + 5, len(lines))):
                cm = re.match(r"^class (\w+)", lines[j])
                if cm:
                    class_name = cm.group(1)
                    break

            if class_name:
                # Find end: next @dataclass, class, def, or other top-level construct
                end = len(lines)
                for j in range(start + 2, len(lines)):
                    stripped = lines[j].strip()
                    if not stripped:
                        continue  # skip blank lines
                    # Check if at top level (no indent)
                    if not lines[j].startswith(" ") and not lines[j].startswith("\t"):
                        if stripped.startswith("@") or stripped.startswith("class ") or stripped.startswith("def "):
                            end = j
                            break
                        if stripped.startswith("#") and j + 1 < len(lines):
                            # Could be a comment before next class, peek ahead
                            continue
                        # Some other top-level code
                        end = j
                        break

                boundaries[class_name] = (start, end)
                i = end
            else:
                i += 1
        else:
            i += 1

    return boundaries


def process_analyzer(methods_to_remove):
    """Remove duplicate methods and their dataclasses from financial_analyzer.py."""
    with open(ANALYZER_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    original_count = len(lines)
    print(f"  Analyzer: {original_count} lines loaded")

    method_bounds = find_method_boundaries(lines)
    dataclass_bounds = find_dataclass_boundaries(lines)

    lines_to_remove = set()
    removed_methods = []
    removed_dataclasses = []
    not_found = []

    for method_name in methods_to_remove:
        if method_name not in method_bounds:
            not_found.append(method_name)
            continue

        start, end, ret_type = method_bounds[method_name]
        for k in range(start, end):
            lines_to_remove.add(k)
        removed_methods.append(method_name)

        # Remove corresponding dataclass
        if ret_type and ret_type in dataclass_bounds:
            dc_start, dc_end = dataclass_bounds[ret_type]
            for k in range(dc_start, dc_end):
                lines_to_remove.add(k)
            removed_dataclasses.append(ret_type)

    # Write filtered file
    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

    # Clean up excessive blank lines (3+ consecutive -> 2)
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

    with open(ANALYZER_FILE, "w", encoding="utf-8") as f:
        f.writelines(cleaned)

    print(f"  Removed {len(removed_methods)} methods, {len(removed_dataclasses)} dataclasses")
    print(f"  Lines: {original_count} -> {len(cleaned)} ({original_count - len(cleaned)} removed)")
    if not_found:
        print(f"  WARNING: {len(not_found)} methods not found: {not_found[:5]}...")

    return removed_methods


def process_insights_page(methods_to_remove):
    """Remove duplicate tabs and render methods from insights_page.py."""
    with open(INSIGHTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    original_count = len(lines)
    print(f"  Insights: {original_count} lines loaded")

    # Step 1: Find all _render_* method definitions and what analyzer method they call
    render_methods = {}  # render_method_name -> (start, end, analyzer_method_called)
    render_pattern = re.compile(r"^    def (_render_\w+)\(self")
    analyzer_call_pattern = re.compile(r"self\.analyzer\.(\w+_analysis)\(")

    render_starts = []
    for i, line in enumerate(lines):
        m = render_pattern.match(line)
        if m:
            render_starts.append((i, m.group(1)))

    for idx, (start, name) in enumerate(render_starts):
        # Find end of this render method
        if idx + 1 < len(render_starts):
            end = render_starts[idx + 1][0]
        else:
            end = len(lines)

        # Find which analyzer method this render method calls
        analyzer_method = None
        for j in range(start, min(end, start + 50)):
            am = analyzer_call_pattern.search(lines[j])
            if am:
                analyzer_method = am.group(1)
                break

        render_methods[name] = (start, end, analyzer_method)

    # Step 2: Find which render methods to remove
    # A render method should be removed if it calls a removed analyzer method
    methods_to_remove_set = set(methods_to_remove)
    render_methods_to_remove = set()

    for render_name, (start, end, analyzer_method) in render_methods.items():
        if analyzer_method and analyzer_method in methods_to_remove_set:
            render_methods_to_remove.add(render_name)

    print(f"  Found {len(render_methods_to_remove)} render methods to remove")

    # Step 3: Find tab blocks that call removed render methods
    # Pattern: "with tabN:\n    self._render_method(df)"
    tab_block_pattern = re.compile(r"^\s+with (tab\d+):")
    tab_render_call = re.compile(r"^\s+self\.(_render_\w+)\(")

    tab_blocks_to_remove = set()  # tab variable names
    tab_lines_to_remove = set()

    i = 0
    while i < len(lines):
        tb = tab_block_pattern.match(lines[i])
        if tb:
            tab_var = tb.group(1)
            # Check the next few lines for the render call
            block_start = i
            block_end = i + 1
            render_call = None

            for j in range(i + 1, min(i + 10, len(lines))):
                rc = tab_render_call.match(lines[j])
                if rc:
                    render_call = rc.group(1)
                    block_end = j + 1
                    break
                # If we hit another "with tab" or something at same/lower indent, stop
                if tab_block_pattern.match(lines[j]):
                    break
                block_end = j + 1

            if render_call and render_call in render_methods_to_remove:
                tab_blocks_to_remove.add(tab_var)
                for k in range(block_start, block_end):
                    tab_lines_to_remove.add(k)
                # Also include any blank lines after the block
                for k in range(block_end, min(block_end + 3, len(lines))):
                    if lines[k].strip() == "":
                        tab_lines_to_remove.add(k)
                    else:
                        break

        i += 1

    # Step 4: Mark render method definitions for removal
    for render_name in render_methods_to_remove:
        if render_name in render_methods:
            start, end, _ = render_methods[render_name]
            for k in range(start, end):
                tab_lines_to_remove.add(k)

    # Step 5: Rebuild the tab variable assignment line and labels list
    # Find the st.tabs() call - it spans multiple lines
    tabs_start = None
    tabs_end = None
    for i, line in enumerate(lines):
        if "= st.tabs([" in line or "st.tabs([" in line:
            tabs_start = i
        if tabs_start is not None and tabs_end is None:
            if "])" in line:
                tabs_end = i + 1
                break

    if tabs_start is not None and tabs_end is not None:
        # Parse existing tab variables from the assignment line
        assignment_line = lines[tabs_start]
        # Extract tab variable names: tab1, tab2, ..., tab367
        tab_vars_match = re.search(r"(tab\d+(?:,\s*tab\d+)*)", assignment_line)

        # Parse existing tab labels
        labels = []
        label_pattern = re.compile(r'"([^"]*)"')
        for j in range(tabs_start, tabs_end):
            for lm in label_pattern.finditer(lines[j]):
                labels.append(lm.group(1))

        # Extract all tab var names in order
        all_tab_vars = []
        for vm in re.finditer(r"tab(\d+)", assignment_line):
            all_tab_vars.append(f"tab{vm.group(1)}")

        # Build keep lists (indices that are NOT removed)
        keep_indices = []
        for idx, tv in enumerate(all_tab_vars):
            if tv not in tab_blocks_to_remove:
                keep_indices.append(idx)

        kept_labels = [labels[i] for i in keep_indices if i < len(labels)]

        # Build new tab variable names (renumbered 1..N)
        new_tab_count = len(keep_indices)
        new_tab_vars = [f"tab{i+1}" for i in range(new_tab_count)]

        # Build variable mapping: old tab var -> new tab var
        var_mapping = {}
        for new_idx, old_idx in enumerate(keep_indices):
            old_var = all_tab_vars[old_idx]
            new_var = new_tab_vars[new_idx]
            var_mapping[old_var] = new_var

        # Mark the entire old st.tabs() block for removal
        for k in range(tabs_start, tabs_end):
            tab_lines_to_remove.add(k)

        # Build new st.tabs() block
        indent = "                    "
        new_tabs_lines = []
        # Variable assignment
        vars_str = ", ".join(new_tab_vars)
        new_tabs_lines.append(f"{indent}{vars_str} = st.tabs([\n")
        for i, label in enumerate(kept_labels):
            comma = "," if i < len(kept_labels) - 1 else ","
            new_tabs_lines.append(f'{indent}    "{label}"{comma}\n')
        new_tabs_lines.append(f"{indent}])\n")

        # Also need to rename tab references in the "with tabN:" blocks
        # We'll handle this after removing lines

    # Step 6: Write filtered file
    new_lines = []
    insert_done = False
    for i, line in enumerate(lines):
        if i in tab_lines_to_remove:
            # Insert new tabs block at the position of old tabs start
            if tabs_start is not None and i == tabs_start and not insert_done:
                new_lines.extend(new_tabs_lines)
                insert_done = True
            continue
        # Rename tab variables in "with tabN:" lines
        if var_mapping:
            tw = re.match(r"^(\s+)with (tab\d+):", line)
            if tw:
                old_var = tw.group(1)
                tab_var = tw.group(2)
                if tab_var in var_mapping:
                    line = line.replace(f"with {tab_var}:", f"with {var_mapping[tab_var]}:")
        new_lines.append(line)

    # Clean up excessive blank lines
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

    print(f"  Removed {len(tab_blocks_to_remove)} tab blocks, {len(render_methods_to_remove)} render methods")
    print(f"  Lines: {original_count} -> {len(cleaned)} ({original_count - len(cleaned)} removed)")

    return render_methods_to_remove


def find_test_files_to_delete(methods_to_remove):
    """Find test files corresponding to removed methods."""
    test_files = []

    # Build mapping: method_name -> possible test file patterns
    for method_name in methods_to_remove:
        # Test files are named like test_phase{N}_{short_name}.py
        # We need to find test files that test the removed method
        # Search for files that import or reference the method
        pass

    # Simpler approach: find all test_phase*.py files and check which ones
    # reference removed methods
    all_test_files = sorted(TESTS_DIR.glob("test_phase*.py"))
    files_to_delete = []
    files_to_keep = []

    methods_set = set(methods_to_remove)

    for tf in all_test_files:
        try:
            content = tf.read_text(encoding="utf-8")
        except Exception:
            files_to_keep.append(tf)
            continue

        # Check if this test file references any removed method
        found_removed = False
        for method in methods_set:
            if method in content:
                found_removed = True
                break

        if found_removed:
            files_to_delete.append(tf)
        else:
            files_to_keep.append(tf)

    return files_to_delete, files_to_keep


def delete_test_files(files_to_delete):
    """Delete the identified test files."""
    deleted = 0
    for tf in files_to_delete:
        try:
            tf.unlink()
            deleted += 1
        except Exception as e:
            print(f"  WARNING: Could not delete {tf.name}: {e}")

    print(f"  Deleted {deleted} test files")
    return deleted


def main():
    print("=" * 60)
    print("PHASE CONSOLIDATION SCRIPT")
    print("=" * 60)

    # Verify files exist
    for path in [DEDUP_MAP, ANALYZER_FILE, INSIGHTS_FILE]:
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    # Step 1: Load dedup map
    print("\n[Step 1] Loading dedup_map.json...")
    methods_to_keep, methods_to_remove = load_dedup_map()
    print(f"  Methods to keep: {len(methods_to_keep)}")
    print(f"  Methods to remove: {len(methods_to_remove)}")

    # Step 2: Process financial_analyzer.py
    print("\n[Step 2] Processing financial_analyzer.py...")
    removed_methods = process_analyzer(methods_to_remove)

    # Step 3: Process insights_page.py
    print("\n[Step 3] Processing insights_page.py...")
    removed_renders = process_insights_page(methods_to_remove)

    # Step 4: Find test files
    print("\n[Step 4] Finding test files to delete...")
    files_to_delete, files_to_keep = find_test_files_to_delete(methods_to_remove)
    print(f"  Test files to delete: {len(files_to_delete)}")
    print(f"  Test files to keep: {len(files_to_keep)}")

    if files_to_delete:
        print("\n  Files to delete:")
        for f in files_to_delete[:10]:
            print(f"    - {f.name}")
        if len(files_to_delete) > 10:
            print(f"    ... and {len(files_to_delete) - 10} more")

        delete_test_files(files_to_delete)

    # Summary
    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"  Methods removed from analyzer: {len(removed_methods)}")
    print(f"  Render methods removed from insights: {len(removed_renders)}")
    print(f"  Test files deleted: {len(files_to_delete)}")
    print(f"  Test files kept: {len(files_to_keep)}")


if __name__ == "__main__":
    main()

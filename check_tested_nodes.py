#!/usr/bin/env python
import glob
import re
import sys

total_nodes = 0
tested_nodes = 0
untested_nodes = 0
tested_classes = []
untested_classes = []

for py_file in glob.glob("*.py"):
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            match = re.match(r'^\s*class\s+(Sox\w+Node)\s*:', line.rstrip())
            if match:
                class_name = match.group(1)
                total_nodes += 1
                next_line_tested = False
                if i + 1 < len(lines) and '# Tested:' in lines[i + 1]:
                    next_line_tested = True
                    tested_nodes += 1
                    tested_classes.append(class_name)
                else:
                    untested_nodes += 1
                    untested_classes.append(class_name)
    except Exception as e:
        print(f"Error reading {py_file}: {e}", file=sys.stderr)

print(f"Total Nodes: {total_nodes}")
print(f"Nodes Tested: {tested_nodes}")
print(f"Nodes Untested: {untested_nodes}")
print("=====================")
print("Tested Nodes:")
if tested_classes:
    current_line = []
    for cls in tested_classes:
        test_line = ", ".join(current_line + [cls])
        if len(test_line) > 80 and current_line:
            print(", ".join(current_line))
            current_line = [cls]
        else:
            current_line.append(cls)
    if current_line:
        print(", ".join(current_line))
else:
    print("None")
print("=====================")
print("Untested Nodes:")
if untested_classes:
    current_line = []
    for cls in untested_classes:
        test_line = ", ".join(current_line + [cls])
        if len(test_line) > 80 and current_line:
            print(", ".join(current_line))
            current_line = [cls]
        else:
            current_line.append(cls)
    if current_line:
        print(", ".join(current_line))
else:
    print("None")

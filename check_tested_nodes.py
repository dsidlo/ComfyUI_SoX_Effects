import glob
import re

total_nodes = 0
tested_nodes = 0
untested_nodes = 0

for py_file in glob.glob("*.py"):
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if re.match(r'^\s*class\s+Sox\w+Node\s*:', line.rstrip()):
                total_nodes += 1
                next_line_tested = False
                if i + 1 < len(lines) and '# Tested:' in lines[i + 1]:
                    next_line_tested = True
                    tested_nodes += 1
                else:
                    untested_nodes += 1
    except Exception as e:
        print(f"Error reading {py_file}: {e}", file=sys.stderr)

print(f"Total Nodes: {total_nodes}")
print(f"Nodes Tested: {tested_nodes}")
print(f"Nodes Untested: {untested_nodes}")

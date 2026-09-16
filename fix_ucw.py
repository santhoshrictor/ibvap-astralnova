"""Fix remaining deprecated use_container_width calls in app.py."""
with open("app.py", "r", encoding="utf-8") as f:
    src = f.read()

old1 = 'channels="BGR",use_container_width=True'
new1 = 'channels="BGR",width="stretch"'
src  = src.replace(old1, new1)

old2 = "use_container_width=True,hide_index=True,height=120)"
new2 = 'width="stretch",hide_index=True,height=120)'
src  = src.replace(old2, new2)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(src)

import ast
ast.parse(src)
import re
remaining = len(re.findall("use_container_width", src))
print(f"Done. Syntax OK. Remaining use_container_width: {remaining} (buttons - OK to keep)")

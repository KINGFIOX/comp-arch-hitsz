import os
from pathlib import Path

c_extensions = (".c", ".h", ".cpp")

path = Path('src')

for file in path.rglob('*'):
    if file.is_file() and file.suffix in c_extensions:
        os.system("clang-format -i -style=file " + str(file))
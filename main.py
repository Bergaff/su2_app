import os
import sys

# запуск из любой папки: корень проекта — в sys.path и cwd
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.main_window import main

if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import rootutils

# このファイルがある dir の絶対パスを取得
abs_dir = Path(sys.modules[__name__].__file__).resolve().parent
print(abs_dir)

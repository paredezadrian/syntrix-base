from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen


TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def download_text8_mini(
    output_path: str = "data/text8-mini.txt", size_mb: int = 5
) -> str:
    """Download the text8 corpus and write a small substring as a mini dataset.

    - Downloads `text8.zip` (~31 MB), extracts `text8` (~100 MB raw), then writes the
      first `size_mb` megabytes into `output_path`.
    - Returns the path written.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with urlopen(TEXT8_URL) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        with zf.open("text8") as z:
            raw = z.read()
    nbytes = int(size_mb * 1024 * 1024)
    subset = raw[:nbytes]
    with open(output_path, "wb") as f:
        f.write(subset)
    return output_path

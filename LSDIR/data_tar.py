import tarfile
import os
from pathlib import Path

# LSDIR 다운로드 경로
base_dir = Path(r"C:\Users\JaegyunIm\Desktop\super_resolution\LSDIR\LSDIR_downloaded")
extract_dir = base_dir / "extracted"
extract_dir.mkdir(exist_ok=True)

# 모든 tar.gz 파일 순회
for tar_path in sorted(base_dir.glob("*.tar.gz")):
    print(f"Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print(f"Done: {tar_path.name}")

print("✅ 모든 압축 해제가 완료되었습니다.")
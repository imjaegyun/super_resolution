import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# 다운로드 받을 저장소 ID
repo_id = "ofsoundof/LSDIR"

# 다운로드 받을 파일 목록 (스크린샷에 보이는 .tar.gz 파일들)
# 필요한 파일만 남기고 목록에서 제거해도 됩니다.
filenames = [
    "shard-00.tar.gz", "shard-01.tar.gz", "shard-02.tar.gz", "shard-03.tar.gz",
    "shard-04.tar.gz", "shard-05.tar.gz", "shard-06.tar.gz", "shard-07.tar.gz",
    "shard-08.tar.gz", "shard-09.tar.gz", "shard-10.tar.gz", "shard-11.tar.gz",
    "shard-12.tar.gz", "shard-13.tar.gz", "shard-14.tar.gz", "shard-15.tar.gz",
    "shard-16.tar.gz",
    "train_v2.tar.gz",
    "train_v3.tar.gz",
    "train_v4.tar.gz",
    "val1.tar.gz"
]

# 파일을 저장할 로컬 폴더 이름
# (폴더가 없으면 자동으로 생성됩니다)
local_dir = "LSDIR_downloaded"
os.makedirs(local_dir, exist_ok=True)
print(f"'{local_dir}' 폴더에 파일을 다운로드합니다.")

# tqdm을 사용하여 다운로드 진행 상황을 표시하며 파일 다운로드
for filename in tqdm(filenames, desc="전체 진행률"):
    # 파일이 이미 있으면 다운로드 건너뛰기
    if os.path.exists(os.path.join(local_dir, filename)):
        print(f"'{filename}' 파일이 이미 존재하므로 건너뜁니다.")
        continue
    
    try: 
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",  # 저장소가 'model' 타입으로 생성된 경우
            local_dir=local_dir,
            local_dir_use_symlinks=False # Windows 호환성을 위해 False로 설정
        )
        print(f"'{filename}' 다운로드 완료.")
    except Exception as e:
        print(f"'{filename}' 다운로드 중 오류 발생: {e}")

print("모든 파일 다운로드가 완료되었습니다.")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from PIL import Image

def get_image_info(path: str):
    with Image.open(path) as im:
        size = im.size  # (width, height)
        dpi = im.info.get("dpi")  # (x_dpi, y_dpi) or None
    return size, dpi

def main():
    parser = argparse.ArgumentParser(description="두 이미지의 해상도(픽셀 크기)가 같은지 확인합니다.")
    parser.add_argument("img1", help="첫 번째 이미지 경로")
    parser.add_argument("img2", help="두 번째 이미지 경로")
    args = parser.parse_args()

    try:
        size1, dpi1 = get_image_info(args.img1)
    except Exception as e:
        print(f"[오류] 첫 번째 이미지를 여는 중 문제 발생: {e}")
        return
    try:
        size2, dpi2 = get_image_info(args.img2)
    except Exception as e:
        print(f"[오류] 두 번째 이미지를 여는 중 문제 발생: {e}")
        return

    w1, h1 = size1
    w2, h2 = size2

    print("=== 이미지 정보 ===")
    print(f"이미지 1: {args.img1}")
    print(f" - 픽셀 크기: {w1} × {h1} (총 {w1*h1} px)")
    print(f" - DPI(있으면 표시): {dpi1}")
    print(f"이미지 2: {args.img2}")
    print(f" - 픽셀 크기: {w2} × {h2} (총 {w2*h2} px)")
    print(f" - DPI(있으면 표시): {dpi2}")

    if size1 == size2:
        print("\n결론: ✅ 두 이미지의 **픽셀 해상도는 같습니다.**")
    else:
        print("\n결론: ❌ 두 이미지의 **픽셀 해상도는 다릅니다.**")

    # 참고: DPI는 출력(인쇄) 밀도 정보라서 픽셀 해상도와는 별개입니다.
    if dpi1 != dpi2:
        print("참고: 두 이미지의 DPI 메타데이터는 서로 다를 수 있지만, 이는 픽셀 해상도와는 무관합니다.")

if __name__ == "__main__":
    main()

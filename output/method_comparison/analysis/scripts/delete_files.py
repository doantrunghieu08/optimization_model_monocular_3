#!/usr/bin/env python3
"""
Xoá các tệp .mp4, .pth, .pkl trong các thư mục con của một thư mục.

Cách dùng:
    python delete_files.py <thư_mục>              # chạy thử: chỉ liệt kê, KHÔNG xoá
    python delete_files.py <thư_mục> --delete     # xoá thật (có hỏi xác nhận)
    python delete_files.py <thư_mục> --delete -y  # xoá thật, không hỏi
    python delete_files.py <thư_mục> --include-root   # xoá cả tệp nằm ngay ở thư mục gốc
    python delete_files.py <thư_mục> --ext .mp4 .pth  # chỉ định đuôi tệp khác
"""

import argparse
import sys
from pathlib import Path

DEFAULT_EXTS = [".mp4", ".pth", ".pkl"]


def human_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def find_files(root: Path, exts: set, include_root: bool):
    """Tìm đệ quy các tệp có đuôi trong exts (không phân biệt hoa/thường)."""
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        if p.suffix.lower() not in exts:
            continue
        if not include_root and p.parent == root:
            continue  # bỏ qua tệp nằm ngay ở thư mục gốc
        yield p


def main():
    parser = argparse.ArgumentParser(description="Xoá tệp .mp4/.pth/.pkl trong các thư mục con.")
    parser.add_argument("root", help="Thư mục gốc cần quét")
    parser.add_argument("--delete", action="store_true", help="Xoá thật (mặc định chỉ chạy thử)")
    parser.add_argument("-y", "--yes", action="store_true", help="Không hỏi xác nhận")
    parser.add_argument("--include-root", action="store_true",
                        help="Xoá cả tệp nằm trực tiếp trong thư mục gốc")
    parser.add_argument("--ext", nargs="+", default=DEFAULT_EXTS,
                        help=f"Danh sách đuôi tệp (mặc định: {' '.join(DEFAULT_EXTS)})")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Lỗi: '{root}' không phải thư mục.")

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    files = sorted(find_files(root, exts, args.include_root))

    if not files:
        print("Không tìm thấy tệp nào phù hợp.")
        return

    total = 0
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"{human_size(size):>10}  {f.relative_to(root)}")
    print(f"\nTổng: {len(files)} tệp, {human_size(total)}")

    if not args.delete:
        print("\n[Chạy thử] Chưa xoá gì. Thêm --delete để xoá thật.")
        return

    if not args.yes:
        ans = input(f"\nXoá {len(files)} tệp trên? Gõ 'yes' để xác nhận: ").strip().lower()
        if ans != "yes":
            print("Đã huỷ.")
            return

    deleted, freed, errors = 0, 0, 0
    for f in files:
        try:
            size = f.stat().st_size
            f.unlink()
            deleted += 1
            freed += size
        except OSError as e:
            errors += 1
            print(f"Không xoá được {f}: {e}")

    print(f"\nĐã xoá {deleted} tệp, giải phóng {human_size(freed)}."
          + (f" Lỗi: {errors} tệp." if errors else ""))


if __name__ == "__main__":
    main()
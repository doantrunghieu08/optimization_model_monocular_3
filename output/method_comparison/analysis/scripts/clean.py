
import os

def clean_directory():
    folder = input("Nhập đường dẫn thư mục X: ").strip().strip('"')

    if not os.path.isdir(folder):
        print("Thư mục không tồn tại!")
        return

    deleted_count = 0
    kept_count = 0

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        # Chỉ xử lý file, không xử lý thư mục con
        if os.path.isfile(file_path):
            extension = os.path.splitext(filename)[1].lower()

            if extension not in [".json", ".txt"]:
                try:
                    os.remove(file_path)
                    print(f"[DELETED] {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[ERROR] {filename}: {e}")
            else:
                print(f"[KEPT] {filename}")
                kept_count += 1

    print("\n========== SUMMARY ==========")
    print(f"Files deleted: {deleted_count}")
    print(f"Files kept: {kept_count}")


if __name__ == "__main__":
    clean_directory()




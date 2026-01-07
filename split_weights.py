import os

# 配置：每块大小 80MB (留出余量，确保小于GitHub的100MB限制)
CHUNK_SIZE = 80 * 1024 * 1024 
SOURCE_FILE = "resources/best_model.pth"

def split_file():
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ 错误：找不到文件 {SOURCE_FILE}")
        return

    print(f"🔪 开始切分 {SOURCE_FILE} ...")
    
    with open(SOURCE_FILE, 'rb') as f:
        part_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            # 生成文件名：best_model.pth.part0, best_model.pth.part1 ...
            part_name = f"{SOURCE_FILE}.part{part_num}"
            with open(part_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            print(f"   -> 生成分卷: {part_name} ({len(chunk)/1024/1024:.2f} MB)")
            part_num += 1

    print("\n✅ 切分完成！")
    print("⚠️  重要提示：请务必删除或移走原始的 best_model.pth 文件，只保留 .part 文件！")

if __name__ == "__main__":
    split_file()
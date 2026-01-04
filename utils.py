import os

def load_map_from_file(filepath):
    """
    从指定的文本文件中加载地图。

    地图文件格式:
    - 每行代表地图的一行。
    - 字符之间可以是空格分隔或不分隔。
    - 示例:
      S . . . . .
      . W W W W .

    Args:
        filepath (str): 地图文件的路径。

    Returns:
        list[list[str]]: 地图的列表的列表表示。

    Raises:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果地图文件为空或格式不正确。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Map file not found: {filepath}")

    grid_map = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line: # 忽略空行
                # 尝试按空格分割, 如果没有空格则按字符分割
                if ' ' in stripped_line:
                    grid_map.append(stripped_line.split())
                else:
                    grid_map.append(list(stripped_line)) # 按字符分割

    if not grid_map:
        raise ValueError(f"Map file is empty or malformed: {filepath}")

    # 简单检查每行长度是否一致 (可选, 但建议)
    # first_row_len = len(grid_map[0])
    # if not all(len(row) == first_row_len for row in grid_map):
    #     raise ValueError("Map rows have inconsistent lengths.")

    print(f"Map loaded successfully from {filepath}")
    return grid_map

if __name__ == '__main__':
    # 简单的测试
    test_map_path = 'maps/map1.txt' # 假设 map1.txt 存在于 maps 目录下
    try:
        loaded_map = load_map_from_file(test_map_path)
        for row in loaded_map:
            print(row)
    except Exception as e:
        print(f"Error loading map: {e}")

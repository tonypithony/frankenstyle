import os
import random
import shutil
from pathlib import Path

def move_random_files(src_dir, dst_dir, n):
    # Преобразуем пути в объекты Path для удобства
    src = Path(src_dir)
    dst = Path(dst_dir)

    # Убедимся, что целевой каталог существует
    dst.mkdir(parents=True, exist_ok=True)

    # Получаем список всех файлов (не папок) в исходной директории
    files = [f for f in src.iterdir() if f.is_file()]

    if not files:
        print("В исходной директории нет файлов.")
        return

    # Ограничиваем n количеством доступных файлов
    n = min(n, len(files))

    # Случайно выбираем n файлов без повторений
    selected_files = random.sample(files, n)

    # Перемещаем каждый выбранный файл
    for file_path in selected_files:
        shutil.move(str(file_path), str(dst / file_path.name))
        print(f"Перемещён: {file_path.name}")

# Пример использования:
if __name__ == "__main__":
    for i in range(1,6):
        source_directory = f"/media/pendrive/styleclasses/{i}/"
        destination_directory = f"/media/pendrive/styleclasses/train/{i}/"
        number_of_files = 200  # сколько файлов переместить
        move_random_files(source_directory, destination_directory, number_of_files)
        
        destination_directory = f"/media/pendrive/styleclasses/validation/{i}/"
        number_of_files = 100  # сколько файлов переместить
        move_random_files(source_directory, destination_directory, number_of_files)
        
        destination_directory = f"/media/pendrive/styleclasses/test/{i}/"
        number_of_files = 40  # сколько файлов переместить
        move_random_files(source_directory, destination_directory, number_of_files)
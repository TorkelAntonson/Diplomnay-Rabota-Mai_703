# create_dirs.py
import os

# Создаем необходимые директории
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Директории созданы")
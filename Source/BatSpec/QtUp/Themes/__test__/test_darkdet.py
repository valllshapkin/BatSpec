import darkdetect
import threading
import time

def on_theme_changed(theme: str):
    print(f"\n[CALLBACK] 🔔 Сработал слушатель darkdetect! Новая тема: {theme}")

def main():
    print("=== Минимальный тест библиотеки darkdetect ===")
    print(f"Текущая тема (darkdetect.theme()): {darkdetect.theme()}")
    print(f"darkdetect.isDark() возвращает: {darkdetect.isDark()}")
    print(f"darkdetect.isLight() возвращает: {darkdetect.isLight()}")
    
    print("\nЗапуск фонового слушателя...")
    t = threading.Thread(target=darkdetect.listener, args=(on_theme_changed,), daemon=True)
    t.start()
    
    print("Слушатель запущен. Попробуй переключить тему ОС.")
    print("Нажми Ctrl+C для выхода...\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nВыход...")

if __name__ == "__main__":
    main()
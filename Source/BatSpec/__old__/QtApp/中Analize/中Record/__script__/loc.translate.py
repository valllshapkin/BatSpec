import os
import xml.etree.ElementTree as ET
from deep_translator import GoogleTranslator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DIRECTORY = Path(__file__).parent.parent / "__locale__"

def get_lang_code(filename):
    locale = filename.replace("this_", "").replace(".ts", "")
    parts = locale.split("_")
    lang = parts[0]

    # Исключения для Google Translate
    if locale == "zh_CN": return "zh-CN"
    if locale == "zh_TW": return "zh-TW"
    if lang == "he": return "iw"
    if lang == "nb": return "no"
    return lang

def process_file(filename):
    """Обрабатывает один файл, возвращает (filename, success, message)."""
    filepath = os.path.join(DIRECTORY, filename)
    lang_code = get_lang_code(filename)

    try:
        translator = GoogleTranslator(source='auto', target=lang_code)
        tree = ET.parse(filepath)
        root = tree.getroot()
        changes_made = False

        for message in root.findall(".//message"):
            source = message.find("source")
            translation = message.find("translation")

            if source is not None and source.text:
                if translation is not None and (not translation.text or translation.get("type") == "unfinished"):
                    original_text = source.text
                    try:
                        translated_text = translator.translate(original_text)
                        translation.text = translated_text
                        if "type" in translation.attrib:
                            del translation.attrib["type"]
                        changes_made = True
                    except Exception as e:
                        return (filename, False, f"Ошибка перевода '{original_text}': {e}")

        if changes_made:
            tree.write(filepath, encoding="utf-8", xml_declaration=True)
            return (filename, True, "✅ Обновлён")
        else:
            return (filename, True, "⚡ Нет строк для перевода")
    except Exception as e:
        return (filename, False, f"❌ Ошибка: {e}")

def translate_ts_files(max_workers=12):
    """Параллельная обработка всех .ts файлов в DIRECTORY."""
    ts_files = [f for f in os.listdir(DIRECTORY) if f.endswith(".ts")]
    if not ts_files:
        print("Нет .ts файлов для обработки.")
        return

    print(f"Найдено {len(ts_files)} файлов. Запускаем обработку с {max_workers} потоками...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Отправляем задачи
        futures = {executor.submit(process_file, filename): filename for filename in ts_files}
        # Обрабатываем результаты по мере завершения
        for future in as_completed(futures):
            filename, success, message = future.result()
            print(f"{filename}: {message}")

if __name__ == "__main__":
    # ВНИМАНИЕ: Сделайте бэкап папки перед запуском!
    translate_ts_files()
import os
import xml.etree.ElementTree as ET
from deep_translator import GoogleTranslator
from pathlib import Path
# Путь к вашей папке с переводами
DIRECTORY = Path(__file__).parent.parent / "__locale__"

def get_lang_code(filename):
    # Убираем "this_" и ".ts", оставляем "ru_RU"
    locale = filename.replace("this_", "").replace(".ts", "")
    parts = locale.split("_")
    
    lang = parts[0] # берем первые две буквы (ru, de, fr)
    
    # --- ИСКЛЮЧЕНИЯ ДЛЯ GOOGLE TRANSLATE ---
    if locale == "zh_CN": return "zh-CN"
    if locale == "zh_TW": return "zh-TW"
    if lang == "he": return "iw"  # Иврит
    if lang == "nb": return "no"  # Норвежский
    
    return lang

def translate_ts_files():
    for filename in os.listdir(DIRECTORY):
        if not filename.endswith(".ts"):
            continue

        filepath = os.path.join(DIRECTORY, filename)
        lang_code = get_lang_code(filename)
        
        print(f"Обработка файла {filename} (Язык: {lang_code})...")
        
        try:
            translator = GoogleTranslator(source='auto', target=lang_code)
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            changes_made = False

            for message in root.findall(".//message"):
                source = message.find("source")
                translation = message.find("translation")
                
                # Если исходный текст есть, а перевода нет или он помечен как unfinished
                if source is not None and source.text:
                    if translation is not None and (not translation.text or translation.get("type") == "unfinished"):
                        
                        original_text = source.text
                        try:
                            translated_text = translator.translate(original_text)
                            translation.text = translated_text
                            
                            # Убираем атрибут unfinished, чтобы Qt понял, что перевод готов
                            if "type" in translation.attrib:
                                del translation.attrib["type"]
                                
                            changes_made = True
                        except Exception as e:
                            print(f"  Ошибка перевода '{original_text}': {e}")

            if changes_made:
                tree.write(filepath, encoding="utf-8", xml_declaration=True)
                print(f"✅ Файл {filename} успешно обновлен.")
            else:
                print(f"⚡ В файле {filename} нет строк для перевода.")
                
        except Exception as e:
            print(f"❌ Ошибка при обработке {filename}: {e}")

if __name__ == "__main__":
    # ВНИМАНИЕ: Сделайте бэкап папки перед запуском!
    translate_ts_files()
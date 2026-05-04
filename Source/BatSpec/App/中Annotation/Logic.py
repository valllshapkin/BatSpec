import uuid
from pathlib import Path
from PySide6.QtCore import QObject
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ScriptDir = Path(__file__).resolve().parent

# ==============================================================================
# 1. ЯВНАЯ ИНИЦИАЛИЗАЦИЯ И КОНФИГУРАЦИЯ БАЗЫ ДАННЫХ
# В точности по твоему примеру.
# ==============================================================================

db_path = ScriptDir / "annotation_local.db"
engine = create_engine(f"sqlite:///{db_path.as_posix()}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Шаг 1: Определяем локальный Base
class AppBase(DeclarativeBase): pass

# Шаг 2: Импортируем ТОЛЬКО конфиг
from BatAnnotation import config as BatAnnotationConfig

# Шаг 3: Назначаем Base ДО импорта остальных модулей
BatAnnotationConfig.BASE = AppBase

# Шаг 4: Теперь можно безопасно импортировать всё остальное
from BatAnnotation.manage import create_all
from BatAnnotation.CommonSeeds.Core import seed_all as seed_core
from BatAnnotation.CommonSeeds.EuropeGeneral import seed_all as seed_eu
from BatAnnotation.Tables import Recording
from BatAnnotation.API import AnnotationManager
from BatAnnotation.QtModels import QtLookups, QtRecording

# Создаем таблицы, если их нет
create_all(engine)

# ==============================================================================
# 2. ЗАПОЛНЕНИЕ БД (СИДЫ) И ГЛОБАЛЬНЫЙ STORE
# ==============================================================================

def setup_synthetic_data(db: Session) -> str:
    """Заполняет БД базовыми справочниками и тестовой записью, если она пуста."""
    seed_core(db)
    seed_eu(db)
    
    rec = db.query(Recording).first()
    if rec:
        return rec.recording_id
        
    print("[Annotation Logic] Создаю синтетические данные для разметки...")
    new_rec = Recording(
        recording_id=str(uuid.uuid4()), 
        filename="synthetic_test_01.wav", 
        sample_rate_hz=384000, 
        duration_s=10.0
    )
    # ... (дальнейшее создание тестовых данных опущено для краткости)
    db.add(new_rec)
    db.commit()
    
    return new_rec.recording_id

class AppStore(QObject):
    def __init__(self, db_session: Session):
        super().__init__()
        self.db = db_session
        self.api = AnnotationManager(db_session)
        
        self.lookups = QtLookups()
        self.recording = QtRecording()
        self.is_dirty = False

    def initialize(self, recording_id: str):
        self.sync_lookups_from_db()
        self.load_recording(recording_id)

    def load_recording(self, recording_id: str):
        mem_rec = self.api.load_recording(recording_id)
        if mem_rec:
            self.recording.load_from(mem_rec)
            self.is_dirty = False

    def sync_lookups_from_db(self):
        mem_lookups = self.api.load_lookups()
        self.lookups.load_from(mem_lookups)

    def save_recording_to_db(self):
        mem_rec_to_save = self.recording.to_memory()
        self.api.save_recording(mem_rec_to_save)
        for seq in self.recording.sequences:
            seq._is_new = False
            for call in seq.calls:
                call._is_new = False
        self.is_dirty = False

# Инициализируем синглтоны модуля при импорте
GLOBAL_DB_SESSION = SessionLocal()
_default_rec_id = setup_synthetic_data(GLOBAL_DB_SESSION)

ANNOTATION_STORE = AppStore(GLOBAL_DB_SESSION)
ANNOTATION_STORE.initialize(_default_rec_id)

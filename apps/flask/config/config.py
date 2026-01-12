import os

class Config:
    """Konfigurasi dasar."""
    # Kunci rahasia untuk sesi, form, dll. Ganti dengan string acak yang kuat!
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess-this-secret-key'
    # URI koneksi MongoDB bisa diletakkan di sini juga
    # MONGO_URI = os.environ.get('MONGO_URI')

class DevelopmentConfig(Config):
    """Konfigurasi untuk Development."""
    DEBUG = True
    # Contoh koneksi ke DB lokal di development
    MONGO_URI = "mongodb://localhost:27017/myDevDatabase"

class ProductionConfig(Config):
    """Konfigurasi untuk Production."""
    DEBUG = False
    # Contoh koneksi ke DB production (bisa jadi di server lain atau MongoDB Atlas)
    # Ambil dari environment variable untuk keamanan
    MONGO_URI = os.environ.get('PROD_MONGO_URI') or "mongodb://localhost:27017/myProdDatabase"

# Dictionary untuk mempermudah pemanggilan config
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}
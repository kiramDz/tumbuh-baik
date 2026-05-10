from functools import wraps
from flask import request, jsonify
from pymongo import MongoClient
from datetime import datetime, timezone
import urllib.parse
import os
import logging

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Coba ambil cookie dari dua kemungkinan nama (production vs localhost)
        raw_token = (request.cookies.get('__Secure-better-auth.session_token') or
                     request.cookies.get('better-auth.session_token'))
        
        if not raw_token:
            # Logging untuk membantu debug: tampilkan semua nama cookie yang tersedia
            logger.warning(f"Unauthorized: No session token cookie. Path={request.path}. Cookies received: {list(request.cookies.keys())}")
            return jsonify({
                "success": False, 
                "error": {"code": "UNAUTHORIZED", "message": "Session required. Please login."}
            }), 401

        # Decode URL encoding (misal %3D menjadi =)
        decoded_token = urllib.parse.unquote(raw_token)
        
        # Jika token mengandung titik (signature), ambil bagian pertama
        session_token = decoded_token.split('.')[0] if '.' in decoded_token else decoded_token
        
        logger.debug(f"Validating token: {session_token[:10]}...")
        
        # Validasi di collection 'session' dengan timezone-aware
        now_utc = datetime.now(timezone.utc)
        session = db.session.find_one({
            "token": session_token,
            "expiresAt": {"$gt": now_utc}
        })
        
        if not session:
            logger.warning(f"Invalid or expired session for token: {session_token[:10]}...")
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_SESSION",
                    "message": "Invalid or expired session. Please login."
                }
            }), 401
        
        # Simpan session ke request context
        request.session = session
        return f(*args, **kwargs)
    return decorated_function
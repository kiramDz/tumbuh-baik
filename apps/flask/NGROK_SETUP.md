# Cara menggunakan ngrok untuk expose Flask lokal:

# 1. Install ngrok dari https://ngrok.com/download

# 2. Jalankan Flask lokal di terminal:
python app.py

# 3. Di terminal lain, expose Flask dengan ngrok:
ngrok http 5001

# 4. Ngrok akan memberikan URL publik seperti:
# https://abcd-123-456.ngrok.io

# 5. Set environment variable di Vercel:
# NEXT_PUBLIC_FLASK_API_URL=https://abcd-123-456.ngrok.io

# 6. Redeploy Next.js atau tunggu auto-deploy

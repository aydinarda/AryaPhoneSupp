# Arya Phones Full Stack App - Setup Guide

## 🎯 Özellikler

### Single Player Leaderboard Sistemi
- ✅ Herkes aynı anda oynar (bağımsız)
- ✅ Aynı supplier pool'undan seçim yapar
- ✅ Aynı user pool'uyla matchlenir
- ✅ Constraint'lere uygunluk kontrolü (feasibility)
- ✅ Real-time leaderboard
- ✅ Profit ve Utility'ye göre sıralama
- ✅ Sadece feasible sonuçları gösterme filtresi

## 📦 Kurulum Adımları

### 1. Supabase Database Setup

1. [Supabase](https://supabase.com) hesabı aç
2. Yeni proje oluştur
3. SQL Editor'e git
4. `database_schema.sql` dosyasındaki SQL'i çalıştır

### 2. Environment Variables

`secrets.toml` dosyası oluştur (proje root'unda):

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key-here"
```

**Supabase credentials'ı nereden bulursun?**
- Supabase Dashboard > Settings > API
- URL: Project URL
- Key: `anon` `public` key

### 3. Backend Server

```powershell
cd arya_fullstack_app/server
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Frontend (Statik Dosyalar)

Frontend otomatik olarak `/` route'unda serve edilir.
Sunucu çalıştıktan sonra: **http://localhost:8000**

## 🎮 Nasıl Oynanır?

### Oyuncu Tarafı

1. **Team Name** ve **Player Name** gir
2. **Supplier seç** (istediğin kadar)
3. **Objective seç** (Max Profit veya Max Utility)
4. **Manual Evaluate** ile sonuçları gör
5. **Submit** ile leaderboard'a kaydet

### Leaderboard

- **Sort by Profit/Utility**: Hangi metriğe göre sıralanacağını seç
- **Show only feasible**: Sadece constraint'leri sağlayan sonuçları göster
- **Top 10**: En iyi 10 sonuç gösterilir
- **Color Coding**:
  - 🟢 Yeşil = Feasible (constraint'ler sağlanıyor)
  - 🔴 Kırmızı = Infeasible (constraint ihlali var)

## 🔧 Constraint'ler

Aşağıdaki kısıtlar otomatik kontrol edilir:

- **Avg Environmental Risk** ≤ 2.75
- **Avg Social Risk** ≤ 3.0
- **En az 1 tedarikçi** seçilmeli

## 📊 Metrikler

Her submission için hesaplanan:

- ✅ **Feasibility** (constraint'lere uygunluk)
- 💰 **Profit** = served_users × (price_per_user - cost_scale × avg_cost)
- 😊 **Utility** = Kullanıcı ağırlıklarına göre toplam fayda
- 📈 **Averages**: env, social, cost, strategic, improvement, low_quality

## 🚀 Deployment (Gelecek)

Henüz deploy edilmedi, sadece local'de çalışıyor.

Deploy için:
- Backend: Railway, Render, Heroku
- Frontend: Netlify, Vercel
- Database: Supabase (zaten cloud)

## 📝 API Endpoints

- `GET /api/config` - Oyun ayarları
- `GET /api/suppliers` - Tedarikçi listesi
- `POST /api/manual-eval` - Manuel değerlendirme
- `POST /api/benchmark` - Gurobi optimal çözümü
- `POST /api/submit` - Leaderboard'a gönder
- `GET /api/leaderboard?sort_by=profit&feasible_only=false` - Lider tablosu

## 🎯 Oyun Mekaniği

### Single Player Mode
- Her oyuncu bağımsız oynar
- Aynı supplier ve user pool'u kullanılır
- Herkes aynı constraint'lere tabidir
- Leaderboard'da yarışma vardır ama birbirlerini etkilemezler

### Nasıl Kazanılır?
- **Max Profit Mode**: En yüksek profit'i elde eden kazanır
- **Max Utility Mode**: En yüksek utility'yi elde eden kazanır
- **DİKKAT**: Sadece **feasible** sonuçlar geçerlidir!

## 🐛 Troubleshooting

**Supabase bağlantı hatası?**
- `secrets.toml` dosyasını kontrol et
- Supabase credentials'ın doğru olduğundan emin ol

**Gurobi hatası?**
- Benchmark için Gurobi gerekli
- Manuel evaluate için Gurobi gerekmez

**Leaderboard boş?**
- Henüz kimse submit etmemiş olabilir
- Database bağlantısını kontrol et

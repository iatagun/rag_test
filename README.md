# rag_test# Cümleden SQL: Akıllı Arama Motoru 🚀

## 📜 Proje Tanımı

Bu proje, klasik SQL sorguları yazma zahmetine son verip, sorgu yerine düzgün bir cümle girerek veritabanında en alakalı sonuçları getirmeni sağlar. İşleyiş adımları:

1. **Cümle Girişi**: Kullanıcı doğal dilde bir cümle yazar.
2. **Güçlü Öbek Çıkarımı**: BiLSTM tabanlı bir NER modeli ile cümlenin en güçlü isim öbeği (NP) seçilir. 🏷️
3. **Concordance & Vektörleştirme**: Concordance modu ile öbeğin semantik bağlamları toplanır ve embedding vektörleriyle anlamsal boyutlar çıkarılır. 📊
4. **Dinamik SQL Sorgusu**: Çıkarılan anahtar kelime(ler) ve semantik yakınlık ölçüleri kullanılarak SQL sorgusu otomatik oluşturulur.
5. **Sonuç Listeleme**: Veritabanından en alakalı kayıtlar çekilip ekrana basılır.

> "Artık SELECT \* yazmayı bırak, konuşarak sorgula!" 😜

## ✨ Özellikler

* Doğal dil girişiyle sorgu yapabilme
* BiLSTM tabanlı NP çıkarımı (BIO etiketli eğitim) ✂️
* Concordance ile bağlam temelli anlamsal vektör üretimi
* Dinamik SQL sorgu jenerasyonu
* SQLite gibi hafif veritabanı desteği

## 🏗️ Mimari Yapı

```
[ Kullanıcı ] -> np_inference.py -> en güçlü NP
                   |
                   v
           condordance_model.py -> semantik bağlam vektörleri
                   |
                   v
            sql_query_builder.py -> otomatik SQL
                   |
                   v
          SQLite DB (.db) -> sonuçlar
```

## 💻 Gereksinimler

* Python 3.8+
* pip paketleri:

  ```bash
  pip install torch nltk sklearn numpy sqlite3
  ```
* NLTK veri paketleri:

  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

## 🚀 Kurulum

1. Depoyu klonlayın:

   ```bash
   git clone https://github.com/senin/kendi-projen.git
   cd kendi-projen
   ```
2. Sanal ortam oluştur (opsiyonel ama tavsiye edilir):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```
3. Gerekli paketleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

## 🗄️ Veritabanı Hazırlığı (Dummy SQL)

Proje klasöründe `schema.sql` ve `dummy.db` dosyaları var. Eğer yoksa kendin oluştur:

1. `schema.sql` içeriği:

   ```sql
   CREATE TABLE items (
     id INTEGER PRIMARY KEY,
     title TEXT,
     description TEXT
   );
   INSERT INTO items (title, description) VALUES
     ('Akıllı Telefon', 'Yüksek performanslı, 128GB depolama'),
     ('Dizüstü PC', '16GB RAM, SSD, hafif tasarım'),
     ('Kulaklık', 'Gürültü engelleme özellikli Bluetooth');
   ```
2. SQLite DB oluştur:

   ```bash
   sqlite3 dummy.db < schema.sql
   ```

## ▶️ Kullanım

1. NP modeli eğitimi (ilk sefer için):

   ```bash
   python np_bio_data_gen.py
   ```
2. En güçlü öbeği çıkarma & test:

   ```bash
   python np_inference.py
   ```
3. Concordance & embedding eğitimi:

   ```bash
   python condordance_model.py
   ```
4. Ana script ile sorgu:

   ```bash
   python main.py  # ya da kendi sql_query_builder entegrasyonunu kullan
   ```

> Örnek:
>
> ```bash
> $ python main.py
> Cümleyi girin: "En iyi performansa sahip dizüstü bilgisayarı göster"
> Sonuçlar:
> 1. Dizüstü PC – 16GB RAM, SSD, hafif tasarım
> ```

## 📂 Dosya Yapısı

```
├── condordance_model.py   # Concordance + embedding eğitim
├── np_bio_data_gen.py     # NP çıkarım modeli eğitimi
├── np_inference.py        # Cümleden NP çıkarma
├── test.py                # Embedding benzerlik testleri
├── schema.sql             # Dummy DB şeması
├── dummy.db               # Örnek SQLite DB
└── main.py                # Tüm adımları birleştiren ana script
```

## 🛠️ Geliştirme & Katkı

Meraklıysan, forktan PR at, kodu şapşal testlere sok! 🧪🐒

## 📝 Lisans

MIT © 2025

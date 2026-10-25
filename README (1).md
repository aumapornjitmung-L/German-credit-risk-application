# 💳 German Credit Risk — Model · Explainability · Fairness

แอป Streamlit สำหรับวิเคราะห์ความเสี่ยงสินเชื่อด้วยชุดข้อมูล **Statlog German Credit** (OpenML `data_id=31`)

## สิ่งที่แอปทำ

| ข้อ | งาน | อยู่ที่ |
|---|---|---|
| 1 | โหลดข้อมูลด้วย `fetch_openml(data_id=31)` และ EDA | แท็บ 📊 EDA, `load_german_credit()`, `eda_summary()` |
| 2.1 | target: positive class = `bad`, split แบบ stratified (`random_state=42`) | `make_xy()`, `split_data()` |
| 2.2 | `Pipeline(ColumnTransformer + model)` 3 อัลกอริทึม: Logistic Regression, Random Forest, Gradient Boosting | `build_preprocessor()`, `model_specs()` |
| 2.3 | `GridSearchCV` + `StratifiedKFold(k=5)` บนชุดฝึกเท่านั้น | `tune_model()` |
| 2.4 | threshold ที่ทำให้ expected cost ต่ำสุด (FN:FP = 5:1) จาก out-of-fold predictions | `best_threshold()`, แท็บ 💰 Threshold |
| 3 | Permutation importance + SHAP บนชุดทดสอบ | `perm_importance()`, `shap_values()`, แท็บ 🔍 |
| 4 | Fairness metrics ตาม `sex` และ `age_group` | `fairness_report()`, แท็บ ⚖️ |
| 5 | Streamlit app + deploy ผ่าน GitHub | `app.py` |

**หมายเหตุวิธีการ**
- `sex` ได้จาก `personal_status` (`female ...` → female, ที่เหลือ → male); `age_group` = `<=25`, `26-40`, `>40`
- โมเดลที่ "ดีที่สุด" เลือกจาก OOF expected cost บนชุดฝึก — ชุดทดสอบใช้ประเมินผลครั้งสุดท้ายเท่านั้น (ไม่รั่วไหล)
- Expected cost = (5·FN + 1·FP) / n; ปรับต้นทุนได้ที่ sidebar
- มีตัวเลือกตัด `personal_status` และ `age` ออกจากฟีเจอร์ เพื่อเปรียบเทียบผลต่อ fairness

## โครงสร้างไฟล์

```
├── app.py               # Streamlit UI
├── credit_core.py       # ML logic ทั้งหมด
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

## รันบนเครื่อง

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy บน Streamlit Community Cloud ผ่าน GitHub

1. สร้าง repository ใหม่บน GitHub แล้ว push ไฟล์ทั้งหมด
   ```bash
   git init
   git add .
   git commit -m "German credit risk app"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```
2. เข้า <https://share.streamlit.io> → **Create app** → เชื่อม GitHub
3. เลือก repository, branch `main`, main file path `app.py` (Advanced settings: Python 3.11–3.12)
4. กด **Deploy** — ครั้งแรกจะใช้เวลาฝึกโมเดลประมาณ 1–3 นาที หลังจากนั้นผลถูก cache ไว้
   (ถ้าช้าเกินไป ติ๊ก "โหมดเร็ว" ใน sidebar)

ถ้าเซิร์ฟเวอร์เข้าถึง OpenML ไม่ได้ ให้เลือก "อัปโหลด CSV" ใน sidebar แล้วอัปโหลดไฟล์ที่มีคอลัมน์เดียวกัน (รวม `class`)

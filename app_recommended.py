"""
SmartShop AI — تراكمي لكل عميل
المستخدم يدخل اسمه/معرّفه + فاتورة جديدة → نضيفها لتاريخه →
نحسب RFM تراكمياً من كل فواتيره (الجديدة + القديمة) →
الموديل يتنبأ بالـ Segment الحالي → نوصي بمنتجات من نفس الـ Segment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

st.set_page_config(page_title="SmartShop AI", page_icon="🛍️", layout="wide")

# ──────────────────────────────────────────────
# تحميل البيانات الأصلية (Online Retail) مرة واحدة
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="جاري تحميل بيانات المتجر...")
def load_retail_data():
    possible_names = [
        "Online_Retail_data.xlsx",
        "Online Retail_data.xlsx",
        "Online_Retail_Data.xlsx",
        "online_retail_data.xlsx",
        "Online Retail Data.xlsx",
    ]
    found_path = None
    for name in possible_names:
        if os.path.exists(name):
            found_path = name
            break
    if found_path is None:
        xlsx_files = [f for f in os.listdir(".") if f.lower().endswith(".xlsx")]
        if xlsx_files:
            found_path = xlsx_files[0]
            st.warning(f"⚠️ استخدمت الملف: {found_path}")
        else:
            st.error(
                "❌ لم أجد ملف Excel في نفس المجلد.\n\n"
                f"المجلد الحالي: {os.getcwd()}\n\n"
                f"الملفات الموجودة: {os.listdir('.')}"
            )
            st.stop()

    df = pd.read_excel(found_path)
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["CustomerID"]  = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].str.strip()
    df["Total_price"] = df["Quantity"] * df["UnitPrice"]
    return df


@st.cache_data(show_spinner="جاري تجهيز بيانات RFM...")
def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Total_price", "sum"),
    ).reset_index()
    return rfm


@st.cache_resource(show_spinner="جاري تحميل النموذج...")
def load_models():
    missing = [f for f in ("model_class.pkl", "scaler.pkl") if not os.path.exists(f)]
    if missing:
        st.error(
            f"❌ ملفات مفقودة: {missing}\n\n"
            f"المجلد الحالي: {os.getcwd()}\n\n"
            f"الملفات الموجودة: {os.listdir('.')}"
        )
        st.stop()
    model  = joblib.load("model_class.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data(show_spinner="جاري تصنيف عملاء المتجر...")
def assign_segments_to_all_customers(rfm: pd.DataFrame,
                                      _model, _scaler) -> pd.DataFrame:
    X = rfm[["Recency", "Frequency", "Monetary"]].values
    X_scaled = _scaler.transform(X)
    preds = _model.predict(X_scaled)
    rfm = rfm.copy()
    rfm["Segment"] = [str(p).strip() for p in preds]
    return rfm


@st.cache_data(show_spinner=False)
def build_segment_top_products(df: pd.DataFrame,
                                rfm_with_seg: pd.DataFrame,
                                top_n: int = 15) -> dict:
    merged = df.merge(
        rfm_with_seg[["CustomerID", "Segment"]],
        on="CustomerID", how="inner"
    )
    seg_products = {}
    for seg in merged["Segment"].unique():
        seg_df = merged[merged["Segment"] == seg]
        top = (
            seg_df.groupby("Description")["Quantity"]
            .sum()
            .nlargest(top_n)
            .reset_index()
        )
        prices = df.groupby("Description")["UnitPrice"].median()
        top["UnitPrice"] = top["Description"].map(prices)
        seg_products[seg] = top
    return seg_products


# ──────────────────────────────────────────────
# RFM تراكمي من كل فواتير عميل معيّن (محلي + جديد)
# ──────────────────────────────────────────────
def compute_cumulative_rfm(all_invoices_for_customer: list,
                            reference_date: datetime.date) -> dict:
    """
    all_invoices_for_customer: قائمة فواتير، كل فاتورة:
        {"invoice_no": ..., "date": ..., "lines": [{Description,Quantity,UnitPrice}, ...]}
    reference_date: "اليوم" المحاكى الذي يُحسب Recency بالنسبة له
    """
    rows = []
    for inv in all_invoices_for_customer:
        for line in inv["lines"]:
            rows.append({
                "InvoiceNo":   inv["invoice_no"],
                "InvoiceDate": pd.Timestamp(inv["date"]),
                "Quantity":    line["Quantity"],
                "UnitPrice":   line["UnitPrice"],
            })
    df = pd.DataFrame(rows)
    df["Total_price"] = df["Quantity"] * df["UnitPrice"]

    last_date = df["InvoiceDate"].max()
    ref_ts    = pd.Timestamp(reference_date)
    recency   = max((ref_ts - last_date).days, 0)
    frequency = df["InvoiceNo"].nunique()
    monetary  = df["Total_price"].sum()

    return {
        "Recency":   recency,
        "Frequency": frequency,
        "Monetary":  round(float(monetary), 2),
    }


def predict_segment(rfm: dict, model, scaler) -> tuple:
    X = np.array([[rfm["Recency"], rfm["Frequency"], rfm["Monetary"]]])
    X_scaled = scaler.transform(X)
    raw   = model.predict(X_scaled)[0]
    label = str(raw).strip()

    proba = None
    if hasattr(model, "predict_proba"):
        classes = [str(c).strip() for c in model.classes_]
        p       = model.predict_proba(X_scaled)[0]
        proba   = dict(zip(classes, [float(v) for v in p]))
    return label, proba


def get_recommendations(label: str, seg_top_products: dict,
                         exclude_names: list, n: int = 8) -> pd.DataFrame:
    if label not in seg_top_products:
        return pd.DataFrame(columns=["Description", "Quantity", "UnitPrice"])
    recs = seg_top_products[label].copy()
    recs = recs[~recs["Description"].isin(exclude_names)]
    return recs.head(n)


# ══════════════════════════════════════════════
# تحميل كل شيء
# ══════════════════════════════════════════════
st.title("🛍️ SmartShop AI")
st.caption("كل عميل له تاريخ تراكمي — كل فاتورة جديدة تُحدِّث شريحته وتوصياته")

df_retail         = load_retail_data()
rfm_table          = build_rfm_table(df_retail)
model, scaler      = load_models()
rfm_with_segments  = assign_segments_to_all_customers(rfm_table, model, scaler)
seg_top_products   = build_segment_top_products(df_retail, rfm_with_segments)
all_products       = sorted(df_retail["Description"].unique().tolist())

st.success(
    f"✅ تم تحميل {len(df_retail):,} عملية شراء لـ "
    f"{df_retail['CustomerID'].nunique():,} عميل"
)

# ──────────────────────────────────────────────
# "اليوم" المحاكى — للتحكم بحساب Recency يدوياً
# ──────────────────────────────────────────────
if "simulated_today" not in st.session_state:
    st.session_state.simulated_today = datetime.date.today()

st.markdown("### 🕒 تاريخ اليوم ")
st.caption(
    "هذا التاريخ يُستخدم لحساب Recency بدل تاريخ جهازك الفعلي — "
    "غيّره لمحاكاة مرور الوقت ومشاهدة كيف تتغيّر شريحة العميل"
)
st.session_state.simulated_today = st.date_input(
    "اليوم الحالي ",
    value=st.session_state.simulated_today,
)

st.markdown("---")

with st.expander("🧪 اختبار حساسية النموذج لـ Recency فقط (تثبيت Frequency و Monetary)"):
    st.caption(
        "هون نثبّت Frequency و Monetary على نفس القيم، ونغيّر Recency فقط، "
        "لنرى هل النموذج فعلاً حساس لتغيّر الأيام لحاله أو لأ."
    )
    fixed_freq = st.number_input("Frequency ثابتة للاختبار", min_value=1, value=2, key="test_freq")
    fixed_mon  = st.number_input("Monetary ثابتة للاختبار ($)", min_value=1.0, value=300.0, key="test_mon")

    recency_values = [0, 7, 30, 60, 100, 150, 200, 300]
    rcols = st.columns(len(recency_values))
    for col, rec_val in zip(rcols, recency_values):
        with col:
            test_rfm = {"Recency": rec_val, "Frequency": fixed_freq, "Monetary": fixed_mon}
            test_label, _ = predict_segment(test_rfm, model, scaler)
            st.markdown(f"**R={rec_val}**")
            st.markdown(f"→ {test_label}")

st.markdown("---")

# ══════════════════════════════════════════════
# Session State
# st.session_state.customers = {
#     "اسم العميل": { "invoices": [ {invoice_no, date, lines:[...]} , ... ] }
# }
# (RFM/Segment تُحسب ديناميكياً عند كل عرض، لا تُخزَّن — حتى تتحدث فوراً
#  مع أي تغيير في "اليوم" المحاكى دون الحاجة لزر إعادة حساب)
# ══════════════════════════════════════════════
if "customers" not in st.session_state:
    st.session_state.customers = {}
if "current_lines" not in st.session_state:
    st.session_state.current_lines = []

# ──────────────────────────────────────────────
# 1) اختيار/إنشاء عميل
# ──────────────────────────────────────────────
st.header("👤 من هو العميل؟")

c_existing, c_new = st.columns(2)
with c_existing:
    existing_names = list(st.session_state.customers.keys())
    chosen_existing = st.selectbox(
        "عميل موجود مسبقاً (له فواتير سابقة)",
        options=["-- اختر --"] + existing_names,
    )
with c_new:
    new_name = st.text_input("أو أدخل اسم عميل جديد")

customer_name = None
if new_name.strip():
    customer_name = new_name.strip()
elif chosen_existing != "-- اختر --":
    customer_name = chosen_existing

if customer_name:
    st.info(f"🧑 العميل الحالي: **{customer_name}**")
    if customer_name in st.session_state.customers:
        n_prev = len(st.session_state.customers[customer_name]["invoices"])
        st.caption(f"📜 لديه {n_prev} فاتورة سابقة مسجّلة")
else:
    st.warning("👆 اختر عميلاً موجوداً أو أدخل اسم عميل جديد للمتابعة")

st.markdown("---")

# ──────────────────────────────────────────────
# 2) بناء فاتورة جديدة لهذا العميل
# ──────────────────────────────────────────────
if customer_name:
    st.header(f"🧾 فاتورة جديدة لـ {customer_name}")

    # المنتج خارج الـ form عشان السعر يتحدث فوراً عند تغيير الاختيار
    product = st.selectbox("المنتج", all_products, key="product_select")
    default_price = float(
        df_retail.loc[df_retail["Description"] == product, "UnitPrice"].median()
    )
    st.caption(f"💲 السعر الافتراضي لهذا المنتج: ${default_price:.2f}")

    with st.form("add_line_form", clear_on_submit=True):
        c2, c3 = st.columns(2)
        with c2:
            quantity = st.number_input("الكمية", min_value=1, value=1)
        with c3:
            unit_price = st.number_input(
                "سعر الوحدة", min_value=0.01,
                value=round(default_price, 2), step=0.1
            )
        add_line = st.form_submit_button("➕ أضف هذا المنتج للفاتورة",
                                          use_container_width=True)
        if add_line:
            st.session_state.current_lines.append({
                "Description": product,
                "Quantity":    quantity,
                "UnitPrice":   unit_price,
            })
            st.rerun()

    if st.session_state.current_lines:
        st.markdown("**أسطر الفاتورة الحالية:**")
        cur_df = pd.DataFrame(st.session_state.current_lines)
        cur_df["الإجمالي"] = cur_df["Quantity"] * cur_df["UnitPrice"]
        st.dataframe(
            cur_df.rename(columns={
                "Description": "المنتج", "Quantity": "الكمية", "UnitPrice": "السعر"
            }),
            use_container_width=True, hide_index=True
        )

        invoice_date = st.date_input("تاريخ هذه الفاتورة",
                                     value=st.session_state.simulated_today)

        cc1, cc2 = st.columns(2)
        with cc1:
            finalize = st.button(
                "✅ إنهاء الفاتورة وتحديث شريحة العميل",
                type="primary", use_container_width=True
            )
        with cc2:
            if st.button("🗑️ إلغاء هذه الفاتورة", use_container_width=True):
                st.session_state.current_lines = []
                st.rerun()

        if finalize:
            new_invoice = {
                "invoice_no": f"{customer_name}-INV{len(st.session_state.customers.get(customer_name, {'invoices': []})['invoices']) + 1:03d}",
                "date":       str(invoice_date),
                "lines":      st.session_state.current_lines.copy(),
            }

            if customer_name not in st.session_state.customers:
                st.session_state.customers[customer_name] = {"invoices": []}

            st.session_state.customers[customer_name]["invoices"].append(new_invoice)

            st.session_state.current_lines = []
            st.rerun()
    else:
        st.info("👆 أضف منتجاً واحداً على الأقل لبدء الفاتورة")

    st.markdown("---")

    # ──────────────────────────────────────────
    # 3) عرض الحالة التراكمية الحالية لهذا العميل
    # ──────────────────────────────────────────
    cust_data = st.session_state.customers.get(customer_name)
    if cust_data and cust_data["invoices"]:
        # يُعاد الحساب تلقائياً في كل مرة (بما فيها عند تغيير "اليوم" المحاكى)
        rfm = compute_cumulative_rfm(cust_data["invoices"], st.session_state.simulated_today)
        label, proba = predict_segment(rfm, model, scaler)

        st.header(f"📊 الحالة التراكمية الحالية لـ {customer_name}")
        st.caption(
            f"🕒 محسوبة بالنسبة لتاريخ اليوم المحاكى: "
            f"{st.session_state.simulated_today.strftime('%Y-%m-%d')}"
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("📅 Recency",   f"{rfm['Recency']} يوم",
                  help="أيام منذ آخر فاتورة لهذا العميل")
        r2.metric("🔁 Frequency", f"{rfm['Frequency']} فاتورة",
                  help="عدد الفواتير المختلفة لهذا العميل حتى الآن")
        r3.metric("💰 Monetary",  f"${rfm['Monetary']:,.2f}",
                  help="إجمالي إنفاق هذا العميل عبر كل فواتيره")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    color:white;border-radius:14px;padding:1.2rem 1.5rem;
                    text-align:center;margin:1rem 0;">
          <div style="font-size:.8rem;opacity:.85;">شريحة العميل الحالية (Segment)</div>
          <div style="font-size:1.8rem;font-weight:800;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

        if proba:
            with st.expander("نسب الثقة لكل شريحة"):
                prob_df = pd.DataFrame(
                    list(proba.items()), columns=["Segment", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Segment"))

        # كل المنتجات الي اشتراها هاد العميل سابقاً (نستثنيها من التوصيات)
        bought_before = []
        for inv in cust_data["invoices"]:
            bought_before.extend([l["Description"] for l in inv["lines"]])

        st.markdown("---")
        st.markdown(f"## 🎯 منتجات مقترحة لـ {customer_name} (شريحة {label})")

        recs = get_recommendations(label, seg_top_products,
                                   exclude_names=bought_before, n=8)
        if recs.empty:
            st.info("لا توجد توصيات إضافية متاحة لهذه الشريحة حالياً.")
        else:
            rcols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                with rcols[i % 4]:
                    st.markdown(f"""
                    <div style="background:#f0f9ff;border:1px solid #bae6fd;
                                border-radius:10px;padding:.8rem;text-align:center;
                                min-height:110px;">
                      <div style="font-size:.82rem;font-weight:600;color:#0369a1;">
                        {row['Description'][:45]}
                      </div>
                      <div style="font-size:.95rem;font-weight:700;color:#6366f1;margin-top:.4rem">
                        ${row['UnitPrice']:.2f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # سجل فواتير هذا العميل
        st.markdown("---")
        with st.expander(f"📜 سجل فواتير {customer_name} ({len(cust_data['invoices'])} فاتورة)"):
            for inv in cust_data["invoices"]:
                st.markdown(f"**{inv['invoice_no']}** — {inv['date']}")
                lines_df = pd.DataFrame(inv["lines"])
                lines_df["الإجمالي"] = lines_df["Quantity"] * lines_df["UnitPrice"]
                st.dataframe(
                    lines_df.rename(columns={
                        "Description": "المنتج", "Quantity": "الكمية", "UnitPrice": "السعر"
                    }),
                    use_container_width=True, hide_index=True
                )

st.markdown("---")
st.caption("SmartShop AI · Eid Ahmed © 2026")
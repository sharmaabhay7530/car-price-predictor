import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="AutoValuate Pro - Vehicle Intelligence Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "step" not in st.session_state:
    st.session_state.step = 1

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

bg_images = {
    1: "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1920&q=80",
    "login": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1920&q=80",
    "register": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=1920&q=80",
    "intent": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?auto=format&fit=crop&w=1920&q=80",
    2: "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=1920&q=80",
    3: "https://images.unsplash.com/photo-1486006920555-c77dce18193b?auto=format&fit=crop&w=1920&q=80",
    4: "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=1920&q=80",
    "payment_gateway": "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?auto=format&fit=crop&w=1920&q=80",
    "success": "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=1920&q=80"
}

current_bg = bg_images.get(st.session_state.step, bg_images[1])

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(10, 12, 16, 0.84), rgba(10, 12, 16, 0.90)), url('{current_bg}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }}
    .glass-card {{
        background: rgba(18, 22, 31, 0.85);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 10px 35px 0 rgba(0, 0, 0, 0.45);
        margin-bottom: 20px;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #2563eb, #059669) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        height: 50px !important;
        font-size: 16px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
    }}
    .stButton>button:hover {{
        background: linear-gradient(90deg, #1d4ed8, #047857) !important;
    }}
    div[data-testid="stMetric"] {{
        background: rgba(18, 22, 31, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_real_model():
    data_path = os.path.join('data', 'dataset.csv') if os.path.exists(os.path.join('data', 'dataset.csv')) else 'dataset.csv'
    df = pd.read_csv(data_path)
    
    df['Brand'] = df['Name'].apply(lambda x: str(x).split()[0])
    df['Car_Age'] = 2026 - df['Year']
    
    df['Mileage'] = df['Mileage'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df['Engine'] = df['Engine'].astype(str).str.extract(r'(\d+)').astype(float)
    df['Power'] = df['Power'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
    
    for col in ['Mileage', 'Engine', 'Power', 'Seats']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            
    features = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Car_Age', 'Brand', 'Fuel_Type', 'Transmission', 'Owner_Type']
    features_present = [f for f in features if f in df.columns]
    
    X = pd.get_dummies(df[features_present], drop_first=True)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, X.columns, df, r2, mae, rmse

model, scaler, feature_columns, df, r2, mae, rmse = load_and_train_real_model()

# STEP 1: LANDING CHOICE
if st.session_state.step == 1:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.8, 1])
    with c2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h1 style="font-size: 50px; margin: 0;">🏎️</h1>
            <h2 style="color: #ffffff; margin-top: 5px;">AutoValuate Pro</h2>
            <p style="color: #8b949e; font-size: 14px;">Next-Gen Indian Automobile Valuation & Marketplace</p>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 20px 0;">
            <h3 style="color: #58a6ff; font-weight: 500; font-size: 18px;">👋 Welcome! Choose an option to get started</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔑 Existing User Login", use_container_width=True):
                st.session_state.step = "login"
                st.rerun()
        with col_b:
            if st.button("✨ Create New Account", use_container_width=True):
                st.session_state.step = "register"
                st.rerun()

# STEP 1A: LOGIN FORM
elif st.session_state.step == "login":
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.8, 1])
    with c2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h2 style="color: #ffffff; margin-top: 5px;">🔒 Enterprise Login</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin123")
            btn = st.form_submit_button("Authenticate & Proceed", use_container_width=True)
            if btn:
                if username == "admin" and password == "admin123":
                    st.session_state.step = "intent"
                    st.rerun()
                else:
                    st.error("Invalid Credentials! Use admin / admin123")
        
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# STEP 1B: CREATE NEW ACCOUNT
elif st.session_state.step == "register":
    st.markdown("## 👤 Create New Account - Personal Details")
    st.markdown("<p style='color: #8b949e;'>Please enter your contact and address information to register.</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", value=st.session_state.user_data.get("name", ""))
            phone = st.text_input("Mobile Number", value=st.session_state.user_data.get("phone", ""))
            email = st.text_input("Email Address", value=st.session_state.user_data.get("email", ""))
        with col2:
            address = st.text_area("Residential / Office Address", value=st.session_state.user_data.get("address", ""))
            pincode = st.text_input("Area Pincode", value=st.session_state.user_data.get("pincode", ""))
        st.markdown('</div>', unsafe_allow_html=True)
        
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("⬅ Back to Home", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with b2:
            if st.button("Next: Choose Service Intent ➔", use_container_width=True):
                if not name or not phone:
                    st.error("Please enter at least your Name and Mobile Number.")
                else:
                    st.session_state.user_data["name"] = name
                    st.session_state.user_data["phone"] = phone
                    st.session_state.user_data["email"] = email
                    st.session_state.user_data["address"] = address
                    st.session_state.user_data["pincode"] = pincode
                    st.session_state.step = "intent"
                    st.rerun()

# STEP 1C: BUY OR SELL INTENT
elif st.session_state.step == "intent":
    st.markdown("## 🚗 Select Your Service Intent")
    st.markdown("<p style='color: #8b949e;'>Are you looking to buy a verified pre-owned vehicle or sell your car for the best market valuation?</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card" style="text-align: center; padding: 40px;">', unsafe_allow_html=True)
        intent = st.radio("What would you like to do today?", ["Sell My Car (Get Valuation & Pricing)", "Buy a Pre-Owned Vehicle (Browse & Finance)"], horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("⬅ Back", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with b2:
            if st.button("Continue to Vehicle Configuration ➔", use_container_width=True):
                st.session_state.user_data["intent"] = intent
                st.session_state.step = 3
                st.rerun()

# STEP 3: VEHICLE, OWNER & LEGAL DOCUMENTS CONFIGURATION
elif st.session_state.step == 3:
    st.markdown("## 🚘 Step 3: Complete Car Details, Owner Info & Legal Documents")
    st.markdown("<p style='color: #8b949e;'>Provide full vehicle specifications, driving license details, and legal document statuses for verification.</p>", unsafe_allow_html=True)
    
    all_indian_brands = sorted(list(set(df['Brand'].unique()).union({
        "Maruti", "Hyundai", "Tata", "Mahindra", "Toyota", "Honda", "Kia", "Skoda", 
        "Volkswagen", "BMW", "Audi", "Mercedes", "Jaguar", "Land Rover", "MG", "Renault", "Nissan", "Jeep"
    })))
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        
        with v1:
            st.subheader("⚙️ Technical Specifications")
            selected_brand = st.selectbox("Car Brand", all_indian_brands, index=0)
            car_age = st.slider("Car Age (Years)", 1, 25, 4)
            km_driven = st.number_input("Kilometers Driven", 1000, 500000, 45000, step=5000)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            mileage = st.slider("Mileage (kmpl)", 5.0, 35.0, 18.5)
            engine = st.slider("Engine Capacity (CC)", 600, 5000, 1248, step=100)
            power = st.slider("Power Output (bhp)", 30.0, 500.0, 88.5)
            seats = st.selectbox("Seating Capacity", [2, 4, 5, 7, 8], index=2)
            
        with v2:
            st.subheader("📜 Owner, DL & Legal Documents")
            owner_name = st.text_input("Current Registered Owner Name", st.session_state.user_data.get("name", "Abhay Sharma"))
            dl_number = st.text_input("Owner Driving License (DL) Number", placeholder="UP15 20230001234")
            owner_type = st.selectbox("Ownership History Status", ["First", "Second", "Third", "Fourth & Above"])
            rc_status = st.selectbox("Registration Certificate (RC) Status", ["Valid & Original Available", "Duplicate Issued", "Hypothecated (Loan Active)"])
            insurance_status = st.selectbox("Car Insurance Status", ["Comprehensive (Active)", "Third-Party Only", "Expired"])
            puc_status = st.selectbox("Pollution Under Control (PUC) Certificate", ["Valid", "Expired / Pending"])
            service_history = st.selectbox("Service Record History", ["Authorized Service Center Maintained", "Partial History", "Local Workshop Serviced"])
            car_papers = st.selectbox("Additional Car Papers (Tax Receipt / NOC)", ["All Clear & Available", "Pending NOC", "Duplicate Files"])
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("⬅ Back to Service Intent", use_container_width=True):
                st.session_state.step = "intent"
                st.rerun()
        with col_next:
            if st.button("Generate Certified Valuation 🚀", use_container_width=True):
                st.session_state.user_data["brand"] = selected_brand
                st.session_state.user_data["age"] = car_age
                st.session_state.user_data["km"] = km_driven
                st.session_state.user_data["fuel"] = fuel_type
                st.session_state.user_data["trans"] = transmission
                st.session_state.user_data["mileage"] = mileage
                st.session_state.user_data["engine"] = engine
                st.session_state.user_data["power"] = power
                st.session_state.user_data["seats"] = seats
                st.session_state.user_data["owner_name"] = owner_name
                st.session_state.user_data["dl_number"] = dl_number
                st.session_state.user_data["owner_type"] = owner_type
                st.session_state.user_data["rc"] = rc_status
                st.session_state.user_data["insurance"] = insurance_status
                st.session_state.user_data["puc"] = puc_status
                st.session_state.user_data["service"] = service_history
                st.session_state.user_data["car_papers"] = car_papers
                st.session_state.step = 4
                st.rerun()

# STEP 4: FINAL VALUATION, DETAILED INVOICE & PAYMENT/EMI HUB
elif st.session_state.step == 4:
    d = st.session_state.user_data
    
    top1, top2 = st.columns([3, 1])
    with top1:
        st.title("🏎️ Official Market Valuation & Checkout Portal")
        st.markdown(f"<p style='color: #8b949e;'>Certified Report for <b>{d.get('name')}</b> | Intent: <b>{d.get('intent', 'Sell')}</b> | Vehicle: <b>{d.get('brand')}</b> ({d.get('age')} Yrs)</p>", unsafe_allow_html=True)
    with top2:
        if st.button("🔄 Start New Valuation", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
            
    input_encoded = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    input_encoded.loc[0, 'Kilometers_Driven'] = d.get('km', 45000)
    input_encoded.loc[0, 'Mileage'] = d.get('mileage', 18.5)
    input_encoded.loc[0, 'Engine'] = d.get('engine', 1248)
    input_encoded.loc[0, 'Power'] = d.get('power', 88.5)
    input_encoded.loc[0, 'Seats'] = d.get('seats', 5)
    input_encoded.loc[0, 'Car_Age'] = d.get('age', 4)
    
    for col, val in [('Brand', d.get('brand')), ('Fuel_Type', d.get('fuel')), ('Transmission', d.get('trans')), ('Owner_Type', d.get('owner_type'))]:
        dummy_col = f"{col}_{val}"
        if dummy_col in input_encoded.columns:
            input_encoded.loc[0, dummy_col] = 1
            
    input_scaled = scaler.transform(input_encoded)
    base_price_lakhs = model.predict(input_scaled)[0]
    base_price_inr = base_price_lakhs * 100000
    
    company_discount = base_price_inr * 0.05
    rto_charges = base_price_inr * 0.08
    insurance_fee = base_price_inr * 0.03
    final_onroad_inr = base_price_inr - company_discount + rto_charges + insurance_fee
    final_onroad_lakhs = final_onroad_inr / 100000

    brand_images = {
        "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=1200&q=80",
        "Audi": "https://images.unsplash.com/photo-1603584173870-7f23fdae1b7a?auto=format&fit=crop&w=1200&q=80",
        "Mercedes": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?auto=format&fit=crop&w=1200&q=80",
        "Hyundai": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=1200&q=80",
        "Maruti": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=1200&q=80",
        "Tata": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?auto=format&fit=crop&w=1200&q=80",
        "Mahindra": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=1200&q=80",
        "Toyota": "https://images.unsplash.com/photo-1629897048766-3d234989ef85?auto=format&fit=crop&w=1200&q=80",
        "Honda": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1200&q=80"
    }
    car_img = brand_images.get(d.get("brand"), "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1200&q=80")

    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        st.markdown(f"#### 🚘 Vehicle Showcase: **{d.get('brand')}**")
        st.image(car_img, caption=f"Representative model for {d.get('brand')}", use_container_width=True)
        
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #58a6ff; margin-top:0;">📋 Client, DL & Document Verification</h4>
            <p style="margin: 4px 0;"><b>Client:</b> {d.get('name')} ({d.get('phone')})</p>
            <p style="margin: 4px 0;"><b>Intent:</b> {d.get('intent')}</p>
            <p style="margin: 4px 0;"><b>Address:</b> {d.get('address')} - {d.get('pincode')}</p>
            <p style="margin: 4px 0;"><b>Owner Name:</b> {d.get('owner_name')} ({d.get('owner_type')})</p>
            <p style="margin: 4px 0;"><b>Driving License (DL):</b> {d.get('dl_number', 'N/A')}</p>
            <p style="margin: 4px 0;"><b>Vehicle Specs:</b> {d.get('brand')} | {d.get('age')} Yrs | {d.get('km'):,} KM</p>
            <p style="margin: 4px 0;"><b>Powertrain:</b> {d.get('engine')} CC | {d.get('power')} bhp | {d.get('fuel')}</p>
            <p style="margin: 4px 0;"><b>Documents:</b> RC: {d.get('rc')} | Ins: {d.get('insurance')} | PUC: {d.get('puc')}</p>
            <p style="margin: 4px 0;"><b>Car Papers:</b> {d.get('car_papers')} | Service: {d.get('service')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(31, 111, 235, 0.25), rgba(35, 134, 54, 0.25)); border: 1px solid rgba(56, 139, 253, 0.4); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 0 25px rgba(31,111,235,0.3);">
            <h3 style="color: #7ee787; margin:0; font-size: 15px;">FINAL ESTIMATED ON-ROAD VALUE</h3>
            <h1 style="font-size: 48px; color: #ffffff; margin: 8px 0;">₹ {final_onroad_lakhs:.2f} Lakhs <span style="font-size: 18px; color: #8b949e;">(₹ {final_onroad_inr:,.0f})</span></h1>
            <p style="color: #8b949e; margin:0; font-size: 13px;">Base Market Valuation: ₹ {base_price_lakhs:.2f} Lakhs | Certified by Random Forest Engine</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab_price, tab_pay, tab_chart = st.tabs(["🧾 Detailed Price Invoice", "💳 Payment Options & EMI Calculator", "📊 Market Depreciation Chart"])
        
        with tab_price:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 💵 Itemized Price Invoice Statement")
            
            p_col1, p_col2 = st.columns([3, 1])
            p_col1.write("Base Estimated Market Valuation:")
            p_col2.write(f"**₹ {base_price_inr:,.0f}**")
            
            p_col1.write("Special Company / Dealer Festival Discount (-5%):")
            p_col2.write(f"**- ₹ {company_discount:,.0f}**")
            
            p_col1.write("Estimated RTO Registration & State Tax (+8%):")
            p_col2.write(f"**+ ₹ {rto_charges:,.0f}**")
            
            p_col1.write("Comprehensive Insurance & Legal Documentation (+3%):")
            p_col2.write(f"**+ ₹ {insurance_fee:,.0f}**")
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)
            p_col1.markdown("#### **Final Net Payable On-Road Price:**")
            p_col2.markdown(f"#### **₹ {final_onroad_inr:,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_pay:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            pay_mode = st.radio("Choose Preferred Payment Mode:", ["Full Outright Payment", "Finance / Monthly EMI Plan"], horizontal=True)
            
            if pay_mode == "Full Outright Payment":
                st.write("---")
                method = st.selectbox("Select Payment Channel:", ["UPI (GPay / PhonePe / Paytm)", "QR Code Scanner", "Credit Card (Instant Bank Offer)", "Debit Card", "Net Banking / NEFT"])
                bank_offer = st.selectbox("Select Partner Bank for Instant Cashback:", ["HDFC Bank (₹ 25,000 Instant Off)", "SBI Card (₹ 15,000 Instant Off)", "ICICI Bank (₹ 20,000 Instant Off)", "Axis Bank (₹ 18,000 Instant Off)", "None"])
                
                discount_val = 0
                if "HDFC" in bank_offer: discount_val = 25000
                elif "SBI" in bank_offer: discount_val = 15000
                elif "ICICI" in bank_offer: discount_val = 20000
                elif "Axis" in bank_offer: discount_val = 18000
                
                final_pay_amount = final_onroad_inr - discount_val
                st.success(f"✨ Bank Discount Applied: - ₹ {discount_val:,.0f}")
                st.markdown(f"### 💳 Final Payable Amount via {method}: **₹ {final_pay_amount:,.0f}**")
                
                if st.button("🔒 Proceed to Secure Payment Details Page ➔", use_container_width=True):
                    st.session_state.user_data["selected_pay_method"] = method
                    st.session_state.user_data["final_pay_amount"] = final_pay_amount
                    st.session_state.step = "payment_gateway"
                    st.rerun()
                    
            else:
                st.write("---")
                st.markdown("##### ⚙️ Customize Auto Loan & EMI Parameters")
                
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    down_payment_pct = st.slider("Down Payment Percentage (%)", 10, 80, 20, step=5)
                    tenure_months = st.selectbox("Loan Tenure (Months)", [12, 24, 36, 48, 60, 72, 84], index=4)
                with e_col2:
                    interest_rate = st.slider("Bank Interest Rate (% p.a.)", 7.5, 15.0, 9.5, step=0.25)
                    selected_bank = st.selectbox("Partner Bank", ["HDFC Auto Loan", "SBI Car Loan", "ICICI Finance", "Axis Bank"])
                    
                down_payment = final_onroad_inr * (down_payment_pct / 100)
                loan_principal = final_onroad_inr - down_payment
                
                r = (interest_rate / 12) / 100
                n = tenure_months
                monthly_emi = (loan_principal * r * ((1 + r)**n)) / (((1 + r)**n) - 1)
                total_payable = (monthly_emi * n) + down_payment
                total_interest = total_payable - final_onroad_inr
                
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Monthly EMI", f"₹ {monthly_emi:,.0f} / mo")
                m2.metric("Down Payment Required", f"₹ {down_payment:,.0f}")
                m3.metric("Total Interest Extra", f"₹ {total_interest:,.0f}")
                
                st.info(f"💡 **Loan Summary**: Total Principal: ₹ {loan_principal:,.0f} | Bank: {selected_bank} @ {interest_rate}% p.a. for {tenure_months} Months.")
                
                if st.button("🚀 Apply for Instant Pre-Approved Loan", use_container_width=True):
                    st.session_state.user_data["final_pay_amount"] = down_payment
                    st.session_state.step = "success"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_chart:
            st.markdown("##### 📊 Market Depreciation Trend")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor('#12161f')
            ax.set_facecolor('#1a202c')
            sample_data = df[df['Brand'] == d.get('brand')] if d.get('brand') in df['Brand'].values else df
            sns.scatterplot(data=sample_data.head(100), x='Car_Age', y='Price', color='#388bfd', ax=ax, alpha=0.85, s=50)
            sns.regplot(data=sample_data.head(100), x='Car_Age', y='Price', ax=ax, scatter=False, color='#238636')
            ax.set_title(f"Market Price vs Age Trend: {d.get('brand')}", color='#c9d1d9', fontsize=10)
            ax.set_xlabel("Car Age (Years)", color='#8b949e', fontsize=8)
            ax.set_ylabel("Price (Lakhs)", color='#8b949e', fontsize=8)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
            st.pyplot(fig)

# STEP 5: DEDICATED SECURE PAYMENT GATEWAY FORM PAGE
elif st.session_state.step == "payment_gateway":
    d = st.session_state.user_data
    method = d.get("selected_pay_method", "UPI")
    amount = d.get("final_pay_amount", 0)
    
    st.markdown("## 🔐 Secure Payment Gateway & Account Details")
    st.markdown(f"<p style='color: #8b949e;'>Complete your transaction of <b>₹ {amount:,.0f}</b> via <b>{method}</b>.</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if "QR Code" in method:
            st.markdown("### 📱 Scan & Pay via Dynamic QR Code")
            col_qr1, col_qr2 = st.columns(2)
            with col_qr1:
                st.image("https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=upi://pay?pa=autovaluate@sbi&pn=AutoValuatePro&am=" + str(amount), caption="Scan with any UPI App (GPay, PhonePe, Paytm)")
            with col_qr2:
                st.info(f"💡 **Instructions**:\n1. Open your UPI App.\n2. Scan the secure QR code on the left.\n3. Verify merchant name: **AutoValuate Pro Gateway**.\n4. Pay **₹ {amount:,.0f}** and click confirmation below.")
        
        elif "UPI" in method:
            st.markdown("### ⚡ Enter UPI Handle / VPA")
            st.text_input("UPI ID / Virtual Payment Address (e.g., abhay@oksbi)", placeholder="username@oksbi")
            
        elif "Credit Card" in method or "Debit Card" in method:
            st.markdown("### 💳 Enter Card Details")
            st.text_input("Cardholder Name", value=d.get("name", ""))
            st.text_input("Card Number", placeholder="XXXX XXXX XXXX XXXX")
            c_col1, c_col2 = st.columns(2)
            c_col1.text_input("Expiry Date", placeholder="MM/YY")
            c_col2.text_input("CVV / Secure Code", type="password", placeholder="123")
            
        else: # Net Banking / NEFT
            st.markdown("### 🏦 Enter Bank Account & IFSC Details")
            st.text_input("Account Holder Name", value=d.get("name", ""))
            st.text_input("Bank Account Number", placeholder="123456789012")
            st.text_input("Bank IFSC Code", placeholder="SBIN0001234")
            st.text_input("Bank Name", placeholder="State Bank of India")
            
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
        
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("⬅ Back to Valuation Portal", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
        with b2:
            if st.button("✅ Confirm & Pay Now 🚀", use_container_width=True):
                st.session_state.step = "success"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# STEP 6: SUCCESS & SMS NOTIFICATION PAGE
elif st.session_state.step == "success":
    d = st.session_state.user_data
    phone_no = d.get("phone", "+91 98765 43210")
    amount = d.get("final_pay_amount", 0)
    
    st.balloons()
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border: 1px solid rgba(40, 167, 69, 0.5);">
            <h1 style="font-size: 60px; margin: 0;">🎉</h1>
            <h2 style="color: #7ee787; margin-top: 10px;">Payment Successful!</h2>
            <p style="color: #e6edf3; font-size: 16px;">Transaction Amount: <b>₹ {amount:,.0f}</b></p>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 20px 0;">
            <div style="background: rgba(35, 134, 54, 0.2); border-left: 4px solid #238636; padding: 15px; border-radius: 8px; text-align: left; margin-bottom: 20px;">
                <p style="color: #7ee787; margin: 0; font-weight: bold;">💬 Simulated SMS Sent to: {phone_no}</p>
                <p style="color: #c9d1d9; margin: 5px 0 0 0; font-size: 13px;"><i>"Congratulations {d.get('name')}! Your booking/payment for {d.get('brand')} is successfully received. Ref ID: AV-2026-9871. Thank you for choosing AutoValuate Pro."</i></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Start New Transaction / Valuation", use_container_width=True):
            st.session_state.step = 1
            st.session_state.user_data = {}
            st.rerun()
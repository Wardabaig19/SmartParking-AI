import streamlit as st
import cv2
import time
import datetime
import numpy as np
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import base64
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'YOLOv11s.pt'         
DB_PATH = 'smartPark.db'

def load_rois(video_filename):
    """Dynamically loads ROIs from a JSON file matching the video name."""
    json_filename = video_filename.replace(".mp4", ".json")
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)
            return {k: np.array(v, np.int32) for k, v in data.items()}
    return {}

def get_base64_of_bin_file(bin_file):
    """Encodes a local image to base64 so it can be injected safely into HTML."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

st.set_page_config(page_title="SmartParking AI System", layout="wide", page_icon="🚘", initial_sidebar_state="collapsed")

# --- UI CSS: MASTER POLISH ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@700;800&display=swap');

/* Hide Defaults */
[data-testid="stToolbar"], header[data-testid="stHeader"], footer {display:none!important}

/* Thematic Colors */
:root {
    --c-navy: #191B3A;      
    --c-panel: #272C4A;     
    --c-slate: #4A8494;     
    --c-mint: #81C29A;      
    --c-yellow: #E5F0B6;    
    --c-coral: #E06B74;     
    --c-text: #F8FAFC;
    --c-muted: #A1A1AA;
}

/* Base App Constraints */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; max-width: 1400px !important; }
html, body, [data-testid="stApp"] { background-color: var(--c-navy) !important; font-family: 'Inter', sans-serif !important; color: var(--c-text) !important; }

/* =========================================
   SLEEK, REDESIGNED HEADER (3-COLUMN GRID)
   ========================================= */
.sp-header {
    display: grid;
    grid-template-columns: 1fr auto 1fr; /* Left, Center, Right */
    align-items: center;
    background: linear-gradient(145deg, var(--c-panel) 0%, #1e223d 100%);
    border: 1px solid var(--c-slate); 
    border-radius: 16px;
    padding: 16px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.header-left { justify-self: start; } 
.header-center { display: flex; align-items: center; justify-self: center; gap: 24px; }
.header-right { justify-self: end; }

/* LARGER CIRCULAR LOGO WITH GLOW */
.img-logo { 
    width: 110px;  
    height: 110px; 
    object-fit: cover; 
    border-radius: 50%;
    border: 3px solid var(--c-mint);
    box-shadow: 0 0 24px rgba(129,194,154,0.35), 0 4px 15px rgba(0,0,0,0.5);
    background-color: var(--c-navy);
    flex-shrink: 0;
}

.sp-title { 
    font-family: 'Poppins', sans-serif; 
    font-size: 38px; 
    font-weight: 800; 
    color: var(--c-text); 
    margin: 0 0 6px 0;
    letter-spacing: -1px;
    line-height: 1.05;
    text-align: left;
}
.sp-sub { 
    font-size: 11px; 
    color: var(--c-mint); 
    font-weight: 700; 
    letter-spacing: 3px; 
    margin: 0;
    text-transform: uppercase;
    text-align: left;
}

.live-pill {
    display: flex; align-items: center; gap: 10px;
    background: rgba(129, 194, 154, 0.15); border: 1px solid var(--c-mint);
    color: var(--c-mint); font-size: 12px; font-weight: 700;
    padding: 10px 20px; border-radius: 30px; box-shadow: 0 0 20px rgba(129, 194, 154, 0.15);
}
.live-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--c-mint); animation: pulse 2s infinite; }
@keyframes pulse { 0% {opacity: 1;} 50% {opacity: 0.3;} 100% {opacity: 1;} }

.header-time-pill {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    background: rgba(74, 132, 148, 0.15); border: 1px solid var(--c-slate);
    color: var(--c-text); font-size: 14px; font-weight: 600; letter-spacing: 1px;
    padding: 10px 20px; border-radius: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* =========================================
   TABS & KPI CARDS
   ========================================= */
.stTabs [data-baseweb="tab-list"] { 
    gap: 20px; 
    margin-bottom: 30px; 
    justify-content: center; 
}

.stTabs [data-baseweb="tab"] { 
    height: 48px; 
    background-color: rgba(39, 44, 74, 0.5); 
    padding: 10px 30px; 
    color: var(--c-muted); 
    font-family: 'Poppins', sans-serif; 
    font-size: 15px; 
    font-weight: 600; 
    letter-spacing: 0.5px;
    border: 1px solid transparent; 
    border-radius: 12px; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--c-text);
    background-color: rgba(74, 132, 148, 0.15); 
}

.stTabs [aria-selected="true"] { 
    color: var(--c-navy) !important; 
    background: var(--c-mint) !important; 
    border: 1px solid var(--c-mint); 
    font-weight: 700;
    box-shadow: 0 0 20px rgba(129, 194, 154, 0.4); 
}

.kpi-container { display: flex; gap: 16px; margin-bottom: 20px; }
.kpi-card { 
    flex: 1; background: var(--c-panel); border: 1px solid var(--c-slate); 
    border-radius: 12px; padding: 20px; text-align: center; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: transform 0.3s ease;
}
.kpi-card:hover { transform: translateY(-5px); box-shadow: 0 10px 24px rgba(0,0,0,0.3); }
.kpi-label { font-size: 13px; font-weight: 600; color: var(--c-yellow); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;}
.kpi-val { font-size: 40px; font-weight: 800; margin: 0; }
.val-blue { color: var(--c-text); } .val-green { color: var(--c-mint); text-shadow: 0 0 15px rgba(129, 194, 154, 0.3);} .val-red { color: var(--c-coral); text-shadow: 0 0 15px rgba(224, 107, 116, 0.3);}

/* =========================================
   VISUALS & GRIDS
   ========================================= */
.grid-container {
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; height: 100%;
}
.grid-slot {
    background: var(--c-panel); 
    aspect-ratio: 1.2; border-radius: 8px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; font-size: 13px; font-weight: 700;
    transition: transform 0.2s ease, filter 0.2s ease; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.grid-slot:hover { transform: scale(1.1); cursor: crosshair; filter: brightness(1.2); box-shadow: 0 4px 15px rgba(0,0,0,0.4);}
.slot-free { border: 1px solid var(--c-mint); color: var(--c-mint); }
.slot-occ { border: 1px solid var(--c-coral); color: var(--c-coral); }

[data-testid="stImage"] { margin-bottom: 0px; }
[data-testid="stImage"] img { border-radius: 12px !important; border: none !important; box-shadow: none !important; width: 100%; }
[data-testid="stPlotlyChart"] { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; }

/* Control Bar Buttons AND Download Button */
div.stButton > button, div.stDownloadButton > button {
    background-color: var(--c-slate) !important; color: var(--c-text) !important;
    border-radius: 8px !important; border: none !important;
    height: 44px !important; font-size: 15px !important; font-weight: 600 !important;
    white-space: nowrap !important; /* Force text to stay on one line */
    transition: all 0.2s ease !important; box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: var(--c-mint) !important; color: var(--c-navy) !important; transform: translateY(-2px); box-shadow: 0 6px 15px rgba(129, 194, 154, 0.4) !important;}

/* =========================================
   ABOUT US 3D CARDS & TEXT
   ========================================= */
.about-container { background: var(--c-panel); border: 1px solid var(--c-slate); border-radius: 16px; padding: 30px 40px; margin-top: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.25); }
.about-header-wrapper { max-width: 800px; margin: 0 auto 15px auto; text-align: center; }
.about-title { font-family: 'Poppins', sans-serif !important; font-size: 34px; font-weight: 800; margin-bottom: 12px; background: linear-gradient(90deg, var(--c-text) 0%, var(--c-mint) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; line-height: 1.2;}
.about-desc { font-family: 'Inter', sans-serif !important; line-height: 1.6; font-size: 15px; color: var(--c-muted); margin: 0; }
.about-divider { width: 60px; height: 3px; background-color: var(--c-slate); border-radius: 2px; margin: 24px auto; opacity: 0.5; }
.about-team-title { font-family: 'Poppins', sans-serif !important; color: var(--c-text); font-weight: 700; font-size: 24px; text-align: center; margin-bottom: 4px; }
.about-team-sub { font-family: 'Inter', sans-serif !important; color: var(--c-mint); font-size: 13px; text-align: center; margin-bottom: 24px; font-weight: 500;}
.team-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-top: 20px; }
.flip-card { background-color: transparent; perspective: 1000px; height: 230px; cursor: pointer; }
.flip-card-inner { position: relative; width: 100%; height: 100%; text-align: center; transition: transform 0.6s cubic-bezier(0.4, 0.2, 0.2, 1); transform-style: preserve-3d; }
.flip-card:hover .flip-card-inner { transform: rotateY(180deg); }
.flip-card-front, .flip-card-back { position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border-radius: 16px; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); }
.flip-card-front { background-color: var(--c-panel); border: 1px solid var(--c-slate); }
.flip-card-front h3 { color: var(--c-text); margin: 0; font-size: 24px; font-weight: 800; }
.flip-card-front p { color: var(--c-yellow); margin-top: 12px; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; text-align: center; line-height: 1.4;}
.flip-card-back { background-color: var(--c-slate); color: var(--c-navy); transform: rotateY(180deg); border: 2px solid var(--c-mint); }
.flip-card-back h4 { margin: 0 0 12px 0; font-size: 15px; font-weight: 800; color: var(--c-navy); text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--c-navy); padding-bottom: 6px;}
.flip-card-back p { margin: 0; font-size: 13px; font-weight: 500; line-height: 1.6; color: #111; text-align: center;}
h1, h2, h3, h4, h5, p { font-family: 'Inter', sans-serif !important;}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'system_initialized' not in st.session_state: st.session_state.system_initialized = False
if 'running' not in st.session_state: st.session_state.running = False
if 'cap' not in st.session_state: st.session_state.cap = None
if 'frame_skip' not in st.session_state: st.session_state.frame_skip = 0
if 'video_choice' not in st.session_state: st.session_state.video_choice = "parking_crop.mp4"

# --- DYNAMIC ROIs ---
PARKING_ROIS = load_rois(st.session_state.video_choice)

# --- DATABASE PIPELINE ---
def init_db(rois):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS parking_slots')
    # Use localtime so initial seed timestamps match the system clock
    cursor.execute('''CREATE TABLE parking_slots (
        bay_id TEXT PRIMARY KEY,
        status TEXT,
        last_updated TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS occupancy_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
        occupied INTEGER,
        free INTEGER
    )''')
    for slot_id in rois.keys():
        cursor.execute("INSERT OR IGNORE INTO parking_slots (bay_id, status) VALUES (?, 'Empty')", (slot_id,))
    conn.commit()
    conn.close()

def update_db(slot_statuses, occupied, free):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    local_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_data = [(status, local_now, slot_id) for slot_id, status in slot_statuses.items()]
    cursor.executemany(
        "UPDATE parking_slots SET status = ?, last_updated = ? WHERE bay_id = ?",
        update_data
    )
    cursor.execute(
        "INSERT INTO occupancy_log (timestamp, occupied, free) VALUES (?, ?, ?)",
        (local_now, occupied, free)
    )
    conn.commit()
    conn.close()

if 'db_initialized' not in st.session_state:
    init_db(PARKING_ROIS)
    st.session_state.db_initialized = True

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def process_frame(frame, model):
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes if len(results[0].boxes) > 0 else []
    slot_statuses = {slot_id: 'Empty' for slot_id in PARKING_ROIS.keys()}
    overlay = frame.copy()
    occupied_count = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id] 
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        for slot_id, polygon in PARKING_ROIS.items():
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                slot_statuses[slot_id] = class_name
                break 

    for slot_id, polygon in PARKING_ROIS.items():
        if 'occupied' in slot_statuses[slot_id].lower():
            cv2.fillPoly(overlay, [polygon], (116, 107, 224)) 
            occupied_count += 1
        else:
            cv2.fillPoly(overlay, [polygon], (154, 194, 129)) 
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    free_count = len(PARKING_ROIS) - occupied_count
    return frame, slot_statuses, occupied_count, free_count

model = load_model()

# --- FRAME PROCESSING ---
annotated_frame = None
current_statuses = {slot: "Empty" for slot in PARKING_ROIS.keys()}
total_slots = len(PARKING_ROIS)
occ_count = 0
free_count = total_slots

# Enforce zero counts if system has not been initialized yet by the user
if not st.session_state.system_initialized:
    total_slots = 0
    free_count = 0
    occ_count = 0
    current_statuses = {}

if st.session_state.running and total_slots > 0:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(st.session_state.video_choice)
    
    ret, frame = st.session_state.cap.read()
    if not ret: 
        st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = st.session_state.cap.read()
        
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame, current_statuses, occ_count, free_count = process_frame(rgb_frame, model)
        
        st.session_state.frame_skip += 1
        if st.session_state.frame_skip % 3 == 0:
            update_db(current_statuses, occ_count, free_count)

# --- GLOBAL TIME FETCH ---
now = datetime.datetime.now()
current_time = now.strftime("%I:%M:%S %p")

# --- HEADER WITH 3-COLUMN GRID ---
logo_base64 = get_base64_of_bin_file("logo.png")
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="img-logo">'
else:
    logo_html = '<svg class="img-logo" style="fill:var(--c-mint); padding:10px;" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>'

st.markdown(f"""
<div class="sp-header">
    <div class="header-left">
        <div class="live-pill"><span class="live-dot"></span>SYSTEM ONLINE</div>
    </div>
    <div class="header-center">
        {logo_html}
        <div class="header-text">
            <div class="sp-title">SmartParking AI System</div>
            <div class="sp-sub">REAL-TIME URBAN OCCUPANCY MATRIX</div>
        </div>
    </div>
    <div class="header-right">
        <div class="header-time-pill">🕒 {current_time}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- CENTERED NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🔴 Live Dashboard", "📊 Database & Analytics", "👥 About Us"])

# ==========================================
# TAB 1: LIVE DASHBOARD
# ==========================================
with tab1:
    # WIDENED THE BUTTON COLUMNS TO PREVENT TEXT WRAPPING
    col_lbl, col_sel, btn_col1, btn_col2, _ = st.columns([1.5, 2.5, 2, 2, 2], gap="small")
    
    with col_lbl:
        st.markdown("<p style='color: var(--c-muted); font-size: 14px; font-weight: 600; line-height: 44px; text-align: right; margin: 0;'>Select Camera Feed:</p>", unsafe_allow_html=True)
    with col_sel:
        selected_video = st.selectbox("Video Selector", ["parking_crop.mp4", "parking_cctv(sm).mp4"], label_visibility="collapsed")
    with btn_col1:
        if st.button("▶ Initialize Feed", width='stretch'):
            st.session_state.system_initialized = True
            st.session_state.video_choice = selected_video
            new_rois = load_rois(selected_video)
            init_db(new_rois)
            st.session_state.running = True
            st.session_state.cap = None
            st.rerun()
    with btn_col2:
        if st.button("⏹ Halt System", width='stretch'):
            st.session_state.running = False
            st.rerun()

    if st.session_state.system_initialized and total_slots == 0:
        st.warning(f"⚠️ Could not find ROI data for `{st.session_state.video_choice}`. Please ensure `{st.session_state.video_choice.replace('.mp4', '.json')}` is in your project folder.")

    if not st.session_state.running and total_slots > 0 and st.session_state.system_initialized:
        conn = sqlite3.connect(DB_PATH)
        df_latest = pd.read_sql_query("SELECT * FROM parking_slots", conn)
        conn.close()
        occ_count = len(df_latest[df_latest['status'] == 'Occupied'])
        free_count = total_slots - occ_count
        for _, row in df_latest.iterrows():
            current_statuses[row['bay_id']] = row['status']

    kpi_ph = st.empty()
    kpi_ph.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card"><div class="kpi-label">System Capacity</div><div class="kpi-val val-blue">{total_slots}</div></div>
        <div class="kpi-card"><div class="kpi-label">Available Bays</div><div class="kpi-val val-green">{free_count}</div></div>
        <div class="kpi-card"><div class="kpi-label">Active Vehicles</div><div class="kpi-val val-red">{occ_count}</div></div>
    </div>
    """, unsafe_allow_html=True)

    feed_col, grid_col = st.columns([1.5, 1], gap="medium")
    with feed_col:
        if annotated_frame is not None:
            st.image(annotated_frame, width='stretch')
        else:
            if st.session_state.system_initialized:
                st.info("Video feed is offline or loading...")
            else:
                st.info("System Standby. Select a camera feed and click 'Initialize Feed' to begin tracking.")
            
    with grid_col:
        if st.session_state.system_initialized:
            html = '<div class="grid-container">'
            for slot_id, status in current_statuses.items():
                is_occ = "occupied" in status.lower()
                css_class = "slot-occ" if is_occ else "slot-free"
                icon = "🚙" if is_occ else "P"
                num = slot_id.split('_')[1]
                html += f'<div class="grid-slot {css_class}" title="Bay {num} is {status}">{num}<br><span style="font-size:18px;">{icon}</span></div>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height: 100%; display: flex; align-items: center; justify-content: center; background: var(--c-panel); border: 1px dashed var(--c-slate); border-radius: 12px; min-height: 300px;">
                <p style="color: var(--c-muted); font-weight: 600; text-align: center;">Awaiting Initialization<br><span style="font-size: 13px; font-weight: 400;">Matrix will generate upon feed selection</span></p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: DATABASE & Analytics
# ==========================================
with tab2:
    st.markdown("<h3 style='margin-bottom: 15px; font-weight: 700; color: #F8FAFC;'>System Telemetry Logs</h3>", unsafe_allow_html=True)
    
    conn = sqlite3.connect(DB_PATH)
    df_live = pd.read_sql_query("SELECT * FROM parking_slots", conn)
    conn.close()

    if not st.session_state.system_initialized:
        df_live = df_live.iloc[0:0] 

    chart_col, data_col = st.columns([1, 2], gap="large")

    with chart_col:
        st.markdown("<h5 style='color: #81C29A; margin-top:0;'>Real-Time Occupancy Ratio</h5>", unsafe_allow_html=True)
        if st.session_state.system_initialized and total_slots > 0:
            fig_pie = px.pie(names=['Available', 'Occupied'], values=[free_count, occ_count], hole=0.6, color_discrete_sequence=['#81C29A', '#E06B74'])
        else:
            fig_pie = px.pie(names=['Standby'], values=[1], hole=0.6, color_discrete_sequence=['#191B3A'])
            fig_pie.update_traces(hovertemplate=None, textinfo='none')
            
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"), showlegend=False)
        st.plotly_chart(fig_pie, width='stretch')

    with data_col:
        raw_html = """
        <div style='height: 320px; overflow-y: auto; border-radius: 12px; border: 1px solid var(--c-slate); background: var(--c-panel); box-shadow: 0 8px 24px rgba(0,0,0,0.15); margin-bottom: 20px;'>
            <table style='width: 100%; border-collapse: collapse; text-align: left; color: var(--c-text);'>
                <thead style='position: sticky; top: 0; background: var(--c-navy); z-index: 1;'>
                    <tr>
                        <th style='padding: 16px; border-bottom: 1px solid var(--c-slate); font-weight: 600; color: var(--c-yellow); font-size: 14px;'>ROI Tag (Bay ID)</th>
                        <th style='padding: 16px; border-bottom: 1px solid var(--c-slate); font-weight: 600; color: var(--c-yellow); font-size: 14px;'>Current Status</th>
                        <th style='padding: 16px; border-bottom: 1px solid var(--c-slate); font-weight: 600; color: var(--c-yellow); font-size: 14px;'>Last Sync Timestamp</th>
                    </tr>
                </thead>
                <tbody>
        """
        if not df_live.empty:
            for _, row in df_live.iterrows():
                status_color = "var(--c-coral)" if row['status'] == "Occupied" else "var(--c-mint)"
                raw_html += f"<tr style='border-bottom: 1px solid rgba(74, 132, 148, 0.3);'><td style='padding: 14px 16px; font-size: 14px;'>{row['bay_id']}</td><td style='padding: 14px 16px; font-size: 14px; font-weight: 600; color: {status_color};'>{row['status']}</td><td style='padding: 14px 16px; font-size: 13px; color: var(--c-muted);'>{row['last_updated']}</td></tr>"
        else:
            raw_html += "<tr><td colspan='3' style='padding: 30px; text-align: center; color: var(--c-muted); font-size: 14px;'>No telemetry data available. System on standby.</td></tr>"
            
        raw_html += "</tbody></table></div>"
        
        clean_table_html = "".join(line.strip() for line in raw_html.split('\n'))
        st.markdown(clean_table_html, unsafe_allow_html=True)
        
        csv = df_live.to_csv(index=False).encode('utf-8')
        # Button moved outside of columns to stretch full width and be completely visible
        st.download_button(
            label="📥 Export Telemetry Log (CSV)", 
            data=csv, 
            file_name=f"SmartParkingAI_data.csv", 
            mime="text/csv", 
            width='stretch', 
            disabled=not st.session_state.system_initialized
        )

# ==========================================
# TAB 3: ABOUT US
# ==========================================
with tab3:
    raw_about = """
    <div class="about-container">
        <div class="about-header-wrapper">
            <h2 class="about-title">Optimizing Urban Parking</h2>
            <p class="about-desc">The <strong>SmartParking AI System</strong> transforms standard CCTV infrastructure into intelligent, real-time spatial data. By deploying a custom-trained object-detection model against fixed Regions of Interest (ROIs), SmartParking maps the physical boundaries of a parking lot to a digital grid, logging state changes asynchronously to a lightweight SQLite database.</p>
        </div>
        
        <div class="about-divider"></div>

        <h3 class="about-team-title">Project Engineering Team</h3>
        <p class="about-team-sub"></p>
        
        <div class="team-grid">
            <div class="flip-card">
                <div class="flip-card-inner">
                    <div class="flip-card-front">
                        <h3>Warda Baig</h3>
                        <p>AI Model & Database Architect</p>
                    </div>
                    <div class="flip-card-back">
                        <h4>Key Contributions</h4>
                        <p>Led the training of the YOLO prototype, established the computer vision logic, and structured the SQLite database schemas alongside technical documentation.</p>
                    </div>
                </div>
            </div>
            
            <div class="flip-card">
                <div class="flip-card-inner">
                    <div class="flip-card-front">
                        <h3>Hiba Zubairi</h3>
                        <p>Backend & Systems Integration</p>
                    </div>
                    <div class="flip-card-back">
                        <h4>Key Contributions</h4>
                        <p>Engineered the core dashboard logic, ensured seamless data pipelines between OpenCV and Streamlit, and contributed to technical documentation.</p>
                    </div>
                </div>
            </div>
            
            <div class="flip-card">
                <div class="flip-card-inner">
                    <div class="flip-card-front">
                        <h3>Areeba Shoaib</h3>
                        <p>Frontend UI/UX Developer</p>
                    </div>
                    <div class="flip-card-back">
                        <h4>Key Contributions</h4>
                        <p>Designed and implemented the dashboard interface, focusing on user experience, real-time visual matrices, and project documentation.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    clean_about_html = "".join(line.strip() for line in raw_about.split('\n'))
    st.markdown(clean_about_html, unsafe_allow_html=True)

# --- TRIGGER THE LOOP ---
if st.session_state.running and total_slots > 0:
    time.sleep(0.05) 
    st.rerun()
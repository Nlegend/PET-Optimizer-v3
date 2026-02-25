import os
import math
import base64
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ==========================================
# 1. BACKEND: HARDWARE & GUIDELINES CONFIG
# ==========================================
class HardwareConfig:
    AXIAL_FOV_MM = 261.0 
    LEGACY_REF_FOV_MM = 162.0
    HYBRID_FOV_MM = 211.5 
    SENSITIVITY_KCPS_MBQ = 9.1
    NECR_PEAK_KCPS = 220.0
    ACTIVITY_AT_PEAK_KBQ_ML = 21.0
    MAX_BED_SPEED_MMS = 200.0
    MIN_BED_SPEED_MMS = 0.1
    SCAN_LENGTH_MM = 1090.0 
    SYSTEM_K_FDG = 0.0225  
    SYSTEM_K_PSMA = 0.0285 
    SNR_ALPHA = 0.41

class ClinicalGuidelines:
    DEFAULT_FDG_SNR = 11.0
    DEFAULT_PSMA_SNR = 14.0
    PED_FDG_SNR = 9.0             
    PED_PSMA_SNR = 10.0           
    LIVER_COV_LIMIT = 0.099       
    MBP_SNR_LIMIT = 10.0          
    PSMA_AT_PRODUCT_TARGET = 8.0  

# ==========================================
# 2. BACKEND: PATIENT & ANATOMY MODELS
# ==========================================
@dataclass
class PatientModel:
    weight_kg: float
    height_cm: float
    gender: str  
    injected_activity_input: float  
    tracer: str
    uptake_time_min: float
    is_pediatric: bool            
    
    @property
    def bmi(self) -> float:
        return self.weight_kg / ((self.height_cm / 100.0) ** 2)
        
    @property
    def activity_mbq(self) -> float:
        return self.injected_activity_input * 37.0
        
    @property
    def bsa(self) -> float:
        return math.sqrt((self.weight_kg * self.height_cm) / 3600)

    @property
    def lbm_kg(self) -> float:
        if self.is_pediatric:
            return max(self.weight_kg * 0.85, 5.0)
        if self.gender == "Female":
            lbm = (0.252 * self.weight_kg) + (0.473 * self.height_cm) - 48.3
        else:
            lbm = (0.407 * self.weight_kg) + (0.267 * self.height_cm) - 19.2
        return max(20.0, min(lbm, self.weight_kg))

    def trunk_attenuation_penalty(self) -> float:
        if self.is_pediatric or self.bmi <= 28.0: return 1.0
        excess_bmi = self.bmi - 28.0
        lambda_val = 0.040 if (self.gender == "Female" and self.height_cm < 160.0) else 0.030
        return max(0.65, math.exp(-lambda_val * excess_bmi))

class AnatomySegmenter:
    @staticmethod
    def get_zones() -> Dict[str, tuple]:
        return {
            "Head & Neck": (0.0, 250.0),
            "Thorax": (250.0, 500.0),
            "Liver / Upper Abdomen": (500.0, 750.0),
            "Pelvis / Bladder": (750.0, 950.0),
            "Lower Extremities": (950.0, 1090.0)
        }

# ==========================================
# 3. BACKEND: PET PHYSICS ENGINE
# ==========================================
class PhysicsCore:
    def __init__(self, patient: PatientModel, config: HardwareConfig):
        self.patient = patient
        self.cfg = config
        self.is_psma = "PSMA" in patient.tracer.upper()
        
        decay_constant = 0.0102 if "68GA" in patient.tracer.upper() else 0.0063
        self.a_scan_mbq = patient.activity_mbq * math.exp(-decay_constant * patient.uptake_time_min)
        
        base_scatter = 0.23 if self.patient.is_pediatric else 0.39
        
        self.necr_a = self.cfg.SENSITIVITY_KCPS_MBQ
        self.necr_c = self.necr_a / (2.0 * self.cfg.NECR_PEAK_KCPS * self.cfg.ACTIVITY_AT_PEAK_KBQ_ML)
        self.necr_b = base_scatter * self.necr_a / self.cfg.NECR_PEAK_KCPS
        self.k_factor = self.cfg.SYSTEM_K_PSMA if self.is_psma else self.cfg.SYSTEM_K_FDG

    def get_necr(self) -> float:
        biodist_factor = 0.75 if self.is_psma else 0.85
        conc_kbq_ml = max((self.a_scan_mbq * biodist_factor * 1000.0) / (self.patient.lbm_kg * 1000.0), 0.1)
        denominator = 1.0 + (self.necr_b * conc_kbq_ml) + (self.necr_c * conc_kbq_ml**2)
        return max((self.necr_a * conc_kbq_ml) / denominator, 0.1)

    def velocity_for_snr(self, target_snr: float) -> float:
        nec_required = (target_snr / self.k_factor) ** (1.0 / self.cfg.SNR_ALPHA)
        return self.cfg.AXIAL_FOV_MM / (nec_required / (self.get_necr() * 1000.0))

    def snr_for_velocity(self, v_mms: float) -> float:
        if v_mms <= 0: return 0.0
        nec = self.get_necr() * 1000.0 * (self.cfg.AXIAL_FOV_MM / v_mms)
        return self.k_factor * (max(nec, 1e-9) ** self.cfg.SNR_ALPHA)

class ProtocolOptimizer:
    def __init__(self, physics: PhysicsCore, guidelines: ClinicalGuidelines, target_snr: float):
        self.phy = physics
        self.cfg = physics.cfg
        self.guide = guidelines
        self.patient = physics.patient
        self.target_snr = target_snr

    def generate_modulated_profile(self) -> Tuple[List[Dict], float, float]:
        v_snr = self.phy.velocity_for_snr(self.target_snr)
        v_cov = self.phy.velocity_for_snr(1.0 / self.guide.LIVER_COV_LIMIT)
        v_mbp = self.phy.velocity_for_snr(self.guide.MBP_SNR_LIMIT)
        trunk_penalty = self.patient.trunk_attenuation_penalty()

        if '18F-PSMA-1007' in self.patient.tracer.upper():
            act_kg = self.phy.a_scan_mbq / self.patient.weight_kg
            v_at = self.cfg.AXIAL_FOV_MM / ((self.guide.PSMA_AT_PRODUCT_TARGET / act_kg) * 60.0) if act_kg > 0 else 0.1
            v_base = (0.7 * v_snr) + (0.3 * v_at)
        else:
            v_base = v_snr

        profile = []
        total_time_min = 0.0
        target_zone_snr = 0.0 

        for zone, (start_z, end_z) in AnatomySegmenter.get_zones().items():
            if not self.phy.is_psma:
                if zone == "Head & Neck": zone_vel = v_base * 1.3 
                elif zone == "Pelvis / Bladder": zone_vel = v_base * 1.25 
                elif zone == "Thorax": zone_vel = min(v_base, v_mbp) * trunk_penalty
                elif zone == "Liver / Upper Abdomen": zone_vel = min(v_base, v_cov) * trunk_penalty
                else: zone_vel = v_base * 1.4 
            else:
                if zone == "Pelvis / Bladder": zone_vel = v_base 
                elif zone == "Head & Neck": zone_vel = v_base * 1.2
                elif zone == "Thorax": zone_vel = min(v_base * 1.3, v_mbp) * trunk_penalty
                elif zone == "Liver / Upper Abdomen": zone_vel = min(v_base * 1.25, v_cov) * trunk_penalty
                else: zone_vel = v_base * 1.4 

            v_final = round(max(self.cfg.MIN_BED_SPEED_MMS, min(zone_vel, self.cfg.MAX_BED_SPEED_MMS)), 1)
            
            profile.append({
                "Zone": zone, "Start_Z_mm": start_z, "End_Z_mm": end_z, 
                "v_snr": v_snr, "v_base": v_base, "Final_Velocity_mms": v_final
            })
            total_time_min += ((end_z - start_z) / v_final) / 60.0
            
            zone_snr = self.phy.snr_for_velocity(v_final)
            if not self.phy.is_psma and zone == "Liver / Upper Abdomen": target_zone_snr = zone_snr
            elif self.phy.is_psma and zone == "Pelvis / Bladder": target_zone_snr = zone_snr

        return profile, total_time_min, target_zone_snr

    def generate_uniform_time_profile(self, time_goal_min: float) -> Tuple[List[Dict], float, float]:
        v_time = self.cfg.SCAN_LENGTH_MM / (time_goal_min * 60.0)
        v_final = round(max(self.cfg.MIN_BED_SPEED_MMS, min(v_time, self.cfg.MAX_BED_SPEED_MMS)), 1)
        profile = [{"Zone": z, "Start_Z_mm": s, "End_Z_mm": e, "Final_Velocity_mms": v_final} 
                   for z, (s, e) in AnatomySegmenter.get_zones().items()]
        return profile, time_goal_min, self.phy.snr_for_velocity(v_final)

    def generate_hybrid_earl_profile(self) -> Tuple[List[Dict], float, float]:
        if not self.phy.is_psma:
            t_bed = 1.5 if self.patient.weight_kg < 85 else (2.0 if self.patient.weight_kg <= 115 else 2.5)
        else:
            act_kg = self.phy.a_scan_mbq / self.patient.weight_kg
            t_bed = max(8.0 / act_kg, 1.5) if act_kg > 0 else 2.0
            
        v_hybrid = self.cfg.HYBRID_FOV_MM / (t_bed * 60.0)
        v_final = round(max(self.cfg.MIN_BED_SPEED_MMS, min(v_hybrid, self.cfg.MAX_BED_SPEED_MMS)), 1)
        profile = [{"Zone": z, "Start_Z_mm": s, "End_Z_mm": e, "Final_Velocity_mms": v_final} 
                   for z, (s, e) in AnatomySegmenter.get_zones().items()]
        return profile, self.cfg.SCAN_LENGTH_MM / (v_final * 60.0), self.phy.snr_for_velocity(v_final)

# ==========================================
# 5. BACKEND: CT CONTRAST FLOW-ANCHORED OPTIMIZER
# ==========================================
class CTContrastConfig:
    CATHETER_FLOWS = {
        "20G (Pink)": {"base": 3.0, "max": 5.0},
        "22G (Blue)": {"base": 2.5, "max": 2.8},
        "24G (Yellow)": {"base": 1.5, "max": 1.8},
        "26G (Purple)": {"base": 1.0, "max": 1.0}
    }

class ContrastMediaOptimizer:
    def __init__(self, patient: PatientModel, age: int, creatinine_mg_dl: float, catheter_name: str):
        self.patient = patient
        self.age = age
        self.creatinine = creatinine_mg_dl
        self.catheter_name = catheter_name
        self.cfg = CTContrastConfig()
        
    def calculate_renal_risk(self) -> Tuple[float, bool]:
        k = 0.9 if self.patient.gender == 'Male' else 0.7
        a = -0.302 if self.patient.gender == 'Male' else -0.241
        f = 1.0 if self.patient.gender == 'Male' else 1.012
        scr = self.creatinine / k
        egfr = 142 * (min(scr, 1)**a) * (max(scr, 1)**-1.2) * (0.9938**self.age) * f
        return round(egfr, 1), egfr < 60

    def generate_protocol(self) -> Dict:
        egfr, is_renal_risk = self.calculate_renal_risk()
        lbw = self.patient.lbm_kg
        bmi = self.patient.bmi
        
        vol_coeff = 1.0 
        ideal_vol = lbw * vol_coeff
        safety_cap = self.patient.weight_kg * 1.13
        final_vol = max(40.0, min(ideal_vol, safety_cap, 130.0))
        
        if final_vol == safety_cap: volume_logic = "Capped by 1.13 TBW Safety Limit"
        elif final_vol == 130.0: volume_logic = "Capped by 130 mL Absolute Max"
        elif final_vol == 40.0: volume_logic = "Floored at 40 mL Minimum Bolus"
        else: volume_logic = "Optimized via 1.0 mL/kg LBW"

        target_idr = max(0.7, self.patient.bsa * 0.6) 
        
        cath_settings = self.cfg.CATHETER_FLOWS.get(self.catheter_name, {"base": 2.5, "max": 2.8})
        base_flow = cath_settings["base"]
        max_flow = cath_settings["max"]
        
        current_flow = base_flow
        conc = 300
        current_idr = (current_flow * conc) / 1000.0
        
        flow_alert = f"Anchored to standard {self.catheter_name} flow ({base_flow} mL/s)."
        requires_warming = False

        if current_idr < target_idr:
            current_flow = max_flow
            current_idr = (current_flow * conc) / 1000.0
            flow_alert = f"Scaled to max {self.catheter_name} flow ({max_flow} mL/s) to optimize IDR."
            
            if current_idr < target_idr:
                conc = 370
                requires_warming = True
                current_flow = base_flow
                current_idr = (current_flow * conc) / 1000.0
                flow_alert = f"Upgraded to 370 mgI/mL @ {base_flow} mL/s to boost iodine load."
                
                if current_idr < target_idr:
                    current_flow = max_flow
                    current_idr = (current_flow * conc) / 1000.0
                    flow_alert = f"Maximized {self.catheter_name} potential: {max_flow} mL/s @ 370 mgI/mL."

        actual_duration = final_vol / current_flow if current_flow > 0 else 0

        if self.patient.is_pediatric:
            kvp_setting = "80 kVp (Pediatric ALARA)"
        elif bmi >= 28.0:
            kvp_setting = "120 kVp (BMI ≥ 28 Penetration Override)"
        else:
            kvp_setting = "100 kVp (High K-Edge Attenuation)"

        base_delay = 45.0
        if actual_duration > 25.0:
            added_delay = actual_duration - 25.0
            base_delay += added_delay
            delay_logic = f"Stretched +{added_delay:.0f}s to chase prolonged {actual_duration:.0f}s bolus."
        else:
            delay_logic = "Ideal compact bolus timing (<25s)."

        return {
            "biometrics": {"lbw": round(lbw, 1), "egfr": egfr, "risk": is_renal_risk},
            "injection": {
                "volume": round(final_vol),
                "saline_flush": round(final_vol),
                "flow_rate": round(current_flow, 1),
                "duration": round(actual_duration, 1),
                "concentration": conc,
                "actual_idr": round(current_idr, 2),
                "warm_alert": requires_warming
            },
            "technique": {
                "kvp": kvp_setting,
                "volume_logic": volume_logic,
                "flow_logic": flow_alert
            },
            "timing": {
                "roi": "Abdominal Aorta",
                "trigger_hu": 110,
                "post_trigger_delay": round(base_delay),
                "delay_logic": delay_logic
            }
        }

# ==========================================
# 6. FRONTEND: STREAMLIT UI
# ==========================================
def render_css():
    st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; max-width: 98%; }
    h1 { font-size: 30px !important; margin-bottom: 0.5rem !important; padding-top: 0 !important; }
    .metric-card { background-color: #F8FDFF; padding: 15px 18px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid #0288D1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .ct-card { background-color: #FAFAFA; padding: 24px 18px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid #7B1FA2; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .compare-card { background-color: #ffffff; padding: 14px; border-radius: 8px; border: 1px solid #ddd; text-align: center; margin-bottom: 12px; }
    .compare-title { font-size: 16px; font-weight: bold; color: #333; margin-bottom: 4px; }
    .compare-val { font-size: 24px; font-weight: 800; color: #0288D1; line-height: 1.1; }
    .big-font { font-size: 24px !important; font-weight: 800; color: #0277BD; line-height: 1.1; margin-bottom: 2px; }
    .ct-font { font-size: 24px !important; font-weight: 800; color: #7B1FA2; line-height: 1.1; margin-bottom: 2px; }
    .zone-label { font-size: 15px !important; font-weight: 600; color: #37474F; margin-bottom: 0px; }
    .z-axis-text { font-size: 12px; color: #E65100; font-weight: 600; margin-bottom: 2px; }
    .sub-text { font-size: 12px; color: #666; margin-top: 4px; line-height: 1.2; }
    .highlight { color: #D81B60; font-weight: 600; }
    .penalty-alert { color: #D32F2F; font-weight: bold; }
    .pediatric-banner { color: #F57C00; font-weight: bold; }
    div[data-testid="stRadio"] > label { display: none; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Biograph Vision Engine", layout="wide", initial_sidebar_state="expanded")
    render_css()

    if "selected_protocol" not in st.session_state:
        st.session_state.selected_protocol = "SNR-Guided"

    def update_protocol(prot_name):
        st.session_state.selected_protocol = prot_name

    # --- INPUTS (SIDEBAR) ---
    st.sidebar.markdown("### 📋 Patient Demographics")
    is_pediatric = st.sidebar.toggle("🧸 Pediatric Mode", value=False)
    
    col_w, col_h = st.sidebar.columns(2)
    with col_w: weight = st.number_input("Weight (kg)", value=25.0 if is_pediatric else 90.0, step=1.0)
    with col_h: height = st.number_input("Height (cm)", value=120.0 if is_pediatric else 175.0, step=1.0)
    gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True) 

    st.sidebar.markdown("### 💉 Radiotracer Assay")
    tracer = st.sidebar.selectbox("Protocol", ["18F-FDG", "18F-PSMA-1007", "68Ga-PSMA-11"], index=0 if is_pediatric else 1)
    
    is_psma_selected = "PSMA" in tracer
    default_uptake = 90.0 if is_psma_selected else 60.0
    
    col_act, col_up = st.sidebar.columns(2)
    with col_act: activity = st.number_input("Act. (mCi)", value=2.0 if is_pediatric else 6.5, step=0.1)
    with col_up: uptake_time = st.number_input("Uptake (min)", value=default_uptake, step=1.0)

    st.sidebar.markdown("### 🩻 Diagnostic CT Contrast")
    enable_contrast = st.sidebar.toggle("Optimize Contrast Media", value=True)
    if enable_contrast:
        col_c1, col_c2 = st.sidebar.columns(2)
        with col_c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=8 if is_pediatric else 50)
        with col_c2:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=0.9, step=0.1)
        catheter_str = st.sidebar.selectbox("Catheter Gauge", ["20G (Pink)", "22G (Blue)", "24G (Yellow)", "26G (Purple)"], index=1)

    st.sidebar.markdown("### 🎯 Tuning Goals")
    if is_pediatric:
        def_fdg = ClinicalGuidelines.PED_FDG_SNR
        def_psma = ClinicalGuidelines.PED_PSMA_SNR
    else:
        def_fdg = ClinicalGuidelines.DEFAULT_FDG_SNR
        def_psma = ClinicalGuidelines.DEFAULT_PSMA_SNR

    target_snr_fdg = st.sidebar.slider("FDG Target SNR", min_value=6.0, max_value=20.0, value=def_fdg, step=0.5)
    target_snr_psma = st.sidebar.slider("PSMA Target SNR", min_value=6.0, max_value=20.0, value=def_psma, step=0.5)
    time_goal_min = st.sidebar.slider("Time Goal (min)", min_value=2.0, max_value=30.0, value=5.0 if is_pediatric else 10.0, step=0.5)

    # --- RUN ENGINE ---
    patient = PatientModel(weight, height, gender, activity, tracer, uptake_time, is_pediatric)
    physics = PhysicsCore(patient, HardwareConfig())
    target_snr = target_snr_psma if physics.is_psma else target_snr_fdg
    target_organ = "Pelvis" if physics.is_psma else "Liver"
    
    optimizer = ProtocolOptimizer(physics, ClinicalGuidelines(), target_snr)
    snr_prof, snr_time, snr_val = optimizer.generate_modulated_profile()
    time_prof, time_time, time_val = optimizer.generate_uniform_time_profile(time_goal_min)
    ss_prof, ss_time, ss_val = optimizer.generate_hybrid_earl_profile()

    # --- DASHBOARD (MAIN) ---
    st.title("Biograph Vision: Dual-Modality Engine")
    
    cols = st.columns(3)
    protocols = [
        ("SNR-Guided", "🌟 SNR-Guided (Modulated)", snr_time, snr_val, f"Target {target_organ}"),
        ("Time-Optimized", "⏱️ Time-Optimized (Uniform)", time_time, time_val, "Resulting"),
        ("EARL-Hybrid", "📜 EARL Hybrid (Uniform)", ss_time, ss_val, "Resulting")
    ]
    
    for col, (prot_id, title, t_min, snr, label) in zip(cols, protocols):
        with col:
            is_active = st.session_state.selected_protocol == prot_id
            bg_color = "#E1F5FE" if is_active else "#ffffff"
            border = "2px solid #0288D1" if is_active else "1px solid #ddd"
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; border:{border}; padding:12px; border-radius:8px; text-align:center; margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size:16px; font-weight:bold; color:#333; margin-bottom:2px;">{title}</div>
                <div style="font-size:24px; font-weight:800; color:#0288D1; line-height:1.1;">{t_min:.1f} min</div>
                <div style="font-size:13px; color:#666; margin-top:2px;">{label} SNR: <b>{snr:.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.button(f"Map {prot_id} Protocol", key=f"btn_{prot_id}", on_click=update_protocol, args=(prot_id,), type="primary" if is_active else "secondary", use_container_width=True)

    if st.session_state.selected_protocol == "SNR-Guided": active_profile = snr_prof
    elif st.session_state.selected_protocol == "Time-Optimized": active_profile = time_prof
    else: active_profile = ss_prof

    # DYNAMIC BANNER CREATION
    ped_tag = "<span style='color:#F57C00; font-weight:bold;'>[🧸 Pediatric]</span> " if is_pediatric else ""
    banner_info = f"{ped_tag}<b>Goal SNR:</b> {target_snr} &nbsp;|&nbsp; <b>COV Limit:</b> ≤10% &nbsp;|&nbsp; <b>PET LBM:</b> {patient.lbm_kg:.1f} kg &nbsp;|&nbsp; <b>Scan Act:</b> {physics.a_scan_mbq:.1f} MBq"
    
    # Append the BSA and Goal IDR directly to the banner if CT Contrast is active
    if enable_contrast:
        target_idr = max(0.7, patient.bsa * 0.6)
        banner_info += f" &nbsp;|&nbsp; <span style='color:#7B1FA2;'><b>BSA:</b> {patient.bsa:.2f} m² &nbsp;|&nbsp; <b>Target IDR:</b> {target_idr:.2f} gI/s</span>"

    st.markdown(f"<div style='background-color:#F5F5F5; border:1px solid #E0E0E0; padding:8px; border-radius:6px; font-size:14px; margin-top:10px; margin-bottom:16px; text-align:center; color:#424242;'>{banner_info}</div>", unsafe_allow_html=True)

    # UI LAYOUT
    if enable_contrast:
        col_img, col_results, col_ct = st.columns([1.0, 1.6, 1.4])
    else:
        col_img, col_results = st.columns([1.2, 2.5])
    
    with col_img:
        image_path = "image_b82247.png"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            html_overlay = f"""
            <div style="position: relative; width: 100%; max-width: 220px; margin: 0 auto; background-color: white;">
                <img src="data:image/png;base64,{img_data}" style="width: 100%; height: auto; display: block;">
                <div style="position: absolute; top: 8%; left: 50%; transform: translateX(-50%); background: rgba(85,85,85,0.9); color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">{active_profile[0]['Final_Velocity_mms']:.1f}</div>
                <div style="position: absolute; top: 25%; left: 50%; transform: translateX(-50%); background: rgba(85,85,85,0.9); color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">{active_profile[1]['Final_Velocity_mms']:.1f}</div>
                <div style="position: absolute; top: 38%; left: 50%; transform: translateX(-50%); background: rgba(194,24,91,0.9); color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">{active_profile[2]['Final_Velocity_mms']:.1f}</div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translateX(-50%); background: rgba(230,81,0,0.9); color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">{active_profile[3]['Final_Velocity_mms']:.1f}</div>
                <div style="position: absolute; top: 75%; left: 50%; transform: translateX(-50%); background: rgba(85,85,85,0.9); color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; font-size: 13px;">{active_profile[4]['Final_Velocity_mms']:.1f}</div>
            </div>
            """
            st.markdown(html_overlay, unsafe_allow_html=True)
        else:
            st.error("Silhouette Image not found.")
            
    with col_results:
        trunk_penalty = patient.trunk_attenuation_penalty()
        for zone in active_profile:
            sub_text = ""
            if st.session_state.selected_protocol == "SNR-Guided":
                if physics.is_psma:
                    if zone['Zone'] == "Pelvis / Bladder": sub_text = f"Target Area: Hybrid Base ({zone['v_base']:.1f} mm/s)"
                    elif zone['Zone'] == "Liver / Upper Abdomen": sub_text = f"Relaxed Base | <span class='highlight'>≤10% COV Applied</span>"
                    elif zone['Zone'] == "Thorax": sub_text = f"Relaxed Base | <span class='highlight'>MBP SNR Applied</span>"
                    else: sub_text = "Accelerated non-target zone."
                else:
                    if zone['Zone'] in ["Pelvis / Bladder", "Head & Neck"]: sub_text = f"Accelerated to prevent saturation."
                    else: sub_text = f"NECR Base Optimized: {zone['v_base']:.1f} mm/s"
                
                if zone['Zone'] in ["Thorax", "Liver / Upper Abdomen"] and trunk_penalty < 1.0:
                    sub_text += f"<br><span class='penalty-alert'>⚠️ BMI Trunk Penalty ({trunk_penalty:.2f}x)</span>"
            else:
                sub_text = "Uniform baseline speed applied."

            st.markdown(f"""
            <div class="metric-card">
                <div class="zone-label">{zone['Zone']} <span class="z-axis-text">({zone['Start_Z_mm']} - {zone['End_Z_mm']})</span></div>
                <div class="big-font">{zone['Final_Velocity_mms']:.1f} mm/s</div>
                <div class="sub-text">{sub_text}</div>
            </div>
            """, unsafe_allow_html=True)

    if enable_contrast:
        with col_ct:
            optimizer_ct = ContrastMediaOptimizer(patient, age, creatinine, catheter_str)
            ct_res = optimizer_ct.generate_protocol()
            
            if ct_res['biometrics']['risk']:
                st.markdown("<div style='background-color:#ffebee; color:#c62828; padding:8px 12px; border-radius:6px; font-size:13px; font-weight:bold; margin-bottom:12px; text-align:center;'>⚠️ HIGH RENAL RISK (eGFR < 60)</div>", unsafe_allow_html=True)

            # Card 1: Volume
            st.markdown(f"<div class='ct-card'><div class='zone-label'>Total Volume</div><div class='ct-font'>{ct_res['injection']['volume']} mL</div><div class='sub-text'>{ct_res['technique']['volume_logic']}</div></div>", unsafe_allow_html=True)
            
            # Card 2: Flow & Concentration
            warm_badge = "<span style='color:#D81B60; font-weight:bold;'>🔥 WARM</span><br>" if ct_res['injection']['warm_alert'] else ""
            alert_class = "penalty-alert" if "WARNING" in ct_res['technique']['flow_logic'] else ("highlight" if "Upgraded" in ct_res['technique']['flow_logic'] or "Maximized" in ct_res['technique']['flow_logic'] else "sub-text")
            st.markdown(f"<div class='ct-card'><div class='zone-label'>Flow Ladder</div><div class='ct-font'>{ct_res['injection']['flow_rate']} mL/s</div><div class='sub-text'>{warm_badge}Conc: {ct_res['injection']['concentration']} mgI | IDR: {ct_res['injection']['actual_idr']} gI/s<br><span class='{alert_class}'>{ct_res['technique']['flow_logic']}</span></div></div>", unsafe_allow_html=True)
            
            # Card 3: kVp
            alert_kvp = "penalty-alert" if "Override" in ct_res['technique']['kvp'] else "sub-text"
            st.markdown(f"<div class='ct-card'><div class='zone-label'>CT Energy</div><div class='ct-font'>{ct_res['technique']['kvp'][:7]}</div><div class='sub-text'><span class='{alert_kvp}'>{ct_res['technique']['kvp'][8:]}</span></div></div>", unsafe_allow_html=True)
            
            # Card 4: Timing
            st.markdown(f"<div class='ct-card'><div class='zone-label'>PV Scan Delay</div><div class='ct-font'>{ct_res['timing']['post_trigger_delay']} s</div><div class='sub-text'>Post-Trigger (Abdominal Aorta @ {ct_res['timing']['trigger_hu']} HU)<br><i>{ct_res['timing']['delay_logic']}</i></div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
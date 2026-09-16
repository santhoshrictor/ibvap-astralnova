"""
IBVAP - Intelligent Border Video Analytics Platform
Prototype v2.0  |  College Hackathon Edition

Capabilities:
  1. Detection & Tracking  - YOLOv8n + centroid tracking + motion trails
  2. Identification        - Face detection (Haar cascade) + ANPR (EasyOCR)
  3. Spatial Analytics     - Virtual fence line crossing -> RED alert
  4. Environmental Adapt.  - Night-mode confidence relaxation + badge
  5. Automated Action      - Real-time event log + behavioral pattern alerts

Run:
  pip install streamlit opencv-python ultralytics easyocr pandas numpy
  streamlit run app.py
"""

import os
import time
import math
import threading
from collections import deque
from datetime import datetime, time as dtime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IBVAP | Border Video Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
:root {
    --ac:#00d4ff; --ac-lo:rgba(0,212,255,0.15); --ac-gl:rgba(0,212,255,0.28);
    --rd:#ff3d3d; --rd-lo:rgba(255,61,61,0.10);
    --am:#ffaa00; --am-lo:rgba(255,170,0,0.10);
    --gn:#00e887; --gn-lo:rgba(0,232,135,0.10);
    --pu:#c084fc; --pu-lo:rgba(192,132,252,0.10);
    --bg:#050a10; --card:rgba(11,18,28,0.92);
    --bd:rgba(0,212,255,0.09); --bd-hi:rgba(0,212,255,0.28);
    --tx:#bdd4e0; --dm:#3a6070;
    --mo:'JetBrains Mono','Consolas',monospace;
    --ui:'Inter',system-ui,sans-serif;
}
html { scroll-behavior: smooth !important; }
.stApp {
    background: var(--bg);
    background-image:
        linear-gradient(rgba(0,212,255,0.022) 1px, transparent 1px),
        linear-gradient(90deg,rgba(0,212,255,0.022) 1px, transparent 1px),
        radial-gradient(ellipse 80% 35% at 50% 0%,rgba(0,212,255,0.055),transparent 65%);
    background-size:30px 30px,30px 30px,100% 100%; font-family:var(--ui);
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bd-hi); border-radius: 4px; border: 1px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: var(--ac); }

*,*::before,*::after{box-sizing:border-box;}
.stButton button, .scard, .cam-card, .t-card { transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; }
.stButton button:hover { box-shadow: 0 0 15px var(--ac-gl); transform: translateY(-2px); }
h1,h2,h3,h4{font-family:var(--mo);color:var(--tx);}
p,.stMarkdown p{color:var(--tx);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#060b12 0%,#04080d 100%);border-right:1px solid var(--bd-hi);}
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stMarkdown p{font-family:var(--mo)!important;font-size:0.64rem!important;text-transform:uppercase;letter-spacing:1.2px;color:var(--dm)!important;}
section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] h4{font-family:var(--mo);color:var(--ac);font-size:0.7rem;letter-spacing:2px;border-bottom:1px solid var(--bd);padding-bottom:4px;margin:14px 0 8px;}
section[data-testid="stSidebar"] .stButton button{font-family:var(--mo)!important;font-weight:700!important;letter-spacing:1.5px!important;border-radius:4px!important;text-transform:uppercase!important;font-size:0.7rem!important;}
.ibvap-title{font-family:var(--mo);font-weight:700;font-size:1.55rem;letter-spacing:3px;color:var(--ac);text-shadow:0 0 22px var(--ac-gl),0 0 55px rgba(0,212,255,0.08);line-height:1.1;}
.ibvap-sub{font-family:var(--mo);font-size:0.6rem;color:var(--dm);letter-spacing:2.5px;text-transform:uppercase;margin-top:3px;margin-bottom:6px;}
.cmd-bar{display:flex;border:1px solid var(--bd-hi);border-radius:8px;overflow:hidden;margin:10px 0 16px;background:linear-gradient(135deg,rgba(0,212,255,0.025),rgba(0,0,0,0.45));}
.cmd-item{flex:1;padding:8px 14px;border-right:1px solid var(--bd);font-family:var(--mo);min-width:0;}
.cmd-item:last-child{border-right:none;}
.cmd-lbl{display:block;font-size:0.53rem;letter-spacing:2px;color:var(--dm);text-transform:uppercase;margin-bottom:2px;}
.cmd-val{display:block;font-size:0.82rem;font-weight:700;color:var(--ac);}
.cmd-val.armed{color:var(--gn);text-shadow:0 0 8px rgba(0,232,135,0.4);}
.cmd-val.standby{color:var(--am);}
.cmd-val.c-red{color:var(--rd);}
.cmd-val.c-amber{color:var(--am);}
.cmd-val.c-blue{color:var(--ac);}
.stats-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:18px;}
.scard{background:var(--card);border:1px solid var(--bd);border-top:2px solid var(--ac);border-radius:8px;padding:11px 13px 10px;font-family:var(--mo);position:relative;overflow:hidden; cursor:default;}
.scard:hover{transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,212,255,0.12); border-color:var(--bd-hi);}
.scard.r{border-top-color:var(--rd);} .scard.a{border-top-color:var(--am);}
.scard.g{border-top-color:var(--gn);} .scard.p{border-top-color:var(--pu);}
.scard-icon{font-size:1.1rem;margin-bottom:3px;}
.scard-lbl{font-size:0.52rem;letter-spacing:2px;color:var(--dm);text-transform:uppercase;margin-bottom:2px;}
.scard-num{font-size:1.85rem;font-weight:700;color:var(--ac);line-height:1;font-variant-numeric:tabular-nums;}
.scard.r .scard-num{color:var(--rd);} .scard.a .scard-num{color:var(--am);}
.scard.g .scard-num{color:var(--gn);} .scard.p .scard-num{color:var(--pu);}
.scard-sub{font-size:0.54rem;color:var(--dm);margin-top:1px;}
.cam-card{border:1px solid var(--bd);border-radius:10px;overflow:hidden;background:var(--card);margin-bottom:10px;box-shadow:0 6px 30px rgba(0,0,0,0.55);}
.cam-card:hover{transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.8), 0 0 15px rgba(0,212,255,0.08); border-color:var(--bd-hi);}
.cam-head{display:flex;justify-content:space-between;align-items:center;padding:7px 11px;background:rgba(0,0,0,0.48);border-bottom:1px solid var(--bd);font-family:var(--mo);font-size:0.68rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--ac);}
.cam-name{display:flex;align-items:center;gap:6px;} .cam-right{display:flex;align-items:center;gap:5px;flex-shrink:0;}
.bdg{font-size:0.54rem;padding:2px 7px;border-radius:20px;font-weight:700;letter-spacing:1px;white-space:nowrap;}
.bdg-live{background:var(--gn-lo);color:var(--gn);border:1px solid rgba(0,232,135,0.28);}
.bdg-down{background:var(--rd-lo);color:#ff8888;border:1px solid rgba(255,61,61,0.28);}
.bdg-night{background:var(--pu-lo);color:var(--pu);border:1px solid rgba(192,132,252,0.28);}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0;}
.dot-g{background:var(--gn);box-shadow:0 0 8px var(--gn);}
.dot-d{background:var(--dm);}
[data-testid="stImage"]{position:relative;overflow:hidden;border-radius:4px;}
[data-testid="stImage"]::after{content:'';position:absolute;inset:0;pointer-events:none;z-index:2;background:repeating-linear-gradient(to bottom,rgba(0,212,255,0.038) 0px,rgba(0,212,255,0.038) 1px,transparent 1px,transparent 4px);}
[data-testid="stImage"]::before{content:'';position:absolute;inset:4px;pointer-events:none;z-index:3;background-image:linear-gradient(to right,var(--ac) 2px,transparent 2px),linear-gradient(to bottom,var(--ac) 2px,transparent 2px),linear-gradient(to left,var(--ac) 2px,transparent 2px),linear-gradient(to bottom,var(--ac) 2px,transparent 2px),linear-gradient(to right,var(--ac) 2px,transparent 2px),linear-gradient(to top,var(--ac) 2px,transparent 2px),linear-gradient(to left,var(--ac) 2px,transparent 2px),linear-gradient(to top,var(--ac) 2px,transparent 2px);background-repeat:no-repeat;background-size:16px 16px;background-position:top left,top left,top right,top right,bottom left,bottom left,bottom right,bottom right;opacity:0.65;}
.t-card{border-radius:7px;padding:9px 13px;margin-bottom:8px;font-family:var(--mo);font-size:0.75rem;font-weight:600;border-left:4px solid;line-height:1.45;}
.t-red{border-color:var(--rd);color:#ff9494;background:var(--rd-lo);}
.t-amber{border-color:var(--am);color:#ffd090;background:var(--am-lo);}
.t-blue{border-color:var(--ac);color:#90d8ff;background:var(--ac-lo);}
.hero{border-radius:10px;padding:14px;text-align:center;font-family:var(--mo);font-weight:800;font-size:0.9rem;letter-spacing:2px;margin-bottom:14px;border:1px solid;}
.hero-r{border-color:var(--rd);color:var(--rd);background:var(--rd-lo);}
.hero-a{border-color:var(--am);color:var(--am);background:var(--am-lo);}
.hero-b{border-color:var(--ac);color:var(--ac);background:var(--ac-lo);}
.sec-hd{font-family:var(--mo);font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#7abdd6;border-bottom:1px solid var(--bd);padding-bottom:5px;margin:0 0 10px;display:flex;align-items:center;gap:7px;}
.sec-hd::before{content:'';width:3px;height:12px;background:var(--ac);box-shadow:0 0 7px var(--ac);display:inline-block;flex-shrink:0;}
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid var(--bd);gap:0;}
.stTabs [data-baseweb="tab"]{font-family:var(--mo)!important;font-size:0.64rem!important;letter-spacing:1.5px!important;text-transform:uppercase;color:var(--dm)!important;padding:7px 16px!important;border-radius:0!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{color:var(--ac)!important;border-bottom-color:var(--ac)!important;background:rgba(0,212,255,0.04)!important;}
.stTabs [data-baseweb="tab-panel"] { animation: tabFadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
@keyframes tabFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.t-card { animation: tabFadeIn 0.3s ease-out forwards; }
.ibvap-footer{margin-top:18px;padding-top:9px;border-top:1px solid var(--bd);font-family:var(--mo);font-size:0.58rem;color:var(--dm);letter-spacing:1.5px;text-transform:uppercase;text-align:center;}
</style>
"""
st.html(_CSS)

# =============================================================================
# CONSTANTS
# =============================================================================

ALLOWED_CLASSES = {0:"PERSON", 2:"CAR", 3:"MOTORCYCLE", 5:"BUS", 7:"TRUCK"}
VEHICLE_TYPES   = {"CAR","MOTORCYCLE","BUS","TRUCK"}

BOX_COLORS = {
    "PERSON":     (0, 220, 255),
    "FACE":       (0, 255, 185),
    "CAR":        (60, 220, 100),
    "MOTORCYCLE": (60, 220, 100),
    "BUS":        (60, 220, 100),
    "TRUCK":      (60, 220, 100),
    "PLATE":      (255, 0, 255),
}

RECON_WINDOW_SEC    = 60
RECON_MIN_CAMERAS   = 3
SHUTTLE_WINDOW_SEC  = 15
EVENT_COOLDOWN_SEC  = 3
INCIDENT_GROUP_WIN  = 45
TIMELINE_RETAIN_SEC = 180
DECAY_TAU_SEC       = 25
ACTIVE_TRACK_TTL    = 45
BLIND_GAP_SEC       = 8
LOW_LIGHT_THRESHOLD = 65
PANEL_REFRESH_N     = 4

CAMERA_POSTS = [
    ("CAM_1", "HAWKINS POST"),
    ("CAM_2", "OUT POST"),
    ("CAM_3", "ALPHA POST"),
]

_DEMO_VIDEOS = sorted(f for f in os.listdir(".") if f.lower().endswith(".mp4"))

# =============================================================================
# ACTIVITY ENGINE
# =============================================================================

class ActivityEngine:
    """
    Per-camera centroid tracker + deterministic rule engine.
    Alerts describe observable motion patterns only.
    They are review cues, not proof of intent.
    """

    def __init__(self, camera_id, *, fence_line=None, protected_polygon=None,
                 protected_side="RIGHT", speed_threshold=260.0,
                 loiter_seconds=20.0, loiter_radius=45.0, min_conf=0.45,
                 fence_band_px=35.0, confirm_frames=3, breach_window_sec=4.0,
                 max_match_distance=90.0, track_ttl=5.0, history_sec=8.0):

        self.camera_id    = camera_id
        self.fence_line   = fence_line
        self.prot_poly    = protected_polygon
        self.prot_sign    = -1.0 if str(protected_side).upper()=="RIGHT" else 1.0
        self.spd_thresh   = float(speed_threshold)
        self.loiter_sec   = float(loiter_seconds)
        self.loiter_rad   = float(loiter_radius)
        self.min_conf     = float(min_conf)
        self.fence_band   = float(fence_band_px)
        self.confirm_frm  = int(confirm_frames)
        self.breach_win   = float(breach_window_sec)
        self.max_match    = float(max_match_distance)
        self.track_ttl    = float(track_ttl)
        self.hist_sec     = float(history_sec)
        self.tracks       = {}
        self.next_id      = 1

    @staticmethod
    def _ctr(b):
        return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    @staticmethod
    def _cross(a, b, p):
        return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])

    @staticmethod
    def _in_poly(pt, poly):
        if not poly: return False
        x,y=pt; inside=False; j=len(poly)-1
        for i in range(len(poly)):
            xi,yi=poly[i]; xj,yj=poly[j]
            if ((yi>y)!=(yj>y)) and (x<(xj-xi)*(y-yi)/((yj-yi) or 1e-9)+xi):
                inside=not inside
            j=i
        return inside

    def _sdist(self, pt):
        if not self.fence_line: return 0.0
        a,b=self.fence_line
        return self._cross(a,b,pt)/max(self._dist(a,b),1.0)

    def _new_track(self, lbl, ctr, bbox, conf, kpts, now):
        t={
            "id":self.next_id,"label":lbl,
            "history":deque([(now,ctr,bbox,conf,kpts)],maxlen=80),
            "last_seen":now,"last_alert":{},
            "pre_side":False,"near_seen":False,
            "near_since":None,"confirm_cnt":0,
            "breach_cue":False,"flash_until":0.0,
        }
        self.next_id+=1; self.tracks[t["id"]]=t; return t

    def _get_track(self, lbl, ctr, bbox, conf, kpts, now):
        cands=[t for t in self.tracks.values()
               if t["label"]==lbl and now-t["last_seen"]<=self.track_ttl]
        if cands:
            best=min(cands,key=lambda t:self._dist(t["history"][-1][1],ctr))
            if self._dist(best["history"][-1][1],ctr)<=self.max_match:
                return best
        return self._new_track(lbl,ctr,bbox,conf,kpts,now)

    def _can_alert(self, track, kind, now, cd=8.0):
        if now-track["last_alert"].get(kind,0.0)<cd: return False
        track["last_alert"][kind]=now; return True

    def _metrics(self, track):
        hist=list(track["history"])
        if len(hist)<4: return 0.0,0,0.0
        speeds,dirs=[],[]
        for p,c in zip(hist[:-1],hist[1:]):
            dt=max(c[0]-p[0],0.001)
            dx=c[1][0]-p[1][0]; dy=c[1][1]-p[1][1]
            speeds.append(math.hypot(dx,dy)/dt); dirs.append((dx,dy))
        turns=0
        for a,b in zip(dirs[:-1],dirs[1:]):
            ma,mb=math.hypot(*a),math.hypot(*b)
            if ma<10 or mb<10: continue
            cos=max(-1.0,min(1.0,(a[0]*b[0]+a[1]*b[1])/(ma*mb)))
            if math.degrees(math.acos(cos))>120: turns+=1
        return sum(speeds)/len(speeds),turns,self._dist(hist[0][1],hist[-1][1])

    def _alert(self, track, kind, level, reason, now, ev):
        return {"kind":kind,"level":level,"track_id":track["id"],
                "label":track["label"],"camera_id":self.camera_id,
                "timestamp":now,"reason":reason,"evidence":ev}

    def update(self, detections, now=None):
        now=time.time() if now is None else now
        self.tracks={k:v for k,v in self.tracks.items()
                     if now-v["last_seen"]<=self.track_ttl}
        alerts=[]
        for det in detections:
            lbl=str(det.get("label",""))
            if lbl not in ALLOWED_CLASSES.values(): continue
            bbox=tuple(int(v) for v in det["bbox"])
            ctr=self._ctr(bbox); conf=float(det.get("conf",0.0))
            kpts=det.get("keypoints")
            if conf<self.min_conf: continue
            track=self._get_track(lbl,ctr,bbox,conf,kpts,now)
            track["history"].append((now,ctr,bbox,conf,kpts))
            track["last_seen"]=now
            while track["history"] and now-track["history"][0][0]>self.hist_sec:
                track["history"].popleft()

            if self.fence_line:
                sd=self._sdist(ctr); pd=sd*self.prot_sign
                near=abs(sd)<=self.fence_band
                prev=track["history"][-2] if len(track["history"])>=2 else None
                if pd<-self.fence_band: track["pre_side"]=True
                if near and track["pre_side"]:
                    track["near_seen"]=True
                    if track["near_since"] is None: track["near_since"]=now
                rv_cue=rt_cue=False
                if prev:
                    dx=ctr[0]-prev[1][0]; dy=ctr[1]-prev[1][1]
                    ph=max(20,bbox[3]-bbox[1]); pw=max(12,bbox[2]-bbox[0])
                    rv_cue=dy<-0.25*ph; rt_cue=abs(dx)>max(35.0,0.60*pw)
                    track["breach_cue"]=track["breach_cue"] or rv_cue or rt_cue
                if pd>self.fence_band: track["confirm_cnt"]+=1
                else: track["confirm_cnt"]=0
                confirmed=(pd>self.fence_band and track["near_seen"] and track["pre_side"]
                           and track["near_since"] is not None
                           and now-track["near_since"]<=self.breach_win
                           and track["confirm_cnt"]>=self.confirm_frm
                           and track["breach_cue"])
                if confirmed and self._can_alert(track,"FENCE_BREACH",now,10.0):
                    kind="JUMP/CLIMB BREACH" if rv_cue else "CONFIRMED FENCE BREACH"
                    reason=("Confirmed entry with upward jump cue." if rv_cue
                            else "Confirmed protected-side entry after near-fence movement.")
                    alerts.append(self._alert(track,kind,"RED",reason,now,
                                              {"signed_dist_px":round(sd,1),"bbox":bbox,
                                               "confirm_frames":track["confirm_cnt"]}))
                    track.update(near_seen=False,pre_side=False,near_since=None,
                                 confirm_cnt=0,breach_cue=False)
                    track["flash_until"]=now+5.0

            if self.prot_poly and self._in_poly(ctr,self.prot_poly):
                if self._can_alert(track,"ZONE_INTRUSION",now,6.0):
                    alerts.append(self._alert(track,"ZONE INTRUSION","RED",
                        "Target entered restricted zone.",now,{"bbox":bbox}))

            if len(track["history"])>=4:
                spd,turns,disp=self._metrics(track)
                if spd>=self.spd_thresh and self._can_alert(track,"HIGH_SPEED",now) and disp > 50:
                    alerts.append(self._alert(track,"HIGH SPEED","AMBER",
                        "Image-plane speed exceeded threshold.",now,{"avg_px_per_sec":round(spd,1)}))
                if turns>=3 and self._can_alert(track,"ERRATIC_MOTION",now):
                    alerts.append(self._alert(track,"ERRATIC MOTION","AMBER",
                        "Repeated direction reversals observed.",now,{"turns":turns}))
                first=track["history"][0]
                if now-first[0]>=self.loiter_sec and disp<=self.loiter_rad:
                    if self._can_alert(track,"LOITERING",now,30.0):
                        alerts.append(self._alert(track,"LOITERING","AMBER",
                            "Target remained in small area for extended period.",now,
                            {"duration_sec":round(now-first[0],1),"displacement_px":round(disp,1)}))
            
            # Independent Climbing/Jumping Detection (only if near fence/zone)
            is_near_fence = False
            if self.fence_line and abs(self._sdist(ctr)) <= self.fence_band * 3:
                is_near_fence = True
            if self.prot_poly:
                is_near_fence = True

            if is_near_fence and len(track["history"])>=3:
                prev=track["history"][-3]
                curr=track["history"][-1]
                
                # Check for skeleton pose data
                if prev[4] is not None and curr[4] is not None and len(prev[4]) >= 17 and len(curr[4]) >= 17:
                    pk, ck = prev[4], curr[4]
                    l_ankle_dy = ck[15][1] - pk[15][1]
                    r_ankle_dy = ck[16][1] - pk[16][1]
                    l_wrist_dy = ck[9][1] - pk[9][1]
                    r_wrist_dy = ck[10][1] - pk[10][1]
                    ph=max(20,curr[2][3]-curr[2][1])
                    
                    if min(l_ankle_dy, r_ankle_dy, l_wrist_dy, r_wrist_dy) < -0.15*ph:
                        if self._can_alert(track,"CLIMBING",now,10.0):
                            alerts.append(self._alert(track,"POSE CLIMBING DETECTED","RED",
                                "Skeletal upward vertical motion detected.",now,{"dy_wrist":round(min(l_wrist_dy, r_wrist_dy),1)}))
                            track["flash_until"]=now+5.0
                else:
                    dy=ctr[1]-prev[1][1]
                    ph=max(20,bbox[3]-bbox[1])
                    rv_cue=dy<-0.20*ph # 20% of bounding box height upward movement
                    if rv_cue and self._can_alert(track,"CLIMBING",now,10.0):
                        alerts.append(self._alert(track,"CLIMBING DETECTED","RED",
                            "Upward vertical motion (climbing/jumping) detected.",now,{"dy":round(dy,1)}))
                        track["flash_until"]=now+5.0
                        
            if now <= track.get("flash_until", 0.0) or bool(self.prot_poly and self._in_poly(ctr,self.prot_poly)):
                det["alert"] = True
                det["conf"] = 0.24
                
        return alerts

    def draw(self, frame, now=None):
        now=time.time() if now is None else now
        for track in self.tracks.values():
            if not track["history"]: continue
            _,ctr,bbox,_,_=track["history"][-1]
            x1,y1,x2,y2=bbox
            flashing=now<=track.get("flash_until",0.0)
            in_zone=bool(self.prot_poly and self._in_poly(ctr,self.prot_poly))
            if flashing:
                col,thick,tag=(0,0,255),3,f"ID{track['id']} {track['label']} BREACH"
            elif in_zone:
                col,thick,tag=(0,120,255),2,f"ID{track['id']} {track['label']} IN-ZONE"
            else:
                col,thick,tag=(0,220,255),2,f"ID{track['id']} {track['label']}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,thick)
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
            cv2.rectangle(frame,(x1,max(0,y1-th-8)),(x1+tw+5,y1),col,-1)
            cv2.putText(frame,tag,(x1+2,max(12,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.46,(8,8,8),1,cv2.LINE_AA)
            trail=list(track["history"])[-20:]
            for a,b in zip(trail[:-1],trail[1:]):
                cv2.line(frame,tuple(map(int,a[1])),tuple(map(int,b[1])),col,2)
        if self.fence_line:
            fa,fb=self.fence_line
            cv2.line(frame,tuple(map(int,fa)),tuple(map(int,fb)),(0,0,200),2)
            mid=(int((fa[0]+fb[0])/2)+4,int((fa[1]+fb[1])/2)-5)
            cv2.putText(frame,"VIRTUAL FENCE",mid,
        cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,50,200),1,cv2.LINE_AA)
        return frame

# =============================================================================
# CAMERA NODE
# =============================================================================

class CameraNode:
    """4-level fail-safe: URL -> file (looped) -> webcam -> synthetic frame."""

    def __init__(self, cam_id, label, url, fpath, webcam_idx=0, allow_webcam=True):
        self.cam_id=cam_id; self.label=label
        self.url=(url or "").strip(); self.fpath=(fpath or "").strip()
        self.webcam_idx=int(webcam_idx); self.allow_webcam=bool(allow_webcam)
        self.cap=None; self.mode="SYNTH"; self.source_desc="none"; self.fail_count=0
        self._lock=threading.Lock()
        self._latest_frame=None
        self._stop=threading.Event()
        self._open()
        self._thread=threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    @property
    def health(self):
        if self.mode=="SYNTH": return "DOWN"
        return "DEGRADED" if self.fail_count>0 else "OK"

    def _try(self, src, mode, desc):
        try:
            cap=cv2.VideoCapture(src)
            if cap and cap.isOpened():
                self.cap,self.mode,self.source_desc=cap,mode,desc; return True
        except Exception: pass
        return False

    def _open(self):
        if self.url   and self._try(self.url,"URL",self.url): return
        if self.fpath and self._try(self.fpath,"FILE",self.fpath): return
        if self.allow_webcam and self._try(self.webcam_idx,"WEBCAM",f"Webcam {self.webcam_idx}"): return
        self.mode,self.source_desc="SYNTH","No signal source"

    def _synth(self, msg="NO SIGNAL"):
        f=np.zeros((360,640,3),dtype=np.uint8); f[:]=(7,11,17)
        cv2.putText(f,f"[{self.label}]",(22,40),cv2.FONT_HERSHEY_SIMPLEX,0.65,(55,75,95),1,cv2.LINE_AA)
        cv2.putText(f,msg,(22,185),cv2.FONT_HERSHEY_SIMPLEX,0.85,(45,45,175),2,cv2.LINE_AA)
        cv2.putText(f,datetime.now().strftime("%H:%M:%S"),(22,315),cv2.FONT_HERSHEY_SIMPLEX,0.55,(40,60,80),1,cv2.LINE_AA)
        return f

    def _capture_loop(self):
        while not self._stop.is_set():
            if self.mode=="SYNTH" or self.cap is None:
                with self._lock:
                    self._latest_frame=self._synth("NO SIGNAL - CHECK SOURCE")
                time.sleep(0.1)
                continue
            
            try: ok,frame=self.cap.read()
            except Exception: ok,frame=False,None
            
            if not ok or frame is None:
                self.fail_count=min(self.fail_count+1,20)
                if self.mode=="FILE":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    try: ok,frame=self.cap.read()
                    except Exception: ok,frame=False,None
                if not ok or frame is None:
                    self._open()
                    if self.mode=="SYNTH": 
                        with self._lock: self._latest_frame=self._synth("SIGNAL LOST - RECONNECTING...")
                        time.sleep(0.5)
                        continue
                    try: ok,frame=self.cap.read()
                    except Exception: ok,frame=False,None
                    if not ok or frame is None:
                        with self._lock: self._latest_frame=self._synth("SIGNAL LOST - RECONNECTING...")
                        time.sleep(0.5)
                        continue
            else:
                self.fail_count=0
                
            with self._lock:
                self._latest_frame=frame
                
            if self.mode=="FILE":
                time.sleep(getattr(self, "playback_delay", 0.015))

    def read(self):
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return self._synth("INITIALIZING...")
        return frame.copy()

    def release(self):
        self._stop.set()
        if hasattr(self, '_thread'): self._thread.join(timeout=1.0)
        try:
            if self.cap: self.cap.release()
        except Exception: pass

# =============================================================================
# MODEL LOADING (cached once per process)
# =============================================================================

@st.cache_resource(show_spinner="Loading YOLOv8n detection model...")
def load_yolo():
    import torch
    from ultralytics import YOLO
    model = YOLO("yolov8n-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        pass
    return model

@st.cache_resource(show_spinner="Loading EasyOCR (ANPR engine)...")
def load_ocr():
    import torch
    import easyocr
    use_gpu = torch.cuda.is_available()
    try:
        return easyocr.Reader(["en"], gpu=use_gpu)
    except Exception:
        return easyocr.Reader(["en"], gpu=False)

@st.cache_resource(show_spinner="Loading YOLOv8 plate model...")
def load_plate_model():
    import torch
    from ultralytics import YOLO
    path = "plate_yolo.pt"
    if os.path.exists(path):
        model = YOLO(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model.to(device)
        except Exception:
            pass
        return model
    return None

@st.cache_resource(show_spinner="Loading YuNet face detection model...")
def load_face_cascade():
    model_path = "face_detection_yunet.onnx"
    if not os.path.exists(model_path):
        try:
            import urllib.request
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, model_path)
        except Exception:
            return None
    try:
        fd = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=0.6, nms_threshold=0.3)
        return fd
    except Exception:
        return None

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

_MODEL_LOCK = threading.Lock()

def run_inference(model, frame, conf_thresh, ocr, face_cascade, plate_model):
    """Run YOLO + face detection + ANPR. Returns list of detection dicts."""
    detections=[]
    gray_mean=float(np.mean(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)))
    is_night=gray_mean<LOW_LIGHT_THRESHOLD
    eff_conf=max(0.15,conf_thresh-0.15) if is_night else conf_thresh

    # YOLO detection
    try:
        with _MODEL_LOCK:
            results=model.predict(frame,conf=eff_conf,imgsz=320,verbose=False)
        for i, box in enumerate(results[0].boxes):
            cls_id=int(box.cls[0])
            if cls_id not in ALLOWED_CLASSES: continue
            label=ALLOWED_CLASSES[cls_id]; conf=float(box.conf[0])
            x1,y1,x2,y2=map(int,box.xyxy[0])
            plate_text=""
            keypoints = None
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                if len(results[0].keypoints.xy) > i:
                    keypoints = results[0].keypoints.xy[i].cpu().numpy()
            
            if ocr and plate_model is not None and label in VEHICLE_TYPES and conf>0.55 and (x2-x1)>65:
                crop=frame[max(0,y1):y2,max(0,x1):x2]
                if crop.size > 0:
                    with _MODEL_LOCK:
                        plate_results = plate_model.predict(crop, conf=0.15, imgsz=320, verbose=False)
                    
                    if len(plate_results[0].boxes) > 0:
                        # Get the plate box with the highest confidence
                        best_plate = max(plate_results[0].boxes, key=lambda b: float(b.conf[0]))
                        px1, py1, px2, py2 = map(int, best_plate.xyxy[0])
                        p_conf = float(best_plate.conf[0])
                        
                        abs_px1 = x1 + px1; abs_py1 = max(0, y1) + py1
                        abs_px2 = x1 + px2; abs_py2 = max(0, y1) + py2
                        
                        # Now OCR just this plate!
                        plate_crop = crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            # Upscale for better OCR accuracy on small text
                            plate_crop = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            try:
                                res=ocr.readtext(plate_crop,detail=0)
                                # Lower thresholds slightly: 3 chars min, since plates can be short or occluded
                                cands=[t for t in res if len(t)>=3 and any(c.isdigit() for c in t)]
                                if cands: 
                                    plate_text="".join(cands).upper().replace(" ", "")
                                    # Add the localized plate as its own detection so we draw a small box
                                    detections.append({"label":"PLATE", "conf":round(p_conf, 2), 
                                                       "bbox":(abs_px1, abs_py1, abs_px2, abs_py2), 
                                                       "plate":"", "is_night":is_night})
                                else:
                                    plate_text="NOT VISIBLE"
                            except Exception: 
                                plate_text="NOT VISIBLE"
                        else:
                            plate_text="NOT VISIBLE"
                    else:
                        plate_text="NOT VISIBLE"

            detections.append({"label":label,"conf":conf,
                                "bbox":(x1,y1,x2,y2),"plate":plate_text,"is_night":is_night,
                                "keypoints": keypoints})
    except Exception: pass

    # Face detection via YuNet (cv2.FaceDetectorYN)
    if face_cascade is not None:
        try:
            h_f, w_f = frame.shape[:2]
            face_cascade.setInputSize((w_f, h_f))
            _, faces = face_cascade.detect(frame)
            if faces is not None:
                for face in faces:
                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    conf_f = float(face[14]) if len(face) > 14 else 0.80
                    detections.append({"label":"FACE","conf":round(conf_f, 2),
                                       "bbox":(fx, fy, fx+fw, fy+fh),
                                       "plate":"","is_night":is_night})
        except Exception: pass

    return detections


def draw_detections(frame, detections, skip_labels=None):
    """Draw bounding boxes + labels. Optionally skip certain labels."""
    skip=skip_labels or set()
    for d in detections:
        lbl=d["label"]
        if lbl in skip: continue
        x1,y1,x2,y2=d["bbox"]
        is_alert = d.get("alert", False)
        if is_alert:
            col = (0, 0, 255)
            tag = f"BREACH {d['conf']*100:.0f}%"
        else:
            col=BOX_COLORS.get(lbl,(255,255,255))
            tag="FACE" if lbl=="FACE" else f"{lbl}  {d['conf']*100:.0f}%"
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
        cv2.rectangle(frame,(x1,max(0,y1-th-8)),(x1+tw+5,y1),col,-1)
        cv2.putText(frame,tag,(x1+2,max(12,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,(8,8,8),1,cv2.LINE_AA)
        if d.get("plate"):
            cv2.putText(frame,f"PLATE: {d['plate']}",(x1,min(frame.shape[0]-6,y2+18)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,255),2,cv2.LINE_AA)
                        
        if d.get("keypoints") is not None and len(d["keypoints"]) > 0:
            kpts = d["keypoints"]
            skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
            for pt1, pt2 in skeleton:
                if pt1 < len(kpts) and pt2 < len(kpts):
                    x1_k, y1_k = kpts[pt1]
                    x2_k, y2_k = kpts[pt2]
                    if x1_k > 0 and y1_k > 0 and x2_k > 0 and y2_k > 0:
                        cv2.line(frame, (int(x1_k), int(y1_k)), (int(x2_k), int(y2_k)), (0, 255, 255), 2)
            for kpt in kpts:
                xk, yk = kpt
                if xk > 0 and yk > 0:
                    cv2.circle(frame, (int(xk), int(yk)), 4, (0, 165, 255), -1)
    return frame

# =============================================================================
# INFERENCE WORKER - background thread per camera
# =============================================================================

class InferenceWorker:
    """Non-blocking detector: submit frame, retrieve latest detections async."""

    def __init__(self, model, conf_thresh, ocr, face_cascade, plate_model):
        self.model=model; self.conf_thresh=conf_thresh
        self.ocr=ocr; self.face_cascade=face_cascade
        self.plate_model=plate_model
        self._lock=threading.Lock()
        self._input=None; self._output=None; self._seq=0
        self._stop=threading.Event()
        self._thread=threading.Thread(target=self._run,daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock: self._input=frame.copy()

    def latest(self):
        with self._lock:
            if self._output is None: return [],self._seq
            dets,seq=self._output; return list(dets),seq

    def _run(self):
        while not self._stop.is_set():
            with self._lock: frame=self._input; self._input=None
            if frame is None: time.sleep(0.002); continue
            dets=run_inference(self.model,frame,self.conf_thresh,self.ocr,self.face_cascade,self.plate_model)
            with self._lock: self._seq+=1; self._output=(dets,self._seq)

    def stop(self):
        self._stop.set(); self._thread.join(timeout=2.0)

# =============================================================================
# EVENT INTELLIGENCE
# =============================================================================

def _prune_timeline(now):
    tl=st.session_state.timeline
    while tl and now-tl[0][0]>TIMELINE_RETAIN_SEC: tl.popleft()


def register_incident(category, level, text, cam_label, asset, evidence):
    now=time.time(); key=(category,asset)
    inc=st.session_state.incidents; ex=inc.get(key)
    if ex and (now-ex["last_ts"])<=INCIDENT_GROUP_WIN:
        ex["count"]+=1; ex["last_ts"]=now; ex["camera"]=cam_label; ex["evidence"]=evidence
    else:
        st.session_state.incident_seq+=1
        inc[key]={"id":st.session_state.incident_seq,"level":level,"text":text,
                  "camera":cam_label,"asset":asset,"first_ts":now,"last_ts":now,
                  "count":1,"evidence":evidence}
        st.session_state.stats["alerts"]+=1
        if level=="RED": st.session_state.stats["breaches"]+=1


def register_event(cam_id, cam_label, asset_type, conf, quality_tag, sched_breach):
    now=time.time(); key=(cam_id,asset_type)
    if now-st.session_state.last_logged.get(key,0.0)<EVENT_COOLDOWN_SEC: return
    st.session_state.last_logged[key]=now
    st.session_state.timeline.append((now,cam_id,asset_type))
    _prune_timeline(now)
    category,pattern,alert_level=_classify(cam_id,asset_type,now,sched_breach)
    unc=max(0.0,100.0-conf*100.0)
    st.session_state.event_log.insert(0,{
        "Time":datetime.fromtimestamp(now).strftime("%H:%M:%S"),
        "Camera":cam_label,"Asset":asset_type,
        "Confidence":f"{conf*100:.0f}%","Quality":quality_tag,
        "Uncertainty":f"+-{unc:.0f}%","Pattern":pattern,
    })
    st.session_state.event_log=st.session_state.event_log[:500]
    if asset_type=="PERSON":   st.session_state.stats["persons"]+=1
    elif asset_type in VEHICLE_TYPES: st.session_state.stats["vehicles"]+=1
    elif asset_type=="FACE":   st.session_state.stats["faces"]+=1
    if category!="ROUTINE":
        ev_str=f"{asset_type} @ {cam_label} - {conf*100:.0f}% ({quality_tag})"
        register_incident(category,alert_level,pattern,cam_label,asset_type,ev_str)


def _classify(cam_id, asset_type, now, sched_breach):
    tl=st.session_state.timeline
    cams={c for (t,c,a) in tl if a==asset_type and now-t<=RECON_WINDOW_SEC}
    if len(cams)>=RECON_MIN_CAMERAS:
        return ("RECON","[RED] RECONNAISSANCE PATTERN - Multi-Node Scouting Profile","RED")
    if asset_type in VEHICLE_TYPES:
        hits=[t for (t,c,a) in tl if a==asset_type and c==cam_id and now-t<=SHUTTLE_WINDOW_SEC]
        if len(hits)>=2:
            return ("SHUTTLE","[AMBER] SHUTTLE LOOP - Rapid Repeat Pass Detected","AMBER")
    if sched_breach:
        return ("FENCE","[AMBER] SCHEDULE BREACH - Activity Outside Authorized Window","AMBER")
    return ("ROUTINE","[BLUE] ROUTINE PATTERN TRACKED","BLUE")


def current_threat_level():
    now=time.time()
    hot=[i for i in st.session_state.incidents.values() if now-i["last_ts"]<=INCIDENT_GROUP_WIN]
    if any(i["level"]=="RED"   for i in hot): return "RED",   "[RED]   CRITICAL - BREACH / RECON ACTIVE"
    if any(i["level"]=="AMBER" for i in hot): return "AMBER", "[AMBER] ELEVATED - SHUTTLE / FENCE ACTIVITY"
    return "BLUE","[BLUE]  NOMINAL - ROUTINE PERIMETER MONITORING"


def is_sched_breach(enabled, start_t, end_t):
    if not enabled: return False
    now_t=datetime.now().time()
    auth=(start_t<=now_t<=end_t) if start_t<=end_t else (now_t>=start_t or now_t<=end_t)
    return not auth

# =============================================================================
# SESSION STATE
# =============================================================================

def _init():
    defaults={
        "running":False,"event_log":[],"timeline":deque(),
        "incidents":{},"incident_seq":0,"last_seen":{},"last_logged":{},
        "frame_counters":{c[0]:0 for c in CAMERA_POSTS},
        "session_start":None,"render_pass":0,"plates":[],
        "stats":{"persons":0,"vehicles":0,"faces":0,"alerts":0,"breaches":0,"plates":0},
        "engine_state": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.html('<div class="ibvap-title" style="font-size:1.05rem;">IBVAP</div>'
            '<div class="ibvap-sub" style="font-size:0.58rem;">Border Analytics Control</div>')
    st.divider()

    st.markdown("#### QUICK START")
    if _DEMO_VIDEOS:
        if st.button("DEMO MODE - AUTO-LOAD VIDEOS", type="primary", use_container_width=True):
            for i,(cam_id,_) in enumerate(CAMERA_POSTS):
                vid=_DEMO_VIDEOS[i%len(_DEMO_VIDEOS)]
                st.session_state[f"fpath_{cam_id}"]=vid
            st.session_state.running=True
            st.session_state.session_start=time.time()
            st.rerun()
        st.caption(f"Found: {', '.join(_DEMO_VIDEOS[:4])}")
    else:
        st.warning("No MP4 files in folder. Add videos or use webcam below.")

    st.markdown("#### NODE CONFIGURATION")
    cam_inputs={}
    for cam_id,label in CAMERA_POSTS:
        with st.expander(f"{label}  ({cam_id})",expanded=(cam_id=="CAM_1")):
            url=st.text_input("RTSP/HTTP URL",key=f"url_{cam_id}",placeholder="rtsp://192.168.x.x/live")
            fpath=st.text_input("Local video file",key=f"fpath_{cam_id}",placeholder="sample.mp4")
            allow_wc=st.checkbox("Webcam fallback",value=(cam_id=="CAM_1"),key=f"awc_{cam_id}")
            wc_idx=st.number_input("Webcam index",0,9,0 if cam_id=="CAM_1" else 1,step=1,key=f"wci_{cam_id}")
            fence_on=st.checkbox("Virtual fence line",False,key=f"fen_{cam_id}")
            fence_x=st.slider("Fence X (px)",0,640,320,5,key=f"fx_{cam_id}")
            prot_side=st.selectbox("Protected side",["RIGHT","LEFT"],key=f"ps_{cam_id}")
            zone_on=st.checkbox("Protected zone polygon",False,key=f"zon_{cam_id}")
            zl=st.slider("Zone left X",0,640,270,5,key=f"zl_{cam_id}")
            zr=st.slider("Zone right X",0,640,370,5,key=f"zr_{cam_id}")
            if zr<=zl: zr=min(640,zl+20)
            cam_inputs[cam_id]={
                "url":url,"fpath":fpath,"wc_idx":int(wc_idx),"allow_wc":allow_wc,
                "fence_line":(((float(fence_x),0.0),(float(fence_x),360.0)) if fence_on else None),
                "prot_side":prot_side,
                "prot_polygon":([(float(zl),0.),(float(zr),0.),(float(zr),360.),(float(zl),360.)] if zone_on else None),
            }

    st.markdown("#### ENGINE SETTINGS")
    conf_thresh=st.slider("Detection confidence",0.20,0.90,0.45,0.05)
    frame_skip=st.slider("Inference every N frames",1,8,2)
    loop_delay=st.slider("Loop delay (sec)",0.0,0.5,0.05,0.01)

    if st.session_state.engine_state is not None:
        for n in st.session_state.engine_state["nodes"].values():
            n.playback_delay = max(0.015, loop_delay)

    st.markdown("#### SCHEDULE FENCE")
    sched_on=st.checkbox("Enable time-window fence",False)
    fc1,fc2=st.columns(2)
    fence_start=fc1.time_input("Auth from",dtime(6,0),label_visibility="visible")
    fence_end=fc2.time_input("Auth until",dtime(20,0),label_visibility="visible")

    st.markdown("#### ACTIVITY ANALYSIS")
    activity_on=st.checkbox("Enable motion analytics",True)
    show_trails=st.checkbox("Show track IDs and trails",True)
    spd_thresh=st.slider("High-speed threshold (px/s)",50,800,260,10)
    loiter_secs=st.slider("Loiter threshold (sec)",5,120,20,5)
    loiter_rad=st.slider("Loiter radius (px)",10,150,45,5)

    st.divider()
    c1,c2=st.columns(2)
    start_btn=c1.button("START",use_container_width=True,type="primary")
    stop_btn=c2.button("STOP",use_container_width=True)
    clear_btn=st.button("CLEAR LOG + STATS",use_container_width=True)
    st.markdown("#### DETECTED PLATES")
    plates_sb_slot=st.empty()

if start_btn:
    st.session_state.running=True; st.session_state.session_start=time.time()
    if st.session_state.engine_state:
        for w in st.session_state.engine_state["workers"].values(): w.stop()
        for n in st.session_state.engine_state["nodes"].values(): n.release()
    st.session_state.engine_state = None
if stop_btn:
    st.session_state.running=False
if clear_btn:
    st.session_state.update({
        "event_log":[],"timeline":deque(),"incidents":{},"incident_seq":0,
        "last_seen":{},"last_logged":{},"plates":[],
        "stats":{"persons":0,"vehicles":0,"faces":0,"alerts":0,"breaches":0,"plates":0},
    })




# =============================================================================
# RENDER HELPERS
# =============================================================================

def _uptime():
    if not st.session_state.session_start: return "00:00:00"
    s=int(time.time()-st.session_state.session_start)
    h,r=divmod(s,3600); m,ss=divmod(r,60)
    return f"{h:02d}:{m:02d}:{ss:02d}"


@st.fragment(run_every=timedelta(seconds=1.0))
def render_cmd_bar():
    if not st.session_state.running: return
    lvl,_=current_threat_level()
    lc={"RED":"c-red","AMBER":"c-amber","BLUE":"c-blue"}[lvl]
    oc="armed" if st.session_state.running else "standby"
    ot="GRID ARMED" if st.session_state.running else "STANDBY"
    st.html(f"""
    <div class="cmd-bar">
      <div class="cmd-item"><span class="cmd-lbl">System Time</span>
        <span class="cmd-val">{datetime.now().strftime('%H:%M:%S')} &nbsp; {datetime.now().strftime('%d %b %Y').upper()}</span></div>
      <div class="cmd-item"><span class="cmd-lbl">Op Status</span>
        <span class="cmd-val {oc}">{ot}</span></div>
      <div class="cmd-item"><span class="cmd-lbl">Sector</span>
        <span class="cmd-val">NORTH GRID - 3-NODE</span></div>
      <div class="cmd-item"><span class="cmd-lbl">Session Uptime</span>
        <span class="cmd-val">{_uptime()}</span></div>
      <div class="cmd-item"><span class="cmd-lbl">Threat Level</span>
        <span class="cmd-val {lc}">&#x25CF; {lvl}</span></div>
    </div>""")


@st.fragment(run_every=timedelta(seconds=1.0))
def render_stats():
    if not st.session_state.running: return
    s=st.session_state.stats
    st.html(f"""
    <div class="stats-grid">
      <div class="scard">
        <div class="scard-lbl">Persons</div><div class="scard-num">{s['persons']}</div>
        <div class="scard-sub">sightings logged</div></div>
      <div class="scard">
        <div class="scard-lbl">Vehicles</div><div class="scard-num">{s['vehicles']}</div>
        <div class="scard-sub">sightings logged</div></div>
      <div class="scard p">
        <div class="scard-lbl">Faces Detected</div><div class="scard-num">{s['faces']}</div>
        <div class="scard-sub">localizations</div></div>
      <div class="scard a">
        <div class="scard-lbl">Alerts Raised</div><div class="scard-num">{s['alerts']}</div>
        <div class="scard-sub">behavioral incidents</div></div>
      <div class="scard r">
        <div class="scard-lbl">Fence Breaches</div><div class="scard-num">{s['breaches']}</div>
        <div class="scard-sub">confirmed crossings</div></div>
      <div class="scard g">
        <div class="scard-lbl">Plates Read</div><div class="scard-num">{s['plates']}</div>
        <div class="scard-sub">unique plates</div></div>
    </div>""")


def render_cam_header(slot, cam_id, label, live, night=False):
    dot="dot-g" if live else "dot-d"
    st_t="TRACKING" if live else "OFFLINE"
    lb=('<span class="bdg bdg-live">LIVE</span>' if live
        else '<span class="bdg bdg-down">NO SIGNAL</span>')
    nb='<span class="bdg bdg-night">NIGHT</span>' if night else ""
    slot.html(f"""
    <div class="cam-head">
      <div class="cam-name"><span class="dot {dot}"></span>{label} &middot; {cam_id}</div>
      <div class="cam-right">{nb}{lb}<span style="color:var(--dm);font-weight:400;">{st_t}</span></div>
    </div>""")


@st.fragment(run_every=timedelta(seconds=1.0))
def render_active_tracks():
    if not st.session_state.running: return
    now=time.time(); rows=[]
    for asset,info in list(st.session_state.last_seen.items()):
        el=now-info["ts"]
        if el>ACTIVE_TRACK_TTL: continue
        dcf=max(0.,info["conf"]*math.exp(-el/DECAY_TAU_SEC))
        seen=list(dict.fromkeys(c for (t,c,a) in st.session_state.timeline
                                if a==asset and now-t<=RECON_WINDOW_SEC))
        rows.append({"Asset":asset,"Last Node":info["camera"],
                     "Nodes Sighted":", ".join(seen) or info["camera"],
                     "Confidence":f"{dcf*100:.0f}%","Idle (s)":f"{el:.1f}",
                     "State":"BLIND GAP" if el>BLIND_GAP_SEC else "TRACKING"})
    cols=["Asset","Last Node","Nodes Sighted","Confidence","Idle (s)","State"]
    df=pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    st.dataframe(df,width="stretch",height=190,hide_index=True)


@st.fragment(run_every=timedelta(seconds=1.0))
def render_hero():
    if not st.session_state.running: return
    level,headline=current_threat_level()
    cls={"RED":"hero-r","AMBER":"hero-a","BLUE":"hero-b"}[level]
    st.html(f'<div class="hero {cls}">{headline}</div>')


@st.fragment(run_every=timedelta(seconds=1.0))
def render_threats():
    if not st.session_state.running: return
    incs=sorted(st.session_state.incidents.values(),key=lambda i:i["last_ts"],reverse=True)
    if not incs:
        st.html('<div class="t-card t-blue">No behavioral anomalies flagged. System nominal.</div>')
        return
    html='<div style="max-height: 450px; overflow-y: auto; padding-right: 5px;">'
    for i in incs:
        cls="t-red" if i["level"]=="RED" else "t-amber" if i["level"]=="AMBER" else "t-blue"
        ft=datetime.fromtimestamp(i["first_ts"]).strftime("%H:%M:%S")
        lt=datetime.fromtimestamp(i["last_ts"]).strftime("%H:%M:%S")
        html+=(f'<div class="t-card {cls}"><b>#{i["id"]} {i["text"]}</b> x{i["count"]}<br>'
               f'<span style="opacity:.78;font-size:.88em;">{i["camera"]} &middot; {ft}-{lt} &middot; {i["evidence"]}</span></div>')
    html+='</div>'
    st.html(html)


@st.fragment(run_every=timedelta(seconds=1.0))
def render_log():
    if not st.session_state.running: return
    cols=["Time","Camera","Asset","Confidence","Quality","Uncertainty","Pattern"]
    df=(pd.DataFrame(st.session_state.event_log) if st.session_state.event_log
        else pd.DataFrame(columns=cols))
    q=(st.session_state.get("log_search") or "").strip().lower()
    if q and not df.empty:
        mask=df.apply(lambda r:q in " ".join(str(v) for v in r.values).lower(),axis=1)
        df=df[mask]
    st.dataframe(df.head(50),width="stretch",height=310,hide_index=True)
    data=(pd.DataFrame(st.session_state.event_log).to_csv(index=False).encode("utf-8")
          if st.session_state.event_log else b"")
    st.download_button("Export event log (CSV)",data=data,
        file_name="ibvap_events.csv",mime="text/csv",
        disabled=not st.session_state.event_log,width="stretch", key=f"export_event_log_csv_btn_{time.time()}")


@st.fragment(run_every=timedelta(seconds=1.0))
def render_plates_tab():
    if not st.session_state.running: return
    if st.session_state.plates:
        st.dataframe(pd.DataFrame(st.session_state.plates),
                                  width="stretch",hide_index=True)
    else:
        st.info("No license plates detected yet.")


def render_plates_sb():
    if st.session_state.plates:
        plates_sb_slot.dataframe(pd.DataFrame(st.session_state.plates),
                                 width="stretch",hide_index=True,height=120)
    else:
        plates_sb_slot.caption("No plates detected yet.")


def render_all():
    pass 


def render_standby():
    for cam_id,label in CAMERA_POSTS:
        hdr,img,cap=cam_slots[cam_id]
        render_cam_header(hdr,cam_id,label,live=False)
        blank=np.zeros((360,640,3),dtype=np.uint8); blank[:]=(7,11,17)
        cv2.putText(blank,"STANDBY - PRESS START",(50,185),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(38,52,68),2,cv2.LINE_AA)
        img.image(blank,channels="BGR",width="stretch")
        cap.caption("Status: STANDBY")

# =============================================================================
# MAIN LAYOUT - reserve all UI slots
# =============================================================================

st.html(
    '<div class="ibvap-title">IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS</div>'
    '<div class="ibvap-sub">YOLOv8 Edge Inference  |  Face Detection  |  ANPR  |  Virtual Fence  |  Night Adaptation  |  Zero Cloud</div>'
)

if st.session_state.running:
    render_cmd_bar()
    render_stats()

st.html('<div class="sec-hd">Live Checkpoint Feeds</div>')
cam_slots={}
cam_cols=st.columns(3,gap="small")
for (cam_id,label),col in zip(CAMERA_POSTS,cam_cols):
    with col:
        st.html('<div class="cam-card">')
        hdr_slot=st.empty()
        img_slot=st.empty()
        cap_slot=st.empty()
        st.html('</div>')
        cam_slots[cam_id]=(hdr_slot,img_slot,cap_slot)

st.html('<div class="sec-hd" style="margin-top:18px;">Behavioral Intelligence</div>')
intel_col,threat_col=st.columns([3,2],gap="medium")

with intel_col:
    tab_tracks,tab_events,tab_plates=st.tabs(
        ["Active Tracks","Event Log","Plates Detected"]
    )
    with tab_tracks:
        if st.session_state.running: render_active_tracks()
    with tab_events:
        st.text_input(
            "Search log",key="log_search",
            placeholder="camera / asset / pattern / quality",
            label_visibility="collapsed",
        )
        if st.session_state.running: render_log()
    with tab_plates:
        if st.session_state.running: render_plates_tab()

with threat_col:
    st.html('<div class="sec-hd">Threat Level + Incidents</div>')
    if st.session_state.running:
        render_hero()
        render_threats()

st.html('<div class="ibvap-footer">IBVAP Edge Engine  |  YOLOv8n On-Device  |  Human-in-the-Loop Review Required  |  No Cloud Uplink</div>')

# =============================================================================
# IDLE STATE
# =============================================================================

if not st.session_state.running:
    render_standby()
    st.info("Configure checkpoint sources in the sidebar, or click DEMO MODE to auto-load MP4 files, then press START.")
    st.stop()

# =============================================================================
# LIVE OPERATIONAL LOOP
# =============================================================================

if st.session_state.engine_state is None:
    model=load_yolo(); ocr=load_ocr(); fc=load_face_cascade(); pm=load_plate_model()
    
    nodes={cam_id:CameraNode(cam_id,label,cam_inputs[cam_id]["url"],cam_inputs[cam_id]["fpath"],
                              cam_inputs[cam_id]["wc_idx"],cam_inputs[cam_id]["allow_wc"])
           for cam_id,label in CAMERA_POSTS}
    
    engines={cam_id:ActivityEngine(cam_id,
                 fence_line=cam_inputs[cam_id]["fence_line"],
                 protected_side=cam_inputs[cam_id]["prot_side"],
                 protected_polygon=cam_inputs[cam_id]["prot_polygon"],
                 speed_threshold=spd_thresh,loiter_seconds=loiter_secs,
                 loiter_radius=loiter_rad,min_conf=conf_thresh)
             for cam_id,_ in CAMERA_POSTS}
    
    workers={c:InferenceWorker(model,conf_thresh,ocr,fc,pm) for c,_ in CAMERA_POSTS}
    
    st.session_state.engine_state = {
        "nodes": nodes, "engines": engines, "workers": workers,
        "latest_det": {c:[] for c,_ in CAMERA_POSTS},
        "last_seq": {c:0 for c,_ in CAMERA_POSTS}
    }

es = st.session_state.engine_state
nodes, engines, workers = es["nodes"], es["engines"], es["workers"]
latest_det, last_seq = es["latest_det"], es["last_seq"]
MODE_LABELS={"URL":"LIVE STREAM","FILE":"LOCAL FILE (LOOPED)","WEBCAM":"SYSTEM WEBCAM","SYNTH":"NO SIGNAL"}

@st.fragment(run_every=timedelta(seconds=max(loop_delay, 1.0/30.0)))
def live_operational_loop():
    if not st.session_state.running:
        return
    for cam_id,label in CAMERA_POSTS:
        node=nodes[cam_id]; hdr,img,cap=cam_slots[cam_id]
        frame=node.read()
        h,w=frame.shape[:2]
        if w!=640:
            frame=cv2.resize(frame,(640,int(h*640/w)),interpolation=cv2.INTER_AREA)

        st.session_state.frame_counters[cam_id]+=1
        if st.session_state.frame_counters[cam_id]%max(1,frame_skip)==0:
            workers[cam_id].submit(frame)

        dets,seq=workers[cam_id].latest()
        if seq!=last_seq[cam_id]:
            last_seq[cam_id]=seq; latest_det[cam_id]=dets
            gray_mean=float(np.mean(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)))
            is_night=gray_mean<LOW_LIGHT_THRESHOLD
            quality_tag="LOW-LIGHT" if is_night else "NORMAL"
            breach=is_sched_breach(sched_on,fence_start,fence_end)

            if activity_on:
                act_alerts=engines[cam_id].update(dets,now=time.time())
                for al in act_alerts:
                    ts=datetime.fromtimestamp(al["timestamp"]).strftime("%H:%M:%S")
                    pat=f"{al['level']} - {al['kind']} - {al['reason']}"
                    st.session_state.event_log.insert(0,{
                        "Time":ts,"Camera":label,"Asset":f"PERSON #{al['track_id']}",
                        "Confidence":"RULE-BASED","Quality":quality_tag,
                        "Uncertainty":"REVIEW REQUIRED","Pattern":pat,
                    })
                    st.session_state.event_log=st.session_state.event_log[:500]
                    register_incident(al["kind"],al["level"],al["reason"],
                                      label,f"PERSON #{al['track_id']}",str(al["evidence"]))

            now_ts=time.time()
            for d in dets:
                st.session_state.last_seen[d["label"]]={"ts":now_ts,"camera":cam_id,"conf":d["conf"]}
                register_event(cam_id,label,d["label"],d["conf"],quality_tag,breach)
                if d.get("plate") and d["plate"] != "NOT VISIBLE":
                    if not any(p["Plate"]==d["plate"] for p in st.session_state.plates):
                        st.session_state.plates.insert(0,{"Plate":d["plate"],"Camera":cam_id,
                                                           "Time":datetime.now().strftime("%H:%M:%S")})
                        st.session_state.stats["plates"]+=1
                    st.session_state.plates=st.session_state.plates[:20]

        annotated=frame.copy()
        skip={"PERSON"} if (activity_on and show_trails) else set()
        draw_detections(annotated,latest_det[cam_id],skip_labels=skip)
        if activity_on and show_trails:
            annotated=engines[cam_id].draw(annotated,now=time.time())

        gm=float(np.mean(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)))
        night_disp=gm<LOW_LIGHT_THRESHOLD
        if night_disp:
            cv2.putText(annotated,"NIGHT MODE",(8,annotated.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,90,255),1,cv2.LINE_AA)
        ts_str=datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated,f"{cam_id} - {ts_str}",(8,annotated.shape[0]-28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(90,160,200),1,cv2.LINE_AA)

        render_cam_header(hdr,cam_id,label,live=(node.mode!="SYNTH"),night=night_disp)
        img.image(annotated,channels="BGR",width="stretch")
        cap.caption(f"{MODE_LABELS[node.mode]} - {node.source_desc}")

    st.session_state.render_pass+=1
    # We no longer call render_all() here, UI updates automatically via fragments!

if st.session_state.running:
    live_operational_loop()
elif st.session_state.engine_state is not None:
    for w in workers.values(): w.stop()
    for n in nodes.values():   n.release()
    st.session_state.engine_state = None

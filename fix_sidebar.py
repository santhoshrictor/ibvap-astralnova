import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = '''    st.markdown("#### NODE CONFIGURATION")
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
    stop_btn=c2.button("STOP",use_container_width=True)'''

pattern = r'    st\.markdown\("#### NODE CONFIGURATION"\)\n    cam_inputs=\{\}\n    for cam_id,label in CAMERA_POSTS:\n    stop_btn=c2\.button\("STOP",use_container_width=True\)'

if re.search(pattern, content):
    content = re.sub(pattern, replacement, content)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS')
else:
    print('PATTERN NOT FOUND')

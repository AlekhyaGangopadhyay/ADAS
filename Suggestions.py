"""
=============================================================
  Autonomous Car Simulation  v11 — Visual Edition
  ─────────────────────────────────────────────────────────
  Full visual overhaul:
  · Night highway with animated city skyline + stars
  · Neon road markings, wet-road reflection strips
  · Per-car headlight cone + taillight glow
  · Engine spark / exhaust particles on boost
  · Speed-streak motion blur on fast cars
  · Animated HUD panels with glowing borders
  · Lane cost displayed as colour-washed lane tint
  · Smooth speedometer with RGB needle
  · Minimap with radar sweep animation
  · All 6 AI features from v10 retained
=============================================================
  Requirements:  pip install pygame numpy
  Controls:  S=spline  G=ghosts  R=reset  ESC=quit
=============================================================
"""

import sys, math, random, colorsys
import numpy as np
import pygame

# ── Screen ────────────────────────────────────────────────
W, H       = 1400, 760
FPS        = 60
NUM_LANES  = 3
LANE_H     = 148
ROAD_TOP   = H - NUM_LANES * LANE_H - 60
ROAD_BOT   = ROAD_TOP + NUM_LANES * LANE_H

# ── Palette ───────────────────────────────────────────────
C_SKY_TOP    = ( 4,   6,  18)
C_SKY_MID    = ( 8,  12,  32)
C_HORIZON    = (18,  28,  55)
C_ROAD       = (22,  24,  30)
C_ROAD_ALT   = (19,  21,  27)
C_ROAD_EDGE  = (255, 200,   0)
C_SHOULDER   = (28,  24,  18)
C_WHITE      = (255, 255, 255)
C_DASH_WARM  = (230, 230, 210)

NEON_BLUE    = ( 40, 140, 255)
NEON_CYAN    = ( 20, 230, 230)
NEON_PINK    = (255,  50, 160)
NEON_GREEN   = ( 50, 255, 120)
NEON_ORANGE  = (255, 140,  20)
NEON_PURPLE  = (180,  60, 255)

# Car body palettes: (body, dark, accent/neon)
PALETTES = {
    "ego":     ((15,  80, 220), ( 8,  40, 130), NEON_BLUE),
    "red":     ((180,  30,  40), (90,  10,  15), (255, 90, 100)),
    "lime":    (( 20, 160,  60), (10,  85,  28), NEON_GREEN),
    "gold":    ((200, 145,   0), (110, 75,   0), (255, 215, 60)),
    "purple":  ((130,  40, 200), (65,  15, 105), NEON_PURPLE),
    "teal":    (( 15, 155, 165), ( 8,  80,  85), NEON_CYAN),
    "orange":  ((210,  90,  15), (115, 48,   5), NEON_ORANGE),
    "grey":    ((100, 105, 115), (52,  55,  62), (190, 195, 205)),
    "hazard":  ((195, 170,   0), (100, 88,   0), (255, 240,  80)),
}

NPC_TYPE_PALETTE = {
    "normal":     ["red","lime","gold","purple","teal","orange"],
    "stopped":    ["hazard"],
    "aggressive": ["red","orange"],
    "truck":      ["grey"],
}
NPC_TYPES_POOL = ["normal","normal","normal","stopped","aggressive","truck"]
NUM_NPCS       = 6

# ── Physics ───────────────────────────────────────────────
EGO_SPEED        = 4.5
EGO_OVERTAKE_SPD = 8.5
NPC_SPEEDS = {
    "normal":     (1.8, 3.0),
    "stopped":    (0.0, 0.0),
    "aggressive": (5.5, 7.0),
    "truck":      (1.0, 2.0),
}
SPAWN_AHEAD_MIN = 750
SPAWN_SPACING   = 540
RESPAWN_AHEAD   = 1050

# ── Threat distances ──────────────────────────────────────
WARN_GAP      = 310;  SAFE_GAP     = 165; EMERGENCY_GAP = 75
OVERTAKE_GAP  = 275;  REAR_WARN_GAP= 330
TTC_REAR_WARN = 80;   TTC_REAR_EVADE=55;  TTC_REAR_EMERG=35
LC_REAR_MIN   = 295;  LC_FRONT_MIN = 215
TTC_LC_PAUSE  = 55;   TTC_LC_SLOW  = 85
SIDE_WARN_X   = 148;  SIDE_CRIT_X  = 105
NEAR_MISS_THRESHOLD  = EMERGENCY_GAP * 1.8
NEAR_MISS_MARGIN_MAX = 95
MARGIN_DECAY_RATE    = 0.002
POS_NOISE_STD = 7.5;  SPEED_NOISE_STD = 0.28
KALMAN_HORIZON = 65
W_FRONT_GAP   = 0.50; W_REAR_TTC=2.20; W_OCCUPANCY=18.0; W_HOP=55.0

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def lane_y(i):
    return ROAD_TOP + i * LANE_H + LANE_H // 2

def ttc(gap, es, os):
    c = os - es
    return (gap / c) if c > 0.01 else 9999

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_col(a, b, t):
    return tuple(max(0,min(255,int(a[i]+(b[i]-a[i])*t))) for i in range(3))

def hsv_col(h, s, v):
    r,g,b = colorsys.hsv_to_rgb(h,s,v)
    return (int(r*255),int(g*255),int(b*255))

def catmull_rom(p0,p1,p2,p3,n=60):
    p0,p1,p2,p3 = map(np.asarray,[p0,p1,p2,p3])
    pts=[]
    for i in range(n):
        t=i/(n-1); t2=t*t; t3=t2*t
        pt=0.5*(2*p1+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3)
        pts.append(pt.tolist())
    return pts

def build_spline(wps,n=55):
    wp=[wps[0]]+list(wps)+[wps[-1]]; path=[]
    for i in range(1,len(wp)-2):
        seg=catmull_rom(wp[i-1],wp[i],wp[i+1],wp[i+2],n)
        if i>1: seg=seg[1:]
        path.extend(seg)
    return path

# ─────────────────────────────────────────────────────────
#  PARTICLE SYSTEM
# ─────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, vx, vy, life, col, radius=3):
        self.x=float(x); self.y=float(y)
        self.vx=vx; self.vy=vy
        self.life=life; self.max_life=life
        self.col=col; self.radius=radius

    def update(self):
        self.x+=self.vx; self.y+=self.vy
        self.vy+=0.06        # gravity
        self.vx*=0.96
        self.life-=1

    def draw(self, surf, cam):
        if self.life<=0: return
        t=self.life/self.max_life
        a=int(t*200)
        r=max(1,int(self.radius*t))
        sx=int(self.x-cam)
        if -10<sx<W+10:
            s=pygame.Surface((r*2+2,r*2+2),pygame.SRCALPHA)
            pygame.draw.circle(s,(*self.col,a),(r+1,r+1),r)
            surf.blit(s,(sx-r-1,int(self.y)-r-1))


# ─────────────────────────────────────────────────────────
#  KALMAN TRACKER
# ─────────────────────────────────────────────────────────
class KalmanTracker:
    def __init__(self,x0,v0=0.0):
        self.state=np.array([float(x0),float(v0),0.0])
        self.P=np.diag([60.0,6.0,1.5])
        self.Q=np.diag([0.08,0.25,0.55])
        self.R=np.array([[POS_NOISE_STD**2]])
        self.H=np.array([[1.0,0.0,0.0]])
        self.F=np.array([[1.0,1.0,0.5],[0.0,1.0,1.0],[0.0,0.0,1.0]])

    def step(self,z):
        self.state=self.F@self.state
        self.P=self.F@self.P@self.F.T+self.Q
        y=z-self.H@self.state
        S=float((self.H@self.P@self.H.T)[0,0]+self.R[0,0])
        K=(self.P@self.H.T)/S
        self.state=self.state+K.ravel()*float(y.item())
        self.P=(np.eye(3)-np.outer(K.ravel(),self.H))@self.P

    @property
    def est_x(self): return float(self.state[0])
    @property
    def est_speed(self): return float(max(self.state[1],0.0))
    def predict_x(self,n):
        x,v,a=self.state
        return float(x+v*n+0.5*a*n*n)


class Obs:
    def __init__(self,car,ex,es,px):
        self.car=car; self.x=ex; self.speed=es; self.pred_x=px
        self.lane=car.lane; self.y=car.y; self.npc_type=car.npc_type


# ─────────────────────────────────────────────────────────
#  CAR
# ─────────────────────────────────────────────────────────
class Car:
    def __init__(self,x,y,lane,pal_key,speed=EGO_SPEED,is_ego=False,npc_type="normal"):
        self.x=float(x); self.y=float(y)
        self.lane=lane; self.pal=PALETTES[pal_key]
        self.speed=float(speed); self.is_ego=is_ego; self.npc_type=npc_type
        self.state="cruise"; self.target_lane=lane
        self.path=[]; self.path_idx=0
        self.on_spline=False; self.spline_pause=False
        self.blinker=0; self.blink_t=0; self.overtake_target=None
        self.hazard_t=0; self.prev_x=float(x)
        self.L,self.CW=(108,52) if npc_type=="truck" else (74,42)

    def draw(self, surf, cam):
        sx=int(self.x-cam); sy=int(self.y)
        L,CW=self.L,self.CW
        body,dark,accent=self.pal

        # ── Speed streaks (motion blur) ──────────────────
        spd_vis=abs(self.x-self.prev_x)
        if spd_vis>3.5:
            for i in range(1,5):
                alpha=int(55*(1-i/5))
                offset=int(spd_vis*i*1.8)
                streak=pygame.Surface((offset+2,CW-8),pygame.SRCALPHA)
                streak.fill((*body,alpha))
                surf.blit(streak,(sx-L//2-offset,sy-CW//2+4))

        # ── Headlight cone ───────────────────────────────
        cone_len=210 if self.is_ego else 140
        cone=pygame.Surface((cone_len,CW+60),pygame.SRCALPHA)
        cx0,cy0=0,(CW+60)//2
        pts=[(cx0,cy0-4),(cx0,cy0+4),(cone_len,(CW+60)//2+32),(cone_len,(CW+60)//2-32)]
        pygame.draw.polygon(cone,(255,255,200,18),pts)
        surf.blit(cone,(sx+L//2,(sy-(CW+60)//2)))

        # ── Taillight glow ───────────────────────────────
        tg_r = 28 if self.npc_type=="stopped" else 18
        tg=pygame.Surface((tg_r*2,tg_r*2),pygame.SRCALPHA)
        pygame.draw.circle(tg,(200,20,20,55),(tg_r,tg_r),tg_r)
        surf.blit(tg,(sx-L//2-tg_r,sy-tg_r))

        # ── Shadow ───────────────────────────────────────
        sh=pygame.Surface((L+8,CW+8),pygame.SRCALPHA)
        pygame.draw.rect(sh,(0,0,0,50),sh.get_rect().inflate(-2,-2),border_radius=7)
        surf.blit(sh,(sx-L//2-2,sy-CW//2+8))

        # ── Road reflection strip ─────────────────────────
        ref=pygame.Surface((L,8),pygame.SRCALPHA)
        ref.fill((*body,30))
        surf.blit(ref,(sx-L//2,sy+CW//2+2))

        # ── Body ─────────────────────────────────────────
        br=pygame.Rect(sx-L//2,sy-CW//2,L,CW)
        pygame.draw.rect(surf,body,br,border_radius=10)

        # Accent stripe along top edge
        pygame.draw.rect(surf,accent,(sx-L//2+8,sy-CW//2,L-16,3),border_radius=2)

        # Body outline
        pygame.draw.rect(surf,dark,br,2,border_radius=10)

        # Windshield
        ws=pygame.Rect(sx+L//2-50,sy-CW//2+7,26,CW-14)
        wss=pygame.Surface((ws.w,ws.h),pygame.SRCALPHA)
        wss.fill((160,220,255,145)); surf.blit(wss,(ws.x,ws.y))
        pygame.draw.rect(surf,dark,ws,1,border_radius=4)

        # Rear window
        rw=pygame.Rect(sx-L//2+5,sy-CW//2+7,18,CW-14)
        rws=pygame.Surface((rw.w,rw.h),pygame.SRCALPHA)
        rws.fill((120,170,220,115)); surf.blit(rws,(rw.x,rw.y))

        # Roof
        rf=pygame.Rect(sx-L//2+22,sy-CW//2+8,L-46,CW-16)
        rfs=pygame.Surface((rf.w,rf.h),pygame.SRCALPHA)
        rfs.fill((*dark,130)); surf.blit(rfs,(rf.x,rf.y))

        # Wheels
        for wx,wy in [(sx-L//2+7,sy-CW//2-9),(sx+L//2-23,sy-CW//2-9),
                      (sx-L//2+7,sy+CW//2+1),(sx+L//2-23,sy+CW//2+1)]:
            pygame.draw.rect(surf,(15,15,18),(wx,wy,16,8),border_radius=3)
            pygame.draw.rect(surf,(65,68,75),(wx+3,wy+2,10,4),border_radius=2)
            # Hubcap glint
            pygame.draw.circle(surf,(100,105,115),(wx+8,wy+4),3)

        # Headlights (neon glow)
        for hy in (sy-CW//2+7,sy+CW//2-7):
            pygame.draw.ellipse(surf,(255,255,200),(sx+L//2-8,hy-4,12,8))
            # outer glow
            gl=pygame.Surface((22,18),pygame.SRCALPHA)
            pygame.draw.ellipse(gl,(255,255,160,60),(0,0,22,18))
            surf.blit(gl,(sx+L//2-5,hy-9))

        # Taillights
        for hy in (sy-CW//2+7,sy+CW//2-7):
            pygame.draw.ellipse(surf,(220,30,40),(sx-L//2+2,hy-4,12,8))

        # Blinker
        self.blink_t=(self.blink_t+1)%36
        if self.blinker!=0 and self.blink_t<18:
            by=(sy-CW//2-7) if self.blinker==-1 else (sy+CW//2-1)
            pygame.draw.rect(surf,(255,165,0),(sx+L//2-32,by,26,5),border_radius=2)
            gl=pygame.Surface((30,10),pygame.SRCALPHA)
            gl.fill((255,165,0,80)); surf.blit(gl,(sx+L//2-33,by-2))

        # Stopped: hazard flash
        if self.npc_type=="stopped":
            self.hazard_t=(self.hazard_t+1)%44
            if self.hazard_t<22:
                hc=(255,150,0)
                for hx2,hy2 in [(sx-L//2-4,sy-CW//2-8),(sx-L//2-4,sy+CW//2),
                                 (sx+L//2-10,sy-CW//2-8),(sx+L//2-10,sy+CW//2)]:
                    pygame.draw.rect(surf,hc,(hx2,hy2,14,6),border_radius=2)
                    hg=pygame.Surface((20,12),pygame.SRCALPHA)
                    hg.fill((255,150,0,55)); surf.blit(hg,(hx2-3,hy2-3))

        # Truck extra detail
        if self.npc_type=="truck":
            pygame.draw.rect(surf,(dark[0]//2,dark[1]//2,dark[2]//2),
                (sx-L//2+5,sy-4,L-10,8),border_radius=2)
            pygame.draw.rect(surf,accent,(sx-L//2+5,sy-4,L-10,2),border_radius=1)

        # Aggressive: pulsing red glow
        if self.npc_type=="aggressive":
            pulse=abs(math.sin(pygame.time.get_ticks()*0.008))
            g=pygame.Surface((L+24,CW+24),pygame.SRCALPHA)
            pygame.draw.rect(g,(255,20,20,int(30*pulse)),g.get_rect(),border_radius=12)
            surf.blit(g,(sx-L//2-12,sy-CW//2-12))

        # Ego blue glow
        if self.is_ego:
            t_pulse=0.5+0.5*math.sin(pygame.time.get_ticks()*0.004)
            al=int(20+12*t_pulse)
            g=pygame.Surface((L+28,CW+28),pygame.SRCALPHA)
            pygame.draw.rect(g,(*NEON_BLUE,al),g.get_rect(),border_radius=15)
            surf.blit(g,(sx-L//2-14,sy-CW//2-14))
            # Neon underline
            ul=pygame.Surface((L,4),pygame.SRCALPHA)
            ul.fill((*NEON_BLUE,int(80*t_pulse)))
            surf.blit(ul,(sx-L//2,sy+CW//2+4))

            if self.state=="passing":
                g2=pygame.Surface((L+40,CW+40),pygame.SRCALPHA)
                pygame.draw.rect(g2,(*NEON_ORANGE,int(40*t_pulse)),g2.get_rect(),border_radius=18)
                surf.blit(g2,(sx-L//2-20,sy-CW//2-20))

    def step_spline(self):
        if not self.path or self.path_idx>=len(self.path):
            self.on_spline=False; self.spline_pause=False
            self.lane=self.target_lane; self.path=[]; self.path_idx=0; self.blinker=0
            return True
        if self.spline_pause: return False
        pt=self.path[self.path_idx]
        self.prev_x=self.x; self.x=pt[0]; self.y=pt[1]; self.path_idx+=1
        return False


# ─────────────────────────────────────────────────────────
#  BACKGROUND SCENERY
# ─────────────────────────────────────────────────────────
class Scenery:
    def __init__(self):
        rng=random.Random(7)
        # Stars
        self.stars=[(rng.randint(0,W),rng.randint(0,ROAD_TOP-40),
                     rng.uniform(0.5,2.0),rng.uniform(0,2*math.pi))
                    for _ in range(220)]
        # City buildings (background layer)
        self.buildings=[]
        x=0
        while x<W*3:
            w2=rng.randint(22,60); h2=rng.randint(30,110)
            col=rng.randint(14,30)
            windows=[(rng.randint(0,w2-6),rng.randint(0,h2-8))
                     for _ in range(rng.randint(3,12))]
            self.buildings.append((x,w2,h2,col,windows))
            x+=w2+rng.randint(2,12)
        # Moving road reflections (puddles)
        self.puddles=[(rng.uniform(0,3000), rng.randint(0,NUM_LANES-1),
                       rng.uniform(30,80))
                      for _ in range(18)]

    def draw(self, surf, cam, frame):
        # Sky gradient
        for y in range(ROAD_TOP+30):
            t=y/(ROAD_TOP+30)
            col=lerp_col(C_SKY_TOP,C_HORIZON,t**(0.6))
            pygame.draw.line(surf,col,(0,y),(W,y))

        # Stars (twinkle)
        for (sx2,sy2,sz,phase) in self.stars:
            brightness=0.5+0.5*math.sin(frame*0.04+phase)
            a=int(brightness*200)
            r=max(1,int(sz))
            s=pygame.Surface((r*2+2,r*2+2),pygame.SRCALPHA)
            pygame.draw.circle(s,(220,230,255,a),(r+1,r+1),r)
            surf.blit(s,(sx2-r-1,sy2-r-1))

        # City silhouette (parallax scroll)
        base=ROAD_TOP-2
        off=int(cam*0.12)%max(1,int(sum(b[1]+4 for b in self.buildings[:40])))
        for (bx,bw,bh,col,wins) in self.buildings:
            rx=(bx-off)%4000-400
            if rx>W+80: continue
            # Building body
            bc=(col,col+4,col+14)
            pygame.draw.rect(surf,bc,(rx,base-bh,bw,bh))
            pygame.draw.rect(surf,(col+8,col+12,col+22),(rx,base-bh,bw,3))
            # Windows
            for (wx2,wy2) in wins:
                w_on=random.Random(bx*100+wx2+frame//45).random()>0.3
                wc=(210,230,180) if w_on else (col,col,col+10)
                pygame.draw.rect(surf,wc,(rx+wx2,base-bh+wy2+4,5,4))

        # Horizon glow
        hg=pygame.Surface((W,28),pygame.SRCALPHA)
        for dy in range(28):
            a=int((1-dy/28)*45)
            pygame.draw.line(hg,(80,130,220,a),(0,dy),(W,dy))
        surf.blit(hg,(0,ROAD_TOP-28))

    def draw_road_details(self, surf, cam, frame):
        # Shoulder
        pygame.draw.rect(surf,C_SHOULDER,(0,ROAD_TOP-26,W,26))
        pygame.draw.rect(surf,C_SHOULDER,(0,ROAD_BOT,W,34))

        # Lane fills
        for i in range(NUM_LANES):
            col=C_ROAD if i%2==0 else C_ROAD_ALT
            pygame.draw.rect(surf,col,(0,ROAD_TOP+i*LANE_H,W,LANE_H))

        # Wet road shimmer at bottom of each lane
        for i in range(NUM_LANES):
            shim_y=ROAD_TOP+i*LANE_H+LANE_H-18
            shim=pygame.Surface((W,18),pygame.SRCALPHA)
            for dy in range(18):
                a=int((1-dy/18)*22)
                pygame.draw.line(shim,(60,80,140,a),(0,dy),(W,dy))
            surf.blit(shim,(0,shim_y))

        # Moving road grit dots
        rng=random.Random(42)
        off=int(cam*0.5)
        for _ in range(320):
            tx=rng.randint(0,3000); ty=rng.randint(ROAD_TOP,ROAD_BOT)
            px=int((tx-off)%W)
            pygame.draw.circle(surf,(45,47,52),(px,ty),1)

        # Puddle reflections (feature)
        for (px2,pl,pw) in self.puddles:
            rx=int((px2-cam*0.7)%W)
            py=lane_y(pl)+LANE_H//2-10
            wobble=math.sin(frame*0.08+px2)*3
            pu=pygame.Surface((int(pw),8),pygame.SRCALPHA)
            for dy in range(8):
                a=int((1-dy/8)*35)
                pygame.draw.line(pu,(80,140,200,a),(0,dy),(int(pw),dy))
            surf.blit(pu,(rx,int(py+wobble)))

        # Edge stripes (neon yellow)
        pygame.draw.rect(surf,C_ROAD_EDGE,(0,ROAD_TOP-5,W,5))
        pygame.draw.rect(surf,C_ROAD_EDGE,(0,ROAD_BOT,W,5))
        # Subtle glow on edges
        eg=pygame.Surface((W,14),pygame.SRCALPHA)
        eg.fill((255,200,0,18))
        surf.blit(eg,(0,ROAD_TOP-16))
        surf.blit(eg,(0,ROAD_BOT))

        # Lane dash lines (animated scroll)
        dash,gap2=60,42; period=dash+gap2
        off2=int(-cam*1.0)%period
        for div in range(1,NUM_LANES):
            y=ROAD_TOP+div*LANE_H
            x=-period+off2
            while x<W+period:
                x0=max(int(x),0); x1=min(int(x+dash),W)
                if x1>x0:
                    pygame.draw.line(surf,C_DASH_WARM,(x0,y),(x1,y),2)
                    # soft glow under dash
                    gl=pygame.Surface((x1-x0,6),pygame.SRCALPHA)
                    gl.fill((220,220,190,25)); surf.blit(gl,(x0,y))
                x+=period

        # Lane labels
        font_tiny=pygame.font.SysFont("Consolas",13)
        for i in range(NUM_LANES):
            lbl=font_tiny.render(f"L{i+1}",True,(65,68,75))
            surf.blit(lbl,(10,lane_y(i)-8))


# ─────────────────────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────────────────────
class Sim:

    def __init__(self):
        pygame.init()
        self.screen=pygame.display.set_mode((W,H))
        pygame.display.set_caption("Autonomous v11  ·  Night Highway")
        self.clock=pygame.time.Clock()
        self.font_s=pygame.font.SysFont("Consolas",14)
        self.font_m=pygame.font.SysFont("Consolas",18,bold=True)
        self.font_xs=pygame.font.SysFont("Consolas",12)
        self.font_l=pygame.font.SysFont("Consolas",22,bold=True)
        self.scenery=Scenery()
        self.particles=[]
        self._reset()

    def _reset(self):
        el=NUM_LANES//2
        self.ego=Car(265,lane_y(el),el,"ego",EGO_SPEED,is_ego=True)
        self.npcs=[]; self.trackers={}
        x_cursor=self.ego.x+SPAWN_AHEAD_MIN
        types_s=NPC_TYPES_POOL[:]; random.shuffle(types_s)
        for i in range(NUM_NPCS):
            t=types_s[i%len(types_s)]; lane=i%NUM_LANES
            pk=random.choice(NPC_TYPE_PALETTE[t])
            sr=NPC_SPEEDS[t]; spd=random.uniform(*sr)
            xs=x_cursor+random.randint(0,240)
            npc=Car(xs,lane_y(lane),lane,pk,spd,npc_type=t)
            self.npcs.append(npc)
            self.trackers[id(npc)]=KalmanTracker(xs,spd)
            x_cursor=xs+SPAWN_SPACING+random.randint(0,320)

        self.cam=0.0; self.road_off=0
        self.show_sp=True; self.show_ghosts=True
        self.sp_preview=[]; self.frame=0
        self.n_over=0; self.n_evade=0; self.n_multihop=0
        self.status="Cruising"; self.trapped=False; self.queued_lane=None
        self.near_miss_count=0; self.adaptive_margin=0.0
        self.last_nm_frame=0; self.frame_count=0
        self.thr_front=False; self.thr_rear=False
        self.thr_left=False; self.thr_right=False; self.thr_merge=False
        self.rear_ttc=9999; self.lc_rear_ttc=9999
        self.lane_costs=[0.0]*NUM_LANES; self.obs_cache=[]
        self.alpha={d:0 for d in ("front","rear","left","right","merge")}
        self.radar_angle=0.0
        self.particles=[]

    def _respawn_npc(self,npc):
        lc={l:0 for l in range(NUM_LANES)}
        for c in self.npcs:
            if c is not npc: lc[c.lane]=lc.get(c.lane,0)+1
        lane=min(lc,key=lc.get)
        fx=max((c.x for c in self.npcs if c is not npc),default=self.ego.x)
        xs=max(fx,self.ego.x+RESPAWN_AHEAD)+random.randint(260,580)
        t=npc.npc_type; sr=NPC_SPEEDS[t]; spd=random.uniform(*sr)
        npc.lane=lane; npc.target_lane=lane; npc.y=lane_y(lane)
        npc.x=float(xs); npc.speed=spd; npc.on_spline=False; npc.path=[]
        self.trackers[id(npc)]=KalmanTracker(xs,spd)

    # ── Sensors + Kalman ─────────────────────────────────
    def _update_sensors(self):
        obs=[]
        for npc in self.npcs:
            nz=npc.x+np.random.normal(0,POS_NOISE_STD)
            tr=self.trackers[id(npc)]; tr.step(nz)
            obs.append(Obs(npc,tr.est_x,tr.est_speed,tr.predict_x(KALMAN_HORIZON)))
        self.obs_cache=obs; return obs

    # ── Lane costs ────────────────────────────────────────
    def _score_lane(self,lane,observations):
        ego=self.ego; ex=ego.x; es=ego.speed; cost=0.0
        cost+=abs(lane-ego.lane)*W_HOP
        in_l=[o for o in observations if o.lane==lane]
        cost+=len(in_l)*W_OCCUPANCY
        front=[o for o in in_l if o.x>ex]
        if front:
            near=min(front,key=lambda o:o.x)
            pg=near.pred_x-(ex+es*KALMAN_HORIZON)
            cost+=(800 if pg<0 else max(0,450-pg)*W_FRONT_GAP)
        else:
            cost-=120
        rear=[o for o in in_l if o.x<=ex]
        if rear:
            near=max(rear,key=lambda o:o.x); gap=ex-near.x
            t=ttc(gap,es,near.speed)
            cost+=(600 if t<30 else 280 if t<55 else 90 if t<85 else 0)
        return cost

    def _ranked_lanes(self,obs):
        scores=[(l,self._score_lane(l,obs)) for l in range(NUM_LANES)]
        scores.sort(key=lambda x:x[1])
        self.lane_costs=[s for _,s in sorted(scores,key=lambda x:x[0])]
        return scores

    def _find_multihop(self,obs):
        ego=self.ego; el=ego.lane
        for final in range(NUM_LANES):
            if final==el: continue
            mid=(el+final)//2
            if mid==el: mid=final
            if mid==el or mid==final: continue
            mf=[o for o in obs if o.lane==mid and o.x>ego.x and o.x-ego.x<LC_FRONT_MIN*1.5]
            if mf: continue
            if self._lane_clear_k(final,obs): return (mid,final)
        return None

    def _scan_threats(self,obs):
        ego=self.ego; ex=ego.x; am=self.adaptive_margin

        def in_l(o,l): return abs(o.y-lane_y(l))<LANE_H*0.55
        def x_ov(o,e=0): return abs(o.x-ex)<(ego.L+e)

        same=      [o for o in obs if in_l(o,ego.lane)]
        ahead=sorted([o for o in same if o.x>ex],  key=lambda o: o.x)
        behind=sorted([o for o in same if o.x<=ex],key=lambda o:-o.x)

        leader=ahead[0]  if ahead  else None
        chaser=behind[0] if behind else None
        fg=(leader.x-ex) if leader else 9999
        rg=(ex-chaser.x) if chaser else 9999

        self.thr_front=leader is not None and fg<WARN_GAP+am

        if chaser:
            self.rear_ttc=ttc(rg,ego.speed,chaser.speed)
            self.thr_rear=rg<REAR_WARN_GAP or self.rear_ttc<TTC_REAR_WARN
        else:
            self.rear_ttc=9999; self.thr_rear=False

        ll=ego.lane-1; rl=ego.lane+1
        if ll>=0:
            lc=[o for o in obs if in_l(o,ll) and x_ov(o,SIDE_WARN_X)]
            self.thr_left=len(lc)>0
        else:
            self.thr_left=False; lc=[]
        if rl<NUM_LANES:
            rc=[o for o in obs if in_l(o,rl) and x_ov(o,SIDE_WARN_X)]
            self.thr_right=len(rc)>0
        else:
            self.thr_right=False; rc=[]

        if ego.on_spline:
            tl=ego.target_lane
            tlr=[o for o in obs if in_l(o,tl) and o.x<=ex]
            self.thr_merge=False; self.lc_rear_ttc=9999
            for o in tlr:
                gr=ex-o.x; t2=ttc(gr,ego.speed,o.speed)
                if t2<self.lc_rear_ttc: self.lc_rear_ttc=t2
                if gr<LC_REAR_MIN+am or t2<TTC_LC_SLOW: self.thr_merge=True
            if (tl>ego.lane and self.thr_right) or (tl<ego.lane and self.thr_left):
                self.thr_merge=True
        else:
            self.thr_merge=False; self.lc_rear_ttc=9999

        if fg<NEAR_MISS_THRESHOLD+am:
            self.near_miss_count+=1
            self.adaptive_margin=min(self.near_miss_count*9.0,float(NEAR_MISS_MARGIN_MAX))
            self.last_nm_frame=self.frame_count

        return {"leader":leader,"chaser":chaser,"front_gap":fg,"rear_gap":rg,"rear_ttc":self.rear_ttc}

    def _lane_clear_k(self,tl,obs):
        ego=self.ego; ex=ego.x; am=self.adaptive_margin
        for o in obs:
            if abs(o.y-lane_y(tl))>LANE_H*0.55: continue
            dx=o.x-ex
            if -(LC_REAR_MIN+am)<=dx<=0:
                rt=ttc(-dx,ego.speed,o.speed)
                if dx>-(LC_REAR_MIN*0.8+am) or rt<TTC_LC_PAUSE+12: return False
            elif 0<dx<LC_FRONT_MIN+am: return False
        return True

    def _plan_change(self,to_lane,run=345):
        ego=self.ego; x0,y0=ego.x,ego.y
        y1=lane_y(to_lane); dy=y1-y0
        wps=[[x0,y0],[x0+run*0.14,y0+dy*0.03],[x0+run*0.42,y0+dy*0.48],
             [x0+run*0.80,y0+dy*0.97],[x0+run,y1],[x0+run+85,y1]]
        path=build_spline(wps)
        ego.path=path; ego.path_idx=0; ego.on_spline=True
        ego.spline_pause=False; ego.target_lane=to_lane
        ego.blinker=1 if to_lane>ego.lane else -1
        self.sp_preview=[(p[0],p[1]) for p in path]

    def _spline_speed_ctrl(self):
        ego=self.ego
        if self.thr_merge and self.lc_rear_ttc<TTC_LC_PAUSE:
            ego.spline_pause=True; ego.speed=max(ego.speed-0.18,1.0)
            self.status=f"⚠ LC PAUSED  {self.lc_rear_ttc:.0f}f"; return
        ego.spline_pause=False
        if self.thr_merge and self.lc_rear_ttc<TTC_LC_SLOW:
            sl=(TTC_LC_SLOW-self.lc_rear_ttc)/TTC_LC_SLOW
            ego.speed=max(ego.speed-sl*0.45,1.2)
            self.status=f"↕ LC Slow  {self.lc_rear_ttc:.0f}f"

    def _regulate_speed(self,threats):
        ego=self.ego; leader=threats["leader"]; chaser=threats["chaser"]
        fg=threats["front_gap"]; rt=threats["rear_ttc"]; am=self.adaptive_margin
        desired=EGO_OVERTAKE_SPD if ego.state=="passing" else EGO_SPEED
        if leader:
            if fg<EMERGENCY_GAP+am:
                desired=min(desired,leader.speed-1.2)
                if fg<ego.L*0.75: desired=0.0
            elif fg<SAFE_GAP+am:
                t=(fg-(EMERGENCY_GAP+am))/max(SAFE_GAP-EMERGENCY_GAP,1)
                desired=min(desired,leader.speed+t*1.2)
            elif fg<WARN_GAP+am and ego.state!="passing":
                t=(fg-(SAFE_GAP+am))/max(WARN_GAP-SAFE_GAP,1)
                desired=min(desired,EGO_SPEED*(0.62+0.38*t))
        if chaser and self.thr_rear:
            diff=max(chaser.speed-ego.speed,0)
            urg=max(0,1.0-rt/TTC_REAR_WARN)
            boost=diff*0.6+urg*2.2
            desired=min(desired+boost,EGO_OVERTAKE_SPD+2.0)
        return max(desired,0.0)

    def _has_passed(self,to):
        if to is None: return True
        return self.ego.x-self.ego.L//2>to.car.x+to.car.L//2+70

    # ── Spawn particles ──────────────────────────────────
    def _emit_boost_particles(self):
        ego=self.ego; x=ego.x; y=ego.y
        for _ in range(3):
            vx=random.uniform(-3,-0.8)
            vy=random.uniform(-1.5,1.5)
            col=random.choice([NEON_ORANGE,(255,255,120),NEON_BLUE])
            self.particles.append(Particle(x-ego.L//2,y,vx,vy,
                                           random.randint(12,28),col,
                                           random.randint(2,5)))

    def _emit_brake_particles(self):
        ego=self.ego
        for _ in range(2):
            vx=random.uniform(-0.5,0.5); vy=random.uniform(-1,1)
            self.particles.append(Particle(ego.x,ego.y,vx,vy,
                                           random.randint(8,18),(255,60,60),2))

    # ── POLICY ────────────────────────────────────────────
    def _policy(self,obs):
        ego=self.ego; threats=self._scan_threats(obs)
        desired=self._regulate_speed(threats)

        if ego.on_spline:
            self._spline_speed_ctrl()
            if not ego.spline_pause:
                ego.speed=EGO_OVERTAKE_SPD if ego.state=="passing" else desired
            return

        ego.speed=desired
        leader=threats["leader"]; chaser=threats["chaser"]
        fg=threats["front_gap"]; rt=threats["rear_ttc"]
        ranked=self._ranked_lanes(obs)

        def best_direct():
            for lane,cost in ranked:
                if lane==ego.lane: continue
                if abs(lane-ego.lane)>1: continue
                if self.thr_left  and lane<ego.lane: continue
                if self.thr_right and lane>ego.lane: continue
                if self._lane_clear_k(lane,obs): return lane
            return None

        if ego.state=="cruise":
            self.trapped=False
            if chaser and rt<TTC_REAR_EVADE:
                best=best_direct()
                if best is not None:
                    self.n_evade+=1; ego.state="overtaking"; ego.overtake_target=chaser
                    self._plan_change(best); self.status=f"⚠ REAR EVADE → L{best+1}"
                else:
                    mh=self._find_multihop(obs)
                    if mh:
                        f2,final=mh; self.n_evade+=1; self.n_multihop+=1
                        ego.state="overtaking"; ego.overtake_target=chaser
                        self.queued_lane=final; self._plan_change(f2)
                        self.status=f"⚠ EVADE CHAIN L{f2+1}→L{final+1}"
                    else:
                        self.trapped=True
                        ego.speed=max(min(chaser.speed*0.72,
                                         leader.speed-0.3 if leader else 9999),0.5)
                        self.status="⚠ TRAPPED"
                return
            if leader and fg<OVERTAKE_GAP:
                best=best_direct()
                if best is not None:
                    self.n_over+=1; side="Left ←" if best<ego.lane else "Right →"
                    ego.state="overtaking"; ego.overtake_target=leader
                    self._plan_change(best); self.status=f"Overtaking {side}  L{best+1}"
                else:
                    mh=self._find_multihop(obs)
                    if mh:
                        f2,final=mh; self.n_over+=1; self.n_multihop+=1
                        ego.state="overtaking"; ego.overtake_target=leader
                        self.queued_lane=final; self._plan_change(f2)
                        self.status=f"Multi-hop L{f2+1}→L{final+1}"
                    else:
                        if chaser:
                            self.trapped=True
                            ego.speed=max(min(chaser.speed*0.72,
                                             leader.speed-0.3),0.5)
                            self.status="⚠ TRAPPED — no lane"
                        else:
                            self.status="Blocked — following"
                return
            self.status=(f"Following" if leader and fg<SAFE_GAP
                         else f"Cruising  L{ego.lane+1}")
            tags=[]
            if self.thr_left:  tags.append("◄L")
            if self.thr_right: tags.append("R►")
            if self.thr_rear:  tags.append(f"⬆({rt:.0f}f)")
            if tags: self.status+="  "+"".join(tags)

        elif ego.state=="overtaking":
            if self.queued_lane is not None:
                ql=self.queued_lane; self.queued_lane=None
                if self._lane_clear_k(ql,obs):
                    self._plan_change(ql); self.status=f"⚡ Chain → L{ql+1}"
                else:
                    ego.state="passing"; ego.speed=EGO_OVERTAKE_SPD
                    self.status=f"⚡ Passing  L{ego.lane+1}"
            else:
                ego.state="passing"; ego.speed=EGO_OVERTAKE_SPD
                self.status=f"⚡ Passing  L{ego.lane+1}"

        elif ego.state=="passing":
            if self._has_passed(ego.overtake_target):
                ego.state="cruise"; ego.overtake_target=None
                ego.speed=EGO_SPEED; self.status=f"✓ Passed  L{ego.lane+1}"
            else:
                if leader and fg<SAFE_GAP:
                    ego.speed=max(desired,leader.speed+0.2)
                    self.status="⚡ Passing (lane slow)"
                else:
                    ego.speed=min(ego.speed+0.12,EGO_OVERTAKE_SPD)
                    self.status=f"⚡ Passing  L{ego.lane+1}"

    # ── UPDATE ────────────────────────────────────────────
    def update(self):
        self.frame_count+=1; self.frame+=1
        ego=self.ego
        if self.adaptive_margin>0:
            self.adaptive_margin=max(0.0,self.adaptive_margin-MARGIN_DECAY_RATE)

        ego.prev_x=ego.x
        if ego.on_spline:
            done=ego.step_spline()
            if done and ego.state=="overtaking":
                obs=self._update_sensors(); self._policy(obs)
        else:
            ego.x+=ego.speed

        # Particles
        if ego.state=="passing" and self.frame%2==0:
            self._emit_boost_particles()
        if ego.speed<0.5 and self.thr_front and self.frame%3==0:
            self._emit_brake_particles()

        for p in self.particles: p.update()
        self.particles=[p for p in self.particles if p.life>0]

        for npc in self.npcs:
            npc.prev_x=npc.x
            if npc.npc_type=="aggressive" and ego.x>npc.x:
                ge=ego.x-npc.x
                if ge>180: npc.speed=min(npc.speed+0.04,NPC_SPEEDS["aggressive"][1])
                else: npc.speed=max(npc.speed-0.08,ego.speed*0.88)
            npc.x+=npc.speed
            if npc.x<ego.x-W*0.65: self._respawn_npc(npc)

        for i,a in enumerate(self.npcs):
            for j,b in enumerate(self.npcs):
                if i==j: continue
                if a.lane==b.lane:
                    g2=b.x-a.x
                    if 0<g2<115: a.speed=max(a.speed-0.07,0.0)
                    elif g2>265: a.speed=min(a.speed+0.012,NPC_SPEEDS[a.npc_type][1])

        self.cam=ego.x-265
        self.radar_angle=(self.radar_angle+3)%360
        obs=self._update_sensors(); self._policy(obs)

    # ─────────────────────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────────────────────
    def _smooth_alpha(self,key,active,ru=12,rd=7,cap=110):
        if active: self.alpha[key]=min(self.alpha[key]+ru,cap)
        else:      self.alpha[key]=max(self.alpha[key]-rd,0)
        return self.alpha[key]

    def _draw_threat_zones(self):
        ego=self.ego; ex=int(ego.x-self.cam); am=int(self.adaptive_margin)

        def lane_rect(l,extra_w=0):
            return (int(lane_y(l)-LANE_H//2+5), LANE_H-10)

        # FRONT
        a=self._smooth_alpha("front",self.thr_front)
        if a>0:
            ly,lh=lane_rect(ego.lane)
            z=pygame.Surface((WARN_GAP+am,lh),pygame.SRCALPHA)
            for dx in range(WARN_GAP+am):
                fade=int(a*(1-dx/(WARN_GAP+am))*0.9)
                pygame.draw.line(z,(255,100,0,fade),(dx,0),(dx,lh))
            self.screen.blit(z,(ex+ego.L//2,ly))

        # REAR
        a=self._smooth_alpha("rear",self.thr_rear)
        if a>0:
            ly,lh=lane_rect(ego.lane)
            urg=max(0,1.0-self.rear_ttc/TTC_REAR_WARN) if self.rear_ttc<TTC_REAR_WARN else 0
            zw=int(REAR_WARN_GAP*(0.38+0.62*urg))
            rx=ex-ego.L//2-zw
            z=pygame.Surface((zw,lh),pygame.SRCALPHA)
            for dx in range(zw):
                fade=int(a*(dx/zw)*0.9)
                pygame.draw.line(z,(180,0,255,fade),(dx,0),(dx,lh))
            self.screen.blit(z,(rx,ly))
            label=self.font_xs.render(
                f"◄ REAR  {self.rear_ttc:.0f}f" if self.rear_ttc<9999 else "◄ REAR",
                True,(200,110,255))
            self.screen.blit(label,(max(rx+5,0),ly+4))

        # LEFT
        a=self._smooth_alpha("left",self.thr_left)
        if a>0 and ego.lane>0:
            ly,lh=lane_rect(ego.lane-1)
            z=pygame.Surface((ego.L+SIDE_WARN_X*2,lh),pygame.SRCALPHA)
            z.fill((255,210,0,a//2)); self.screen.blit(z,(ex-ego.L//2-SIDE_WARN_X,ly))
            self.screen.blit(self.font_xs.render("◄ LEFT",True,(255,230,80)),(ex-30,ly+4))

        # RIGHT
        a=self._smooth_alpha("right",self.thr_right)
        if a>0 and ego.lane<NUM_LANES-1:
            ly,lh=lane_rect(ego.lane+1)
            z=pygame.Surface((ego.L+SIDE_WARN_X*2,lh),pygame.SRCALPHA)
            z.fill((0,210,255,a//2)); self.screen.blit(z,(ex-ego.L//2-SIDE_WARN_X,ly))
            self.screen.blit(self.font_xs.render("RIGHT ►",True,(80,230,255)),(ex-30,ly+4))

        # MERGE
        a=self._smooth_alpha("merge",self.thr_merge)
        if a>0 and ego.on_spline:
            ly,lh=lane_rect(ego.target_lane)
            z=pygame.Surface((ego.L+SIDE_CRIT_X*2,lh),pygame.SRCALPHA)
            z.fill((255,30,150,a//2)); self.screen.blit(z,(ex-ego.L//2-SIDE_CRIT_X,ly))
            ls=(f"LC {self.lc_rear_ttc:.0f}f" if self.lc_rear_ttc<9999 else "BLOCKED")
            self.screen.blit(self.font_xs.render(ls,True,(255,100,200)),(ex-30,ly+4))

        # TRAPPED halo
        if self.trapped:
            t=0.5+0.5*math.sin(self.frame*0.12)
            halo=pygame.Surface((ego.L+50,ego.CW+50),pygame.SRCALPHA)
            pygame.draw.rect(halo,(255,30,30,int(60*t)),halo.get_rect(),border_radius=16)
            self.screen.blit(halo,(ex-ego.L//2-25,int(ego.y)-ego.CW//2-25))

        # PASSING trail
        if ego.state=="passing":
            trail=pygame.Surface((210,ego.CW+28),pygame.SRCALPHA)
            for i in range(8):
                al=int(42*(1-i/8))
                c=lerp_col(NEON_ORANGE,NEON_BLUE,i/8)
                pygame.draw.rect(trail,(*c,al),(i*26,0,210-i*26,ego.CW+28),border_radius=10)
            self.screen.blit(trail,(ex-ego.L//2-210,int(ego.y)-ego.CW//2-14))

        # Target ring + arrow
        if ego.overtake_target is not None:
            tc=ego.overtake_target.car
            tx=int(tc.x-self.cam); ty=int(tc.y)
            t2=0.5+0.5*math.sin(self.frame*0.15)
            r=int(tc.L//2+8+4*t2)
            tg=pygame.Surface((r*2+4,r*2+4),pygame.SRCALPHA)
            pygame.draw.circle(tg,(255,210,0,int(180*t2)),(r+2,r+2),r,2)
            self.screen.blit(tg,(tx-r-2,ty-r-2))
            pygame.draw.line(self.screen,(255,210,0,150),(ex+ego.L//2,int(ego.y)),(tx-tc.L//2,ty),2)
            lbl=self.font_xs.render("TARGET",True,(255,230,60))
            self.screen.blit(lbl,(tx-lbl.get_width()//2,ty-tc.CW//2-18))

    # ── Lane cost overlay ─────────────────────────────────
    def _draw_lane_cost_tint(self):
        if not self.lane_costs: return
        mx=max(self.lane_costs) if max(self.lane_costs)>0 else 1
        for i,cost in enumerate(self.lane_costs):
            t=max(0.0, min(cost/max(mx,1),1.0))   # clamp: negative costs (open lane bonus) must not go below 0
            col=lerp_col((0,180,80),(220,40,40),t)
            tint=pygame.Surface((W,LANE_H),pygame.SRCALPHA)
            tint.fill((*col, max(0, min(255, int(18*t)))))
            self.screen.blit(tint,(0,ROAD_TOP+i*LANE_H))
            # Cost badge on right edge
            badge=pygame.Surface((50,22),pygame.SRCALPHA)
            badge.fill((*col,180))
            lbl=self.font_xs.render(f"{cost:.0f}",True,C_WHITE)
            badge.blit(lbl,(badge.get_width()//2-lbl.get_width()//2,3))
            self.screen.blit(badge,(W-58,lane_y(i)-11))

    # ── Kalman ghosts ─────────────────────────────────────
    def _draw_kalman_ghosts(self):
        if not self.show_ghosts: return
        for o in self.obs_cache:
            gx=int(o.pred_x-self.cam); gy=int(o.y)
            if 0<gx<W:
                t=0.4+0.6*abs(math.sin(self.frame*0.06+o.x*0.01))
                al=int(t*160)
                pygame.draw.circle(self.screen,(*NEON_CYAN,al),(gx,gy),7,2)
                cx=int(o.x-self.cam)
                dash_pts=[]
                n=8
                for i in range(n):
                    px=int(cx+(gx-cx)*i/n); py=int(gy)
                    if i%2==0: dash_pts.append((px,py))
                if len(dash_pts)>1:
                    for p3 in dash_pts:
                        pygame.draw.circle(self.screen,(*NEON_CYAN,60),p3,1)

    # ── Spline preview ────────────────────────────────────
    def _draw_spline(self):
        if not self.show_sp or len(self.sp_preview)<2: return
        pts=[(int(p[0]-self.cam),int(p[1])) for p in self.sp_preview]
        vis=[p for p in pts if -10<p[0]<W+10]
        if len(vis)>1:
            col=(255,80,160) if self.ego.spline_pause else (55,230,110)
            # Draw with glow
            for lw,al in [(5,40),(3,80),(2,200)]:
                s2=pygame.Surface((W,H),pygame.SRCALPHA)
                pygame.draw.lines(s2,(*col,al),False,vis,lw)
                self.screen.blit(s2,(0,0))
        for i in range(0,len(pts),40):
            if 0<=pts[i][0]<=W:
                pygame.draw.circle(self.screen,col,pts[i],4)

    # ── Particles ─────────────────────────────────────────
    def _draw_particles(self):
        for p in self.particles:
            p.draw(self.screen,self.cam)

    # ── Speedometer ──────────────────────────────────────
    def _draw_speedometer(self):
        s=self.screen; cx,cy,r=W-95,H-95,68
        # Outer ring glow
        glow_col=NEON_ORANGE if self.ego.state=="passing" else NEON_BLUE
        for gr in range(r+14,r+4,-2):
            al=max(0,int((r+14-gr)*18))
            g=pygame.Surface((gr*2,gr*2),pygame.SRCALPHA)
            pygame.draw.circle(g,(*glow_col,al),(gr,gr),gr,2)
            s.blit(g,(cx-gr,cy-gr))

        pygame.draw.circle(s,(16,18,24),(cx,cy),r+4)
        pygame.draw.circle(s,(28,30,38),(cx,cy),r+4,2)
        pygame.draw.circle(s,(10,12,18),(cx,cy),r)

        # Tick marks with colour gradient
        for i in range(21):
            ang=math.radians(-225+i*(270/20))
            lf=r-9 if i%5==0 else r-5
            x1=cx+math.cos(ang)*(r-2); y1=cy+math.sin(ang)*(r-2)
            x2=cx+math.cos(ang)*lf;    y2=cy+math.sin(ang)*lf
            tc=lerp_col((60,180,255),(255,80,50),i/20)
            pygame.draw.line(s,tc,(int(x1),int(y1)),(int(x2),int(y2)),2 if i%5==0 else 1)

        # Arc fill
        arc_surf=pygame.Surface((r*2,r*2),pygame.SRCALPHA)
        mxs=EGO_OVERTAKE_SPD+2.0; spd_t=min(self.ego.speed/mxs,1.0)
        arc_col=lerp_col(NEON_BLUE,NEON_ORANGE,spd_t)
        start_a=-225; end_a=start_a+spd_t*270
        if spd_t>0.01:
            for deg in range(int(start_a),int(end_a),2):
                ang=math.radians(deg)
                px=int(r+math.cos(ang)*(r-6)); py=int(r+math.sin(ang)*(r-6))
                pygame.draw.circle(arc_surf,(*arc_col,120),(px,py),3)
        s.blit(arc_surf,(cx-r,cy-r))

        # Needle
        ang=math.radians(-225+spd_t*270)
        nx=cx+math.cos(ang)*(r-12); ny=cy+math.sin(ang)*(r-12)
        pygame.draw.line(s,(*glow_col,220),(cx,cy),(int(nx),int(ny)),3)
        pygame.draw.line(s,(255,255,255,100),(cx,cy),(int(nx),int(ny)),1)
        pygame.draw.circle(s,(40,44,55),(cx,cy),8)
        pygame.draw.circle(s,(*glow_col,200),(cx,cy),5)

        # Speed readout
        t1=self.font_m.render(f"{self.ego.speed*20:.0f}",True,C_WHITE)
        t2=self.font_xs.render("km/h",True,(80,85,100))
        s.blit(t1,(cx-t1.get_width()//2,cy+14))
        s.blit(t2,(cx-t2.get_width()//2,cy+30))

    # ── HUD ───────────────────────────────────────────────
    def _draw_hud(self):
        s=self.screen
        panel_w,panel_h=290,272
        # Frosted glass panel with glowing border
        pan=pygame.Surface((panel_w,panel_h),pygame.SRCALPHA)
        pan.fill((6,8,18,195))
        s.blit(pan,(10,10))
        # Animated neon border
        t2=0.5+0.5*math.sin(self.frame*0.05)
        border_col=lerp_col(NEON_BLUE,NEON_CYAN,t2)
        pygame.draw.rect(s,(*border_col,int(120+60*t2)),(10,10,panel_w,panel_h),1,border_radius=4)

        state_col={
            "cruise":    NEON_GREEN,
            "overtaking":NEON_ORANGE,
            "passing":   NEON_ORANGE,
        }.get(self.ego.state,C_WHITE)

        ttc_col=((255,70,70) if self.rear_ttc<TTC_REAR_EVADE else
                 (255,195,60) if self.rear_ttc<TTC_REAR_WARN else NEON_GREEN)
        lc_col=((255,70,70) if self.lc_rear_ttc<TTC_LC_PAUSE else
                (255,195,60) if self.lc_rear_ttc<TTC_LC_SLOW  else NEON_GREEN)
        am_pct=int(self.adaptive_margin/NEAR_MISS_MARGIN_MAX*100)
        am_col=(255,80,80) if am_pct>60 else (255,200,80) if am_pct>25 else NEON_GREEN
        tgt=(f"L{self.ego.overtake_target.car.lane+1}({self.ego.overtake_target.npc_type[0].upper()})"
             if self.ego.overtake_target else "—")

        rows=[
            ("STATUS",   self.status[:30],               C_WHITE),
            ("STATE",    self.ego.state.upper(),          state_col),
            ("LANE",     f"L{self.ego.lane+1}",           C_WHITE),
            ("SPEED",    f"{self.ego.speed*20:.0f} km/h", C_WHITE),
            ("REAR TTC", f"{self.rear_ttc:.0f}f" if self.rear_ttc<9999 else "clear", ttc_col),
            ("LC TTC",   f"{self.lc_rear_ttc:.0f}f" if self.lc_rear_ttc<9999 else "—", lc_col),
            ("TARGET",   tgt,                             (255,215,55)),
            ("MULTIHOP", "QUEUED" if self.queued_lane else "—",(100,220,255)),
            ("MARGIN",   f"+{self.adaptive_margin:.0f}px",am_col),
            ("NEAR MISS",str(self.near_miss_count),       am_col),
        ]
        for i,(k,v,c) in enumerate(rows):
            # Key
            ks=self.font_xs.render(f"{k:<10}",True,(80,85,110))
            s.blit(ks,(18,18+i*24))
            # Value with subtle glow for active states
            vs=self.font_xs.render(v,True,c)
            s.blit(vs,(150,18+i*24))

        # Threat pills row
        indicators=[("FWD",self.thr_front,(255,80,0)),
                    ("REAR",self.thr_rear,(160,0,255)),
                    ("LEFT",self.thr_left,(255,200,0)),
                    ("RIGHT",self.thr_right,(0,200,255)),
                    ("MERGE",self.thr_merge,(255,30,140))]
        ix=14
        for label,active,col in indicators:
            pill=pygame.Surface((50,20),pygame.SRCALPHA)
            pill.fill((*col,215 if active else 35))
            s.blit(pill,(ix,258))
            ls=self.font_xs.render(label,True,C_WHITE if active else (60,65,80))
            s.blit(ls,(ix+4,260)); ix+=54

        # Warning banners
        by=285
        if self.thr_front:
            self._banner(s,"⚠ FRONT COLLISION RISK",(200,40,0),by); by+=34
        if self.thr_rear:
            uc=(255,0,80) if self.rear_ttc<TTC_REAR_EVADE else (140,0,220)
            self._banner(s,f"◄ REAR  {self.rear_ttc:.0f}f",uc,by); by+=34
        if self.thr_left:
            self._banner(s,"◄ LEFT BLOCKED",(170,150,0),by); by+=34
        if self.thr_right:
            self._banner(s,"► RIGHT BLOCKED",(0,150,190),by); by+=34
        if self.thr_merge:
            self._banner(s,f"! LC THREAT  {self.lc_rear_ttc:.0f}f",(195,15,125),by); by+=34
        if self.trapped:
            self._banner(s,"⚠ TRAPPED — BALANCING",(150,15,15),by)

        # State badge (top centre) with glow
        bd=pygame.Surface((300,32),pygame.SRCALPHA)
        bd.fill((*state_col,195))
        ts=self.font_l.render(self.ego.state.upper(),True,C_WHITE)
        bd.blit(ts,(bd.get_width()//2-ts.get_width()//2,5))
        s.blit(bd,(W//2-150,10))

        # Counters strip (bottom)
        cnt_s=self.font_xs.render(
            f"  OVERTAKES {self.n_over}   EVADES {self.n_evade}   MULTIHOP {self.n_multihop}",
            True,(100,180,100))
        cb=pygame.Surface((cnt_s.get_width()+16,20),pygame.SRCALPHA)
        cb.fill((0,0,0,150)); s.blit(cb,(10,H-32))
        s.blit(cnt_s,(18,H-30))

        # Radar minimap (bottom right)
        self._draw_radar()

        hint=self.font_xs.render("S spline   G ghosts   R reset   ESC quit",True,(50,52,65))
        s.blit(hint,(W//2-hint.get_width()//2,H-14))

    # ── Radar minimap ─────────────────────────────────────
    def _draw_radar(self):
        s=self.screen; rcx=W-90; rcy=H-220; rr=70
        # Background
        radar=pygame.Surface((rr*2+4,rr*2+4),pygame.SRCALPHA)
        pygame.draw.circle(radar,(0,10,20,210),(rr+2,rr+2),rr)
        pygame.draw.circle(radar,(*NEON_CYAN,60),(rr+2,rr+2),rr,1)
        # Grid circles
        for gr in [rr//3, 2*rr//3]:
            pygame.draw.circle(radar,(*NEON_CYAN,30),(rr+2,rr+2),gr,1)
        # Cross lines
        pygame.draw.line(radar,(*NEON_CYAN,30),(2,rr+2),(rr*2+2,rr+2))
        pygame.draw.line(radar,(*NEON_CYAN,30),(rr+2,2),(rr+2,rr*2+2))
        # Radar sweep
        sweep_ang=math.radians(self.radar_angle)
        sweep_col=pygame.Surface((rr*2+4,rr*2+4),pygame.SRCALPHA)
        for da in range(0,45,3):
            ang=sweep_ang-math.radians(da)
            al=max(0,int((45-da)/45*80))
            ex2=int(rr+2+math.cos(ang)*rr); ey2=int(rr+2+math.sin(ang)*rr)
            pygame.draw.line(sweep_col,(*NEON_GREEN,al),(rr+2,rr+2),(ex2,ey2),2)
        radar.blit(sweep_col,(0,0))
        # Ego dot (centre)
        pygame.draw.circle(radar,(*NEON_BLUE,255),(rr+2,rr+2),5)
        # NPC dots
        view=1600
        for nc in self.npcs:
            rx=(nc.x-self.ego.x)/view; ry=(nc.y-lane_y(1))/(NUM_LANES*LANE_H)
            px=int(rr+2+rx*rr*0.88); py=int(rr+2+ry*rr*0.88)
            if 0<px<rr*2+4 and 0<py<rr*2+4:
                col=nc.pal[0]
                if nc.npc_type=="stopped": col=(220,200,0)
                elif nc.npc_type=="aggressive": col=(255,40,40)
                elif nc.npc_type=="truck": col=(160,160,180)
                pygame.draw.circle(radar,col,(px,py),4)
                pygame.draw.circle(radar,(*col,80),(px,py),7,1)
        pygame.draw.circle(radar,(*NEON_CYAN,140),(rr+2,rr+2),rr,1)
        s.blit(radar,(rcx-rr-2,rcy-rr-2))
        title=self.font_xs.render("RADAR",True,(*NEON_CYAN,180))
        s.blit(title,(rcx-title.get_width()//2,rcy-rr-18))

    def _banner(self,surf,text,colour,y):
        bw=pygame.Surface((290,30),pygame.SRCALPHA)
        bw.fill((*colour,200))
        # Glow edge
        pygame.draw.rect(bw,(*colour,80),(0,0,290,30),1,border_radius=2)
        t=self.font_s.render(text,True,C_WHITE)
        bw.blit(t,(bw.get_width()//2-t.get_width()//2,6))
        surf.blit(bw,(10,y))

    # ── MAIN DRAW ─────────────────────────────────────────
    def draw(self):
        # Background + scenery
        self.scenery.draw(self.screen, self.cam, self.frame)
        # Road
        self.scenery.draw_road_details(self.screen, self.cam, self.frame)
        # Lane cost tints
        self._draw_lane_cost_tint()
        # Threat zones (before cars so cars render on top)
        self._draw_threat_zones()
        # Spline path
        self._draw_spline()
        # Kalman ghosts
        self._draw_kalman_ghosts()
        # Cars (sorted by y for pseudo-depth)
        for c in sorted(self.npcs+[self.ego],key=lambda c:c.y):
            c.draw(self.screen,self.cam)
        # Particles on top of cars
        self._draw_particles()
        # UI
        self._draw_hud()
        self._draw_speedometer()
        pygame.display.flip()

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE,pygame.K_q):
                        pygame.quit(); sys.exit()
                    elif ev.key==pygame.K_s: self.show_sp=not self.show_sp
                    elif ev.key==pygame.K_g: self.show_ghosts=not self.show_ghosts
                    elif ev.key==pygame.K_r: self._reset()
            self.update(); self.draw()
            self.clock.tick(FPS)


if __name__=="__main__":
    Sim().run()
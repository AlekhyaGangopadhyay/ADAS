import itertools
import math
import random
import sys

import numpy as np
import pygame

W, H = 1280, 720
FPS = 60
NUM_LANES = 3
LANE_H = 92
ROAD_TOP = (H - NUM_LANES * LANE_H) // 2
ROAD_BOT = ROAD_TOP + NUM_LANES * LANE_H
SHOULDER = 34

C_SKY = (185, 210, 228)
C_HAZE = (239, 219, 177)
C_ROAD = (58, 58, 60)
C_ROAD_ALT = (64, 64, 66)
C_SHOULDER = (156, 138, 96)
C_EDGE = (244, 192, 59)
C_DASH = (230, 230, 218)
C_WHITE = (250, 250, 248)
C_TEXT_DIM = (110, 114, 125)
C_GREEN = (74, 166, 102)
C_RED = (215, 64, 54)
C_ORANGE = (232, 129, 43)
C_AMBER = (247, 188, 56)
C_CYAN = (65, 196, 214)
C_PURPLE = (171, 78, 209)

EGO_SPEED = 5.1
EGO_OVERTAKE_SPD = 8.2
RESPAWN_AHEAD = 2300
RADAR_RANGE = 920
RADAR_NOISE_X = 8.0
RADAR_NOISE_Y = 3.2
MAX_ACTIVE_NPCS = 3
RESPAWN_BUFFER_MIN = 900
RESPAWN_BUFFER_MAX = 1300

WARN_GAP = 165
SAFE_GAP = 105
EMERGENCY_GAP = 58
OVERTAKE_GAP = 155
REAR_WARN_GAP = 200
TTC_REAR_WARN = 42
TTC_REAR_EVADE = 24
TTC_LC_PAUSE = 24
TTC_LC_SLOW = 40
LC_REAR_MIN = 130
LC_FRONT_MIN = 120
SIDE_WARN_X = 85
REAR_LANECHANGE_LOOKAHEAD = 36
REAR_PASS_HOLD_TTC = 28
REAR_SPEED_MARGIN = 0.5
FOLLOW_BUFFER = 145
SIDE_PATH_FRONT_BUFFER = 170
SIDE_PATH_REAR_BUFFER = 150

BRAKE_DECEL_SOFT = 0.10
BRAKE_DECEL_HARD = 0.26
BRAKE_DECEL_EMERG = 0.58

TC_FRONT = (255, 118, 62)
TC_REAR = (176, 74, 214)
TC_LEFT = (247, 201, 70)
TC_RIGHT = (69, 201, 219)
TC_MERGE = (255, 92, 155)
TC_BRAKE = (232, 63, 52)

TRACK_ID = itertools.count(1)

VEHICLE_TYPES = {
    "sedan": {"name": "Sedan", "length": 72, "width": 38, "speed_range": (2.8, 4.0), "palette": ((34, 122, 215), (18, 67, 131), (173, 216, 255))},
    "hatchback": {"name": "Hatchback", "length": 64, "width": 36, "speed_range": (2.8, 4.1), "palette": ((203, 73, 56), (118, 38, 27), (255, 174, 164))},
    "suv": {"name": "SUV", "length": 82, "width": 42, "speed_range": (2.9, 4.1), "palette": ((49, 156, 113), (25, 91, 63), (158, 240, 204))},
    "truck": {"name": "Truck", "length": 126, "width": 48, "speed_range": (1.8, 3.0), "palette": ((216, 140, 42), (123, 77, 20), (255, 214, 145))},
    "bus": {"name": "Bus", "length": 138, "width": 50, "speed_range": (1.9, 3.0), "palette": ((198, 51, 108), (121, 24, 63), (255, 173, 211))},
    "auto": {"name": "Auto", "length": 54, "width": 30, "speed_range": (2.2, 3.4), "palette": ((231, 194, 36), (41, 53, 33), (255, 238, 152))},
}

SCENARIOS = [
    {"name": "NH Freight Overtake", "description": "Slow container truck ahead, clear passing lane.", "ego_lane": 1, "preferred_lane": 0, "decor": "highway", "npcs": [{"lane": 1, "x": 700, "vehicle_type": "truck", "speed": 2.1}, {"lane": 0, "x": 1220, "vehicle_type": "sedan", "speed": 3.2}, {"lane": 2, "x": 1800, "vehicle_type": "suv", "speed": 3.6}, {"lane": 1, "x": 2460, "vehicle_type": "bus", "speed": 2.5}]},
    {"name": "Urban Auto Cluster", "description": "Mixed city flow with autos and a bus.", "ego_lane": 1, "preferred_lane": 0, "decor": "city", "npcs": [{"lane": 1, "x": 660, "vehicle_type": "auto", "speed": 2.3}, {"lane": 0, "x": 1140, "vehicle_type": "hatchback", "speed": 3.0}, {"lane": 2, "x": 1660, "vehicle_type": "auto", "speed": 2.5}, {"lane": 1, "x": 2240, "vehicle_type": "bus", "speed": 2.2}, {"lane": 0, "x": 2860, "vehicle_type": "sedan", "speed": 3.4}]},
    {"name": "Bus Stop Merge", "description": "A bus slows near the shoulder while autos squeeze gaps.", "ego_lane": 0, "preferred_lane": 0, "decor": "bus_stop", "npcs": [{"lane": 0, "x": 620, "vehicle_type": "bus", "speed": 1.9}, {"lane": 1, "x": 1120, "vehicle_type": "auto", "speed": 2.4}, {"lane": 1, "x": 1720, "vehicle_type": "truck", "speed": 2.8}, {"lane": 2, "x": 2360, "vehicle_type": "hatchback", "speed": 3.6}]},
    {"name": "Monsoon Mixed Traffic", "description": "Slower traffic with puddles, trucks, and autos.", "ego_lane": 1, "preferred_lane": 0, "decor": "monsoon", "npcs": [{"lane": 1, "x": 720, "vehicle_type": "truck", "speed": 2.0}, {"lane": 2, "x": 1260, "vehicle_type": "auto", "speed": 2.1}, {"lane": 0, "x": 1860, "vehicle_type": "suv", "speed": 2.7}, {"lane": 1, "x": 2480, "vehicle_type": "auto", "speed": 2.2}, {"lane": 2, "x": 3140, "vehicle_type": "bus", "speed": 2.0}]},
    {"name": "Night Freight Convoy", "description": "Two trucks convoy with faster traffic around them.", "ego_lane": 2, "preferred_lane": 0, "decor": "night", "npcs": [{"lane": 2, "x": 760, "vehicle_type": "truck", "speed": 2.2}, {"lane": 2, "x": 1420, "vehicle_type": "truck", "speed": 2.3}, {"lane": 1, "x": 1960, "vehicle_type": "sedan", "speed": 3.5}, {"lane": 0, "x": 2580, "vehicle_type": "suv", "speed": 3.8}, {"lane": 1, "x": 3240, "vehicle_type": "auto", "speed": 2.6}]},
]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lane_y(idx):
    return ROAD_TOP + idx * LANE_H + LANE_H // 2


def b_spline(p0, p1, p2, p3, n=28):
    p0, p1, p2, p3 = map(np.asarray, [p0, p1, p2, p3])
    pts = []
    for i in range(n):
        t = i / (n - 1)
        t2 = t * t
        t3 = t2 * t
        b0 = (1 - t) ** 3 / 6.0
        b1 = (3 * t3 - 6 * t2 + 4) / 6.0
        b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
        b3 = t3 / 6.0
        pts.append((b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3).tolist())
    return pts


def build_spline(waypoints, n=28):
    wp = [waypoints[0], waypoints[0]] + list(waypoints) + [waypoints[-1], waypoints[-1]]
    path = []
    for i in range(len(wp) - 3):
        seg = b_spline(wp[i], wp[i + 1], wp[i + 2], wp[i + 3], n)
        if i > 0:
            seg = seg[1:]
        path.extend(seg)
    return path


def ttc(gap, ego_spd, other_spd):
    closing = other_spd - ego_spd
    return gap / closing if closing > 0.01 else 9999


class KalmanTrack:
    def __init__(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4) * 20.0
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.diag([RADAR_NOISE_X ** 2, RADAR_NOISE_Y ** 2])
        self.frames_since_update = 0

    def predict(self, dt=1.0):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = 0.22
        G = np.array([[0.5 * dt * dt], [0.5 * dt * dt], [dt], [dt]], dtype=float)
        Q = (G @ G.T) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.frames_since_update += 1

    def update(self, mx, my):
        z = np.array([[mx], [my]], dtype=float)
        innovation = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.frames_since_update = 0

    @property
    def pos(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self):
        return float(self.x[2, 0]), float(self.x[3, 0])


class Car:
    def __init__(self, x, lane, vehicle_type, speed, is_ego=False):
        profile = VEHICLE_TYPES[vehicle_type]
        self.track_id = next(TRACK_ID)
        self.vehicle_type = vehicle_type
        self.vehicle_name = profile["name"]
        self.length = profile["length"]
        self.width = profile["width"]
        self.pal = profile["palette"]
        self.x = float(x)
        self.lane = lane
        self.y = float(lane_y(lane))
        self.speed = speed
        self.is_ego = is_ego
        self.state = "cruise"
        self.target_lane = lane
        self.path = []
        self.path_idx = 0
        self.on_spline = False
        self.spline_pause = False
        self.overtake_target = None
        self.blinker = 0
        self.blink_t = 0
        self.braking = False
        self.brake_intensity = 0.0
        self.passed_by_ego = False

    @property
    def rect(self):
        return pygame.Rect(self.x - self.length // 2, self.y - self.width // 2, self.length, self.width)

    def step_spline(self):
        if not self.path or self.path_idx >= len(self.path):
            self.on_spline = False
            self.spline_pause = False
            self.lane = self.target_lane
            self.y = lane_y(self.target_lane)
            self.path = []
            self.path_idx = 0
            self.blinker = 0
            return True
        if self.spline_pause:
            return False
        px, py = self.path[self.path_idx]
        self.x = px
        self.y = py
        self.path_idx += 1
        return False

    def draw(self, surf, cam):
        sx = int(self.x - cam)
        sy = int(self.y)
        body, dark, glass = self.pal
        angle = 0.0
        if self.on_spline and self.path and self.path_idx < len(self.path):
            idx = self.path_idx
            lookahead = min(idx + 4, len(self.path) - 1)
            dx = self.path[lookahead][0] - self.path[idx][0]
            dy = self.path[lookahead][1] - self.path[idx][1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                angle = -math.degrees(math.atan2(dy, dx)) * 0.42

        pad = 56
        car_surf = pygame.Surface((self.length + pad, self.width + pad), pygame.SRCALPHA)
        cx = car_surf.get_width() // 2
        cy = car_surf.get_height() // 2
        base = pygame.Rect(cx - self.length // 2, cy - self.width // 2, self.length, self.width)
        pygame.draw.rect(car_surf, (0, 0, 0, 52), base.move(2, 5), border_radius=9)

        if self.vehicle_type in {"truck", "bus"}:
            cab_w = int(self.length * 0.28)
            cargo = pygame.Rect(base.x, base.y, self.length - cab_w, self.width)
            cab = pygame.Rect(base.right - cab_w, base.y + 4, cab_w, self.width - 8)
            pygame.draw.rect(car_surf, body, cargo, border_radius=7)
            pygame.draw.rect(car_surf, dark, cargo, 2, border_radius=7)
            pygame.draw.rect(car_surf, tuple(clamp(c + 18, 0, 255) for c in body), cab, border_radius=7)
            pygame.draw.rect(car_surf, dark, cab, 2, border_radius=7)
        elif self.vehicle_type == "auto":
            shell = pygame.Rect(base.x + 6, base.y + 4, self.length - 12, self.width - 8)
            pygame.draw.rect(car_surf, body, shell, border_radius=8)
            roof = [(base.x + 12, base.bottom - 4), (base.x + 22, base.y + 5), (base.right - 16, base.y + 5), (base.right - 8, base.bottom - 4)]
            pygame.draw.polygon(car_surf, dark, roof)
            pygame.draw.rect(car_surf, glass, (base.x + 20, base.y + 9, self.length - 32, self.width - 18), border_radius=4)
            pygame.draw.rect(car_surf, dark, shell, 2, border_radius=8)
        else:
            pygame.draw.rect(car_surf, body, base, border_radius=10)
            pygame.draw.rect(car_surf, dark, base, 2, border_radius=10)
            roof = pygame.Rect(base.x + 16, base.y + 6, self.length - 32, self.width - 12)
            roof_surf = pygame.Surface((roof.w, roof.h), pygame.SRCALPHA)
            roof_surf.fill((*glass, 160))
            car_surf.blit(roof_surf, roof.topleft)
            pygame.draw.rect(car_surf, dark, roof, 1, border_radius=5)

        for wx, wy in ((base.x + 8, base.y - 6), (base.right - 22, base.y - 6), (base.x + 8, base.bottom - 2), (base.right - 22, base.bottom - 2)):
            pygame.draw.rect(car_surf, (30, 30, 32), (wx, wy, 14, 8), border_radius=3)

        for hy in (base.y + 5, base.bottom - 12):
            pygame.draw.ellipse(car_surf, (255, 246, 199), (base.right - 8, hy, 9, 8))
            tail = (255, 70, 54) if self.braking else (196, 54, 48)
            pygame.draw.ellipse(car_surf, tail, (base.x - 1, hy, 10, 8))

        if self.blinker != 0 and self.blink_t < 20:
            by = base.y - 4 if self.blinker < 0 else base.bottom - 2
            pygame.draw.rect(car_surf, (255, 162, 36), (base.right - 28, by, 22, 6), border_radius=2)
        self.blink_t = (self.blink_t + 1) % 40

        if self.is_ego:
            glow_color = (255, 92, 70) if self.braking else (49, 148, 241)
            glow_alpha = 44 if not self.braking else int(60 + 70 * self.brake_intensity)
            glow = pygame.Surface((self.length + 20, self.width + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*glow_color, glow_alpha), glow.get_rect(), border_radius=15)
            car_surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))

        if angle:
            rotated = pygame.transform.rotate(car_surf, angle)
            surf.blit(rotated, rotated.get_rect(center=(sx, sy)))
        else:
            surf.blit(car_surf, (sx - cx, sy - cy))

class Sim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("ADAS Indian Highway B-Spline Overtake")
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 15)
        self.font_m = pygame.font.SysFont("Consolas", 19, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        self.scenario_index = 0
        self._reset()

    def _reset(self, scenario_index=None):
        if scenario_index is not None:
            self.scenario_index = scenario_index % len(SCENARIOS)
        self.scenario = SCENARIOS[self.scenario_index]
        self.ego = Car(250, self.scenario["ego_lane"], "sedan", EGO_SPEED, is_ego=True)
        self.preferred_lane = self.scenario["preferred_lane"]
        active_specs = self.scenario["npcs"][:MAX_ACTIVE_NPCS]
        self.npcs = [Car(self.ego.x + spec["x"], spec["lane"], spec["vehicle_type"], spec["speed"]) for spec in active_specs]
        self.cam = 0.0
        self.road_off = 0.0
        self.show_sp = True
        self.sp_preview = []
        self.status = "Cruising"
        self.trapped = False
        self.braking = False
        self.n_over = 0
        self.n_evade = 0
        self.n_collisions = 0
        self.thr_front = False
        self.thr_rear = False
        self.thr_left = False
        self.thr_right = False
        self.thr_merge = False
        self.rear_ttc = 9999
        self.lc_rear_ttc = 9999
        self.alpha = {key: 0 for key in ("front", "rear", "left", "right", "merge", "brake")}
        self.radar_tracks = {npc.track_id: KalmanTrack(npc.x, npc.y) for npc in self.npcs}
        self.last_radar_hits = []

    def _tracked_state(self, car):
        track = self.radar_tracks.get(car.track_id)
        if not track:
            return {"x": car.x, "y": car.y, "vx": car.speed, "vy": 0.0}
        tx, ty = track.pos
        vx, vy = track.velocity
        return {"x": tx, "y": ty, "vx": vx, "vy": vy}

    def _simulate_radar(self):
        hits = []
        for car in self.npcs:
            dx = car.x - self.ego.x
            if abs(dx) > RADAR_RANGE:
                continue
            mx = car.x + random.gauss(0.0, RADAR_NOISE_X)
            my = car.y + random.gauss(0.0, RADAR_NOISE_Y)
            hits.append((car.track_id, mx, my))
        self.last_radar_hits = hits
        return hits

    def _update_radar_tracks(self):
        for track in self.radar_tracks.values():
            track.predict(1.0)
        for track_id, mx, my in self._simulate_radar():
            if track_id not in self.radar_tracks:
                self.radar_tracks[track_id] = KalmanTrack(mx, my)
            self.radar_tracks[track_id].update(mx, my)

    def _respawn_npc(self, npc):
        template = random.choice(self.scenario["npcs"])
        profile = VEHICLE_TYPES[template["vehicle_type"]]
        npc.vehicle_type = template["vehicle_type"]
        npc.vehicle_name = profile["name"]
        npc.length = profile["length"]
        npc.width = profile["width"]
        npc.pal = profile["palette"]
        lo, hi = profile["speed_range"]
        npc.speed = clamp(random.uniform(lo, hi), lo, hi)
        npc.lane = random.randint(0, NUM_LANES - 1)
        npc.target_lane = npc.lane
        npc.y = lane_y(npc.lane)
        npc.x = max(
            self.ego.x + RESPAWN_AHEAD,
            max((c.x for c in self.npcs if c is not npc), default=self.ego.x) + random.randint(RESPAWN_BUFFER_MIN, RESPAWN_BUFFER_MAX),
        )
        npc.path = []
        npc.path_idx = 0
        npc.on_spline = False
        npc.spline_pause = False
        self.radar_tracks[npc.track_id] = KalmanTrack(npc.x, npc.y)

    def _scan_threats(self):
        ego = self.ego
        ex = ego.x

        def in_lane(car, lane_idx):
            return abs(self._tracked_state(car)["y"] - lane_y(lane_idx)) < LANE_H * 0.42

        def overlap_x(car, extra=0):
            return abs(self._tracked_state(car)["x"] - ex) < ego.length + extra

        same = [c for c in self.npcs if in_lane(c, ego.lane)]
        ahead_same = sorted([c for c in same if self._tracked_state(c)["x"] > ex], key=lambda c: self._tracked_state(c)["x"])
        behind_same = sorted([c for c in same if self._tracked_state(c)["x"] <= ex], key=lambda c: -self._tracked_state(c)["x"])
        leader = ahead_same[0] if ahead_same else None
        chaser = behind_same[0] if behind_same else None
        front_gap = self._tracked_state(leader)["x"] - ex - (leader.length + ego.length) * 0.5 if leader else 9999
        rear_gap = ex - self._tracked_state(chaser)["x"] - (chaser.length + ego.length) * 0.5 if chaser else 9999

        self.thr_front = leader is not None and front_gap < WARN_GAP
        self.thr_rear = chaser is not None and (rear_gap < REAR_WARN_GAP or ttc(rear_gap, ego.speed, chaser.speed) < TTC_REAR_WARN)
        self.rear_ttc = ttc(rear_gap, ego.speed, chaser.speed) if chaser else 9999

        left_lane = ego.lane - 1
        right_lane = ego.lane + 1
        self.thr_left = left_lane >= 0 and any(in_lane(c, left_lane) and overlap_x(c, SIDE_WARN_X) for c in self.npcs)
        self.thr_right = right_lane < NUM_LANES and any(in_lane(c, right_lane) and overlap_x(c, SIDE_WARN_X) for c in self.npcs)

        self.thr_merge = False
        self.lc_rear_ttc = 9999
        if ego.on_spline:
            for car in self.npcs:
                if not in_lane(car, ego.target_lane):
                    continue
                gap_r = ex - self._tracked_state(car)["x"] - (car.length + ego.length) * 0.5
                if gap_r < 0:
                    continue
                t2 = ttc(gap_r, ego.speed, car.speed)
                self.lc_rear_ttc = min(self.lc_rear_ttc, t2)
                if gap_r < LC_REAR_MIN or t2 < TTC_LC_SLOW:
                    self.thr_merge = True
            if (ego.target_lane > ego.lane and self.thr_right) or (ego.target_lane < ego.lane and self.thr_left):
                self.thr_merge = True

        return {"leader": leader, "chaser": chaser, "front_gap": front_gap, "rear_gap": rear_gap}

    def _lane_clear(self, target_lane, ego_x):
        for car in self.npcs:
            state = self._tracked_state(car)
            if abs(state["y"] - lane_y(target_lane)) > LANE_H * 0.42:
                continue
            dx = state["x"] - ego_x
            if -LC_REAR_MIN <= dx <= 0:
                if dx > -(LC_REAR_MIN * 0.8) or ttc(-dx, self.ego.speed, car.speed) < (TTC_LC_PAUSE + 8):
                    return False
            elif 0 < dx < LC_FRONT_MIN:
                return False
            future_ego_x = ego_x + max(self.ego.speed, EGO_SPEED) * REAR_LANECHANGE_LOOKAHEAD
            future_car_x = state["x"] + car.speed * REAR_LANECHANGE_LOOKAHEAD
            future_dx = future_car_x - future_ego_x
            if future_dx <= 0:
                future_gap = -future_dx - (car.length + self.ego.length) * 0.5
                if future_gap < LC_REAR_MIN * 1.2:
                    return False
        return True

    def _path_lane_blocked(self, target_lane):
        ego = self.ego
        for car in self.npcs:
            state = self._tracked_state(car)
            if abs(state["y"] - lane_y(target_lane)) > LANE_H * 0.42:
                continue
            dx = state["x"] - ego.x
            if -SIDE_PATH_REAR_BUFFER <= dx <= SIDE_PATH_FRONT_BUFFER:
                return True
        return False

    def _can_change_to_lane(self, target_lane):
        ego = self.ego
        if target_lane < 0 or target_lane >= NUM_LANES or target_lane == ego.lane:
            return False
        if target_lane < ego.lane and self.thr_left:
            return False
        if target_lane > ego.lane and self.thr_right:
            return False
        if not self._lane_clear(target_lane, ego.x):
            return False
        if self._path_lane_blocked(target_lane):
            return False
        return True

    def _target_lane_conflict(self):
        ego = self.ego
        if not ego.on_spline:
            return False
        if self._path_lane_blocked(ego.target_lane):
            return True
        if ego.target_lane < ego.lane and self.thr_left:
            return True
        if ego.target_lane > ego.lane and self.thr_right:
            return True
        return False

    def _plan_change(self, to_lane):
        ego = self.ego
        if not self._can_change_to_lane(to_lane):
            return False
        x0, y0 = ego.x, ego.y
        y1 = lane_y(to_lane)
        dy = y1 - y0
        target = ego.overtake_target
        if target:
            tx = self._tracked_state(target)["x"]
            merge_start = clamp(tx - max(90, ego.speed * 22), x0 + 90, x0 + 170)
            target_front = tx + target.length * 0.5
            exit_x = max(target_front + max(150, ego.speed * 26), x0 + 330)
            wps = [[x0, y0], [merge_start - 40, y0], [merge_start, y0 + dy * 0.30], [tx - 40, y0 + dy * 0.82], [target_front + 40, y1], [exit_x, y1], [exit_x + 120, y1]]
        else:
            run = clamp(170 + ego.speed * 28, 170, 320)
            wps = [[x0, y0], [x0 + run * 0.18, y0 + dy * 0.06], [x0 + run * 0.42, y0 + dy * 0.45], [x0 + run * 0.82, y0 + dy * 0.96], [x0 + run, y1], [x0 + run + 80, y1]]
        ego.path = build_spline(wps)
        ego.path_idx = 0
        ego.on_spline = True
        ego.spline_pause = False
        ego.target_lane = to_lane
        ego.blinker = 1 if to_lane > ego.lane else -1
        self.sp_preview = [(p[0], p[1]) for p in ego.path]
        return True

    def _apply_brake(self, leader, front_gap, rear_threat):
        ego = self.ego
        leader_speed = leader.speed if leader else 0.0
        if front_gap < EMERGENCY_GAP:
            decel = BRAKE_DECEL_EMERG
            target = 0.0 if front_gap < ego.length * 0.45 else max(leader_speed - 0.7, 0.0)
            intensity = 1.0
        elif front_gap < SAFE_GAP:
            decel = BRAKE_DECEL_HARD
            target = max(leader_speed - 0.2, 0.0)
            intensity = 0.55 + 0.45 * (SAFE_GAP - front_gap) / max(SAFE_GAP - EMERGENCY_GAP, 1)
        else:
            decel = BRAKE_DECEL_SOFT
            ratio = (front_gap - SAFE_GAP) / max(WARN_GAP - SAFE_GAP, 1)
            target = EGO_SPEED * (0.55 + 0.45 * ratio)
            intensity = 0.2 + 0.3 * (1.0 - ratio)
        if rear_threat:
            target = max(target, ego.speed - decel * 0.5)
        ego.braking = True
        ego.brake_intensity = clamp(intensity, 0.0, 1.0)
        self.braking = True
        return max(max(ego.speed - decel, target), 0.0)

    def _safe_overtake_lanes(self):
        ego = self.ego
        order = list(range(ego.lane + 1, NUM_LANES)) + list(range(ego.lane - 1, -1, -1))
        return [lane for lane in order if self._can_change_to_lane(lane)]

    def _policy(self):
        ego = self.ego
        threats = self._scan_threats()
        leader = threats["leader"]
        chaser = threats["chaser"]
        front_gap = threats["front_gap"]

        ego.braking = False
        ego.brake_intensity = 0.0
        self.braking = False

        if ego.on_spline:
            if self._target_lane_conflict():
                ego.spline_pause = True
                ego.speed = max(min(ego.speed, EGO_SPEED) - 0.08, 0.8)
                self.status = "Lane change stopped, obstacle detected in target lane"
            elif self.thr_merge and self.lc_rear_ttc < TTC_LC_PAUSE:
                ego.spline_pause = True
                ego.speed = max(min(ego.speed, EGO_SPEED) - 0.06, 0.8)
                self.status = "Lane change paused for rear merge traffic"
            else:
                ego.spline_pause = False
                ego.speed = min(ego.speed + 0.12, EGO_OVERTAKE_SPD) if ego.state == "passing" else EGO_SPEED
            return

        if ego.state == "cruise":
            self.trapped = False
            if chaser and self.rear_ttc < TTC_REAR_EVADE:
                safe = self._safe_overtake_lanes()
                if safe:
                    ego.overtake_target = chaser
                    if self._plan_change(safe[0]):
                        self.n_evade += 1
                        ego.state = "overtaking"
                        self.status = f"Rear evade via lane {safe[0] + 1}"
                    else:
                        ego.overtake_target = None
                        self.trapped = True
                        ego.speed = self._apply_brake(leader, front_gap, True) if leader else min(chaser.speed + 0.8, EGO_SPEED + 2.5)
                        self.status = "Side lane blocked, cannot evade"
                else:
                    self.trapped = True
                    ego.speed = self._apply_brake(leader, front_gap, True) if leader else min(chaser.speed + 0.8, EGO_SPEED + 2.5)
                    self.status = "Rear threat trapped, braking"
                return
            if leader and front_gap < OVERTAKE_GAP:
                safe = self._safe_overtake_lanes()
                if safe:
                    ego.overtake_target = leader
                    if self._plan_change(safe[0]):
                        self.n_over += 1
                        ego.state = "overtaking"
                        self.status = f"B-spline overtake around {leader.vehicle_name}"
                    else:
                        ego.overtake_target = None
                        self.trapped = True
                        ego.speed = self._apply_brake(leader, front_gap, chaser is not None)
                        self.status = "Side lane occupied, staying behind obstacle"
                else:
                    self.trapped = True
                    ego.speed = self._apply_brake(leader, front_gap, chaser is not None)
                    self.status = "Lane blocked, braking behind obstacle"
                return
            if leader and front_gap < FOLLOW_BUFFER:
                desired_follow_speed = max(leader.speed - 0.1, 0.0)
                ego.speed = min(ego.speed, desired_follow_speed)
                if front_gap < SAFE_GAP:
                    ego.speed = self._apply_brake(leader, front_gap, chaser is not None)
                    self.status = f"Braking behind {leader.vehicle_name}"
                else:
                    self.status = f"Following {leader.vehicle_name} in same lane"
                return
            if (
                ego.lane != self.preferred_lane
                and self._can_change_to_lane(self.preferred_lane)
                and not leader
            ):
                ego.overtake_target = None
                if self._plan_change(self.preferred_lane):
                    self.status = f"Returning to keep-left lane {self.preferred_lane + 1}"
                    return
            ego.speed = EGO_SPEED
            if chaser and self.rear_ttc < REAR_PASS_HOLD_TTC:
                ego.speed = min(max(ego.speed, chaser.speed + REAR_SPEED_MARGIN), EGO_OVERTAKE_SPD)
                self.status = f"Rear vehicle closing, holding speed in lane {ego.lane + 1}"
            else:
                self.status = f"Cruising in lane {ego.lane + 1}" if not leader or front_gap >= SAFE_GAP else f"Following {leader.vehicle_name}"
        elif ego.state == "overtaking":
            ego.state = "passing"
            ego.speed = EGO_OVERTAKE_SPD
            self.status = "Passing on B-spline path"
        elif ego.state == "passing":
            passed = ego.overtake_target is None or (ego.x - ego.length * 0.5 > self._tracked_state(ego.overtake_target)["x"] + ego.overtake_target.length * 0.5 + 70)
            if passed:
                ego.state = "cruise"
                ego.overtake_target = None
                ego.speed = EGO_SPEED
                if chaser and self.rear_ttc < REAR_PASS_HOLD_TTC:
                    ego.speed = min(max(ego.speed, chaser.speed + REAR_SPEED_MARGIN), EGO_OVERTAKE_SPD)
                    self.status = f"Pass complete, keeping speed for rear traffic"
                else:
                    self.status = f"Pass complete, holding lane {ego.lane + 1}"
            else:
                ego.speed = min(ego.speed + 0.08, EGO_OVERTAKE_SPD)
                self.status = "Passing obstacle"

    def update(self):
        ego = self.ego
        if ego.on_spline:
            done = ego.step_spline()
            if done and ego.state == "overtaking":
                self._policy()
        else:
            ego.x += ego.speed

        retired_ids = []
        for npc in self.npcs:
            if ego.x - ego.length * 0.5 > npc.x + npc.length * 0.5 + 70:
                npc.passed_by_ego = True
            npc.x += npc.speed
            same_lane_now = abs(npc.y - ego.y) < LANE_H * 0.35
            rear_gap_now = ego.x - npc.x - (npc.length + ego.length) * 0.5
            if same_lane_now and 0 < rear_gap_now < 28 and npc.speed > ego.speed:
                ego.speed = min(max(ego.speed, npc.speed + REAR_SPEED_MARGIN), EGO_OVERTAKE_SPD + 0.8)
                ego.x += ego.speed * 0.35
            if npc.x < ego.x - W * 0.7:
                if npc.passed_by_ego:
                    retired_ids.append(npc.track_id)
                    continue
                self._respawn_npc(npc)
            if ego.rect.colliderect(npc.rect):
                if not getattr(npc, "collided", False):
                    self.n_collisions += 1
                    npc.collided = True
            else:
                npc.collided = False

        if retired_ids:
            self.npcs = [npc for npc in self.npcs if npc.track_id not in retired_ids]
            for track_id in retired_ids:
                self.radar_tracks.pop(track_id, None)

        all_cars = self.npcs + [ego]
        for car in self.npcs:
            min_gap = 9999
            closest = None
            for other in all_cars:
                if car is other:
                    continue
                same_lane = car.lane == (other.target_lane if other is ego and other.on_spline else other.lane)
                if not same_lane:
                    continue
                gap = other.x - car.x - (car.length + other.length) * 0.5
                if 0 < gap < min_gap:
                    min_gap = gap
                    closest = other
            if closest:
                if min_gap < 92:
                    car.speed = max(car.speed - 0.16, max(closest.speed - 0.9, 0.0))
                elif min_gap < 170:
                    car.speed = max(car.speed - 0.07, max(closest.speed - 0.2, 0.0))
                else:
                    hi = VEHICLE_TYPES[car.vehicle_type]["speed_range"][1]
                    car.speed = min(car.speed + 0.02, hi)
            else:
                hi = VEHICLE_TYPES[car.vehicle_type]["speed_range"][1]
                car.speed = min(car.speed + 0.02, hi)

        self._update_radar_tracks()
        self.cam = ego.x - 250
        self.road_off = (self.road_off + ego.speed * 1.7) % 160
        self._policy()

    def _draw_roadside(self):
        decor = self.scenario["decor"]
        self.screen.fill(C_SKY if decor != "night" else (20, 26, 48))
        pygame.draw.rect(self.screen, C_HAZE if decor != "night" else (28, 34, 58), (0, 0, W, ROAD_TOP - SHOULDER))
        pygame.draw.rect(self.screen, C_HAZE if decor != "night" else (28, 34, 58), (0, ROAD_BOT + SHOULDER, W, H - ROAD_BOT))
        for y in (ROAD_TOP - SHOULDER, ROAD_BOT):
            pygame.draw.rect(self.screen, C_SHOULDER, (0, y, W, SHOULDER))
        if decor in {"city", "bus_stop"}:
            for idx in range(6):
                x = int((idx * 240 - self.road_off * 0.8) % (W + 260)) - 120
                pygame.draw.rect(self.screen, (178, 146, 105), (x, 54, 72, 74), border_radius=4)
                pygame.draw.rect(self.screen, (242, 205, 84), (x + 18, 80, 44, 13), border_radius=4)
        elif decor == "night":
            for idx in range(8):
                x = int((idx * 190 - self.road_off * 0.5) % (W + 200)) - 80
                pygame.draw.circle(self.screen, (255, 216, 110), (x, 58), 5)
                pygame.draw.line(self.screen, (72, 76, 90), (x, 58), (x, ROAD_TOP - 8), 2)
        else:
            for idx in range(8):
                x = int((idx * 180 - self.road_off * 0.6) % (W + 220)) - 100
                pygame.draw.circle(self.screen, (84, 154, 92), (x, 72), 28)
                pygame.draw.rect(self.screen, (96, 78, 55), (x - 5, 72, 10, 34))
        if decor == "bus_stop":
            pygame.draw.rect(self.screen, (61, 106, 164), (70, ROAD_TOP - 58, 105, 44), border_radius=8)
            self.screen.blit(self.font_s.render("CITY BUS STOP", True, C_WHITE), (82, ROAD_TOP - 42))
        if decor == "monsoon":
            for idx in range(5):
                px = int((idx * 250 - self.road_off) % (W + 260)) - 120
                pygame.draw.ellipse(self.screen, (92, 112, 134), (px, ROAD_BOT + 28, 120, 22))

    def _draw_road(self):
        self._draw_roadside()
        for lane in range(NUM_LANES):
            col = C_ROAD if lane % 2 == 0 else C_ROAD_ALT
            pygame.draw.rect(self.screen, col, (0, ROAD_TOP + lane * LANE_H, W, LANE_H))
        pygame.draw.rect(self.screen, C_EDGE, (0, ROAD_TOP - 4, W, 6))
        pygame.draw.rect(self.screen, C_EDGE, (0, ROAD_BOT - 2, W, 6))
        dash = 54
        gap = 42
        period = dash + gap
        offset = -int(self.road_off) % period
        for div in range(1, NUM_LANES):
            y = ROAD_TOP + div * LANE_H
            x = -period + offset
            while x < W + period:
                pygame.draw.line(self.screen, C_DASH, (x, y), (x + dash, y), 3)
                x += period
        for idx in range(NUM_LANES):
            self.screen.blit(self.font_s.render(f"L{idx + 1}", True, C_TEXT_DIM), (8, lane_y(idx) - 8))

    def _smooth_alpha(self, key, active, rate_up=11, rate_dn=7, cap=110):
        self.alpha[key] = min(self.alpha[key] + rate_up, cap) if active else max(self.alpha[key] - rate_dn, 0)
        return self.alpha[key]

    def _draw_threat_zones(self):
        ego = self.ego
        ex_s = int(ego.x - self.cam)
        a = self._smooth_alpha("front", self.thr_front)
        if a > 0:
            zone = pygame.Surface((WARN_GAP, LANE_H - 10), pygame.SRCALPHA)
            zone.fill((*TC_FRONT, a))
            self.screen.blit(zone, (ex_s + ego.length // 2, int(lane_y(ego.lane) - LANE_H // 2 + 5)))
        a = self._smooth_alpha("rear", self.thr_rear)
        if a > 0:
            width = int(REAR_WARN_GAP * (0.4 + 0.6 * max(0.0, 1.0 - self.rear_ttc / TTC_REAR_WARN)))
            zone = pygame.Surface((width, LANE_H - 10), pygame.SRCALPHA)
            zone.fill((*TC_REAR, a))
            self.screen.blit(zone, (ex_s - ego.length // 2 - width, int(lane_y(ego.lane) - LANE_H // 2 + 5)))
        a = self._smooth_alpha("left", self.thr_left)
        if a > 0 and ego.lane > 0:
            zone = pygame.Surface((ego.length + SIDE_WARN_X * 2, LANE_H - 10), pygame.SRCALPHA)
            zone.fill((*TC_LEFT, a))
            self.screen.blit(zone, (ex_s - ego.length // 2 - SIDE_WARN_X, int(lane_y(ego.lane - 1) - LANE_H // 2 + 5)))
        a = self._smooth_alpha("right", self.thr_right)
        if a > 0 and ego.lane < NUM_LANES - 1:
            zone = pygame.Surface((ego.length + SIDE_WARN_X * 2, LANE_H - 10), pygame.SRCALPHA)
            zone.fill((*TC_RIGHT, a))
            self.screen.blit(zone, (ex_s - ego.length // 2 - SIDE_WARN_X, int(lane_y(ego.lane + 1) - LANE_H // 2 + 5)))
        a = self._smooth_alpha("merge", self.thr_merge)
        if a > 0 and ego.on_spline:
            zone = pygame.Surface((ego.length + 120, LANE_H - 10), pygame.SRCALPHA)
            zone.fill((*TC_MERGE, a))
            self.screen.blit(zone, (ex_s - ego.length // 2 - 60, int(lane_y(ego.target_lane) - LANE_H // 2 + 5)))
        a = self._smooth_alpha("brake", self.braking)
        if a > 0:
            width = int(WARN_GAP * (0.55 + 0.45 * ego.brake_intensity))
            zone = pygame.Surface((width, LANE_H - 10), pygame.SRCALPHA)
            for dx in range(width):
                fade = int(a * (1.0 - dx / max(width, 1)))
                pygame.draw.line(zone, (*TC_BRAKE, min(fade, 190)), (dx, 0), (dx, LANE_H - 10))
            self.screen.blit(zone, (ex_s + ego.length // 2, int(lane_y(ego.lane) - LANE_H // 2 + 5)))

    def _draw_spline_preview(self):
        if not self.show_sp or len(self.sp_preview) < 2:
            return
        pts = [(int(px - self.cam), int(py)) for px, py in self.sp_preview]
        visible = [pt for pt in pts if -20 < pt[0] < W + 20]
        if len(visible) > 1:
            pygame.draw.lines(self.screen, (255, 104, 130) if self.ego.spline_pause else (76, 219, 116), False, visible, 3)
        for pt in pts[::28]:
            if 0 <= pt[0] <= W:
                pygame.draw.circle(self.screen, (250, 249, 210), pt, 3)

    def _draw_radar_overlay(self):
        panel = pygame.Surface((314, 120), pygame.SRCALPHA)
        panel.fill((8, 12, 18, 206))
        self.screen.blit(panel, (W - 330, 92))
        self.screen.blit(self.font_m.render("RADAR TRACKING", True, C_WHITE), (W - 316, 102))
        leader = None
        leader_gap = 9999
        for car in self.npcs:
            gap = self._tracked_state(car)["x"] - self.ego.x
            if 0 < gap < leader_gap:
                leader = car
                leader_gap = gap
        if leader:
            state = self._tracked_state(leader)
            rel = leader.speed - self.ego.speed
            rows = [f"Target   : {leader.vehicle_name} T{leader.track_id}", f"Range    : {leader_gap:5.1f} px", f"Track XY : {state['x'] - self.ego.x:5.1f}, {state['y'] - self.ego.y:5.1f}", f"Vel XY   : {state['vx']:5.2f}, {state['vy']:5.2f}", f"Closing  : {rel:5.2f} px/f"]
        else:
            rows = ["Target   : none", "Range    : clear", "Track XY : -", "Vel XY   : -", "Closing  : -"]
        for idx, row in enumerate(rows):
            self.screen.blit(self.font_s.render(row, True, (182, 224, 200)), (W - 316, 130 + idx * 16))
        self.screen.blit(self.font_s.render(f"Hits this frame: {len(self.last_radar_hits)}", True, (121, 162, 236)), (W - 316, 192))
        for car in self.npcs:
            tx, ty = self._tracked_state(car)["x"], self._tracked_state(car)["y"]
            sx = int(tx - self.cam)
            if -30 <= sx <= W + 30:
                pygame.draw.rect(self.screen, (80, 250, 214), (sx - 10, int(ty - 14), 20, 28), 1, border_radius=4)
                self.screen.blit(self.font_s.render(f"T{car.track_id}", True, (78, 218, 200)), (sx - 11, int(ty - 34)))

    def _draw_speedometer(self):
        cx, cy, r = W - 110, H - 100, 74
        pygame.draw.circle(self.screen, (24, 26, 30), (cx, cy), r + 5)
        pygame.draw.circle(self.screen, (54, 56, 62), (cx, cy), r + 5, 3)
        pygame.draw.circle(self.screen, (11, 13, 16), (cx, cy), r)
        for idx in range(21):
            ang = math.radians(-225 + idx * (270 / 20))
            inner = r - 9 if idx % 5 == 0 else r - 5
            x1 = cx + math.cos(ang) * (r - 2)
            y1 = cy + math.sin(ang) * (r - 2)
            x2 = cx + math.cos(ang) * inner
            y2 = cy + math.sin(ang) * inner
            pygame.draw.line(self.screen, C_WHITE if idx % 5 == 0 else (92, 92, 96), (x1, y1), (x2, y2), 2 if idx % 5 == 0 else 1)
        ang = math.radians(-225 + min(self.ego.speed / (EGO_OVERTAKE_SPD + 2.0), 1.0) * 270)
        nx = cx + math.cos(ang) * (r - 15)
        ny = cy + math.sin(ang) * (r - 15)
        needle = C_RED if self.braking else C_ORANGE if self.ego.state == "passing" else (255, 94, 58)
        pygame.draw.line(self.screen, needle, (cx, cy), (nx, ny), 4)
        pygame.draw.circle(self.screen, (65, 68, 75), (cx, cy), 8)
        spd = self.font_l.render(f"{self.ego.speed * 20:.0f}", True, C_WHITE)
        self.screen.blit(spd, (cx - spd.get_width() // 2, cy + 12))
        self.screen.blit(self.font_s.render("km/h", True, C_TEXT_DIM), (cx - 18, cy + 43))

    def _banner(self, text, color, y):
        bw = pygame.Surface((360, 34), pygame.SRCALPHA)
        bw.fill((*color, 214))
        title = self.font_m.render(text, True, C_WHITE)
        bw.blit(title, (bw.get_width() // 2 - title.get_width() // 2, 6))
        self.screen.blit(bw, (12, y))

    def _draw_hud(self):
        panel = pygame.Surface((368, 242), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 190))
        self.screen.blit(panel, (12, 12))
        self.screen.blit(self.font_m.render(self.scenario["name"], True, C_WHITE), (20, 18))
        self.screen.blit(self.font_s.render(self.scenario["description"], True, (186, 191, 200)), (20, 42))
        state_color = {"cruise": C_GREEN, "overtaking": C_AMBER, "passing": C_ORANGE}.get(self.ego.state, C_WHITE)
        if self.braking:
            state_color = C_RED
        rows = [("Status", self.status, C_WHITE), ("State", self.ego.state.upper(), state_color), ("Lane", f"L{self.ego.lane + 1} / keep-left L{self.preferred_lane + 1}", C_WHITE), ("Speed", f"{self.ego.speed * 20:.0f} km/h", C_WHITE), ("Rear TTC", "clear" if self.rear_ttc >= 9999 else f"{self.rear_ttc:.0f} f", C_PURPLE if self.rear_ttc < TTC_REAR_WARN else C_GREEN), ("LC TTC", "-" if self.lc_rear_ttc >= 9999 else f"{self.lc_rear_ttc:.0f} f", C_RED if self.lc_rear_ttc < TTC_LC_PAUSE else C_AMBER if self.lc_rear_ttc < TTC_LC_SLOW else C_GREEN), ("Brake", f"{self.ego.brake_intensity * 100:.0f}%" if self.braking else "off", C_RED if self.braking else C_GREEN), ("Scenarios", f"{self.scenario_index + 1}/{len(SCENARIOS)}", C_WHITE)]
        for idx, (label, value, color) in enumerate(rows):
            self.screen.blit(self.font_s.render(f"{label:<10}", True, C_TEXT_DIM), (20, 74 + idx * 20))
            self.screen.blit(self.font_s.render(value, True, color), (136, 74 + idx * 20))
        indicators = [("FWD", self.thr_front, TC_FRONT), ("REAR", self.thr_rear, TC_REAR), ("LEFT", self.thr_left, TC_LEFT), ("RIGHT", self.thr_right, TC_RIGHT), ("MERGE", self.thr_merge, TC_MERGE), ("BRAKE", self.braking, TC_BRAKE)]
        ix = 20
        for label, active, color in indicators:
            box = pygame.Surface((50, 22), pygame.SRCALPHA)
            box.fill((*color, 220 if active else 48))
            self.screen.blit(box, (ix, 237))
            self.screen.blit(self.font_s.render(label, True, C_WHITE if active else (96, 98, 104)), (ix + 4, 239))
            ix += 56
        by = 270
        if self.thr_front:
            self._banner("Front obstacle risk", TC_FRONT, by)
            by += 38
        if self.thr_rear:
            self._banner(f"Rear TTC {self.rear_ttc:.0f} frames", TC_REAR, by)
            by += 38
        if self.thr_merge:
            self._banner("Lane change conflict", TC_MERGE, by)
            by += 38
        if self.braking:
            self._banner("Emergency braking active" if self.ego.brake_intensity > 0.85 else "Controlled braking active", TC_BRAKE, by)
        foot = self.font_s.render("N/P or 1-5: scenarios   S: spline   R: reset   ESC: quit", True, (75, 78, 86))
        self.screen.blit(foot, (W // 2 - foot.get_width() // 2, H - 18))
        count = self.font_s.render(f"Overtakes {self.n_over}   Evades {self.n_evade}   Collisions {self.n_collisions}", True, C_GREEN if self.n_collisions == 0 else C_RED)
        self.screen.blit(count, (12, H - 40))

    def draw(self):
        self._draw_road()
        self._draw_threat_zones()
        self._draw_spline_preview()
        for car in sorted(self.npcs + [self.ego], key=lambda c: c.y):
            car.draw(self.screen, self.cam)
        self._draw_hud()
        self._draw_radar_overlay()
        self._draw_speedometer()
        pygame.display.flip()

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()
                    if ev.key == pygame.K_s:
                        self.show_sp = not self.show_sp
                    elif ev.key == pygame.K_r:
                        self._reset()
                    elif ev.key == pygame.K_n:
                        self._reset(self.scenario_index + 1)
                    elif ev.key == pygame.K_p:
                        self._reset(self.scenario_index - 1)
                    elif pygame.K_1 <= ev.key <= pygame.K_5:
                        self._reset(ev.key - pygame.K_1)
            self.update()
            self.draw()
            self.clock.tick(FPS)


if __name__ == "__main__":
    Sim().run()

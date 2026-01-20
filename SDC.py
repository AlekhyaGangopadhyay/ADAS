import pygame
import numpy as np
import random
from scipy.interpolate import splprep, splev

# Constants
# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
ROAD_WIDTH = 600
LANE_COUNT = 4
LANE_WIDTH = ROAD_WIDTH // LANE_COUNT
ROAD_LEFT = WIDTH // 2 - ROAD_WIDTH // 2
ROAD_RIGHT = WIDTH // 2 + ROAD_WIDTH // 2
LANE_CENTERS = [ROAD_LEFT + LANE_WIDTH // 2 + i * LANE_WIDTH for i in range(LANE_COUNT)]

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (34, 139, 34)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("B-Spline Overtaking SDC - 4 Lane")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 18)

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, color, speed, main=False):
        super().__init__()
        self.image = pygame.Surface((40, 80), pygame.SRCALPHA) # Enable Alpha for rotation
        self.image.fill(color)
        self.original_image = self.image.copy() # Store original for rotation
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = speed
        self.main = main
        # Determine lane based on x position
        self.lane = 0
        min_dist = float('inf')
        for i, center in enumerate(LANE_CENTERS):
             dist = abs(x - center)
             if dist < min_dist:
                 min_dist = dist
                 self.lane = i

    def update(self):
        # Move down for traffic (simulating relative speed)
        # Player car's movement is controlled by Main loop logic
        pass

class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        size = random.randint(10, 20)
        self.image = pygame.Surface((size, size))
        self.image.fill((200, 200, 200)) # Grey smoke
        self.rect = self.image.get_rect(center=(x, y))
        self.alpha = 255
        self.life = 30
        self.image.set_alpha(self.alpha)

    def update(self):
        self.life -= 1
        self.alpha -= 8
        if self.alpha < 0: self.alpha = 0
        self.image.set_alpha(self.alpha)
        self.rect.y += 2 # Smoke strictly falls back relative to world
        if self.life <= 0:
            self.kill()

class PlayerCar(Car):
    def __init__(self, x, y):
        super().__init__(x, y, RED, 0, main=True)
        self.state = "CRUISE" # CRUISE, FOLLOW, OVERTAKE
        self.target_speed = 10
        self.current_speed = 0
        self.acceleration = 0.2
        self.path_points = []
        self.path_index = 0
        self.target_lane_idx = 0
        self.angle = 0 # Rotation angle
        self.smoke_group = pygame.sprite.Group() # Particles
        
        # Determine initial lane
        self.lane_idx = 0
        min_dist = float('inf')
        for i, center in enumerate(LANE_CENTERS):
             dist = abs(x - center)
             if dist < min_dist:
                 min_dist = dist
                 self.lane_idx = i
        
        # Sensor
        self.sensor_dist = 300
        self.detected_obj = None

    def get_lane_x(self, lane_idx):
        return ROAD_LEFT + LANE_WIDTH // 2 + lane_idx * LANE_WIDTH

    def is_lane_free(self, target_lane, traffic_group):
        # Allow driving on edges (0.5, 1.5, etc) but keep within road bounds
        # Max index is LANE_COUNT - 1. So valid range is roughly [-0.5, LANE_COUNT - 0.5]? 
        # Actually simplest is 0 to LANE_COUNT-1. 
        # Lane splitting means going to e.g. 0.5.
        if target_lane < 0 or target_lane >= LANE_COUNT:
            return False
            
        target_x = self.get_lane_x(target_lane)
        
        # Check wide area: Width 70 (Car 40), Look Ahead 300, Look Behind 300
        # If splitting (on line), we are effectively occupying TWO lanes partially?
        # Or just checking that specific narrow strip?
        # Let's check a slightly narrower width if splitting to allow squeeze? 
        # Or same width to be safe.
        check_width = 60 if isinstance(target_lane, float) and not target_lane.is_integer() else 70
        
        check_rect = pygame.Rect(target_x - check_width//2, self.rect.top - 300, check_width, 600) 
        for car in traffic_group:
            if car.rect.colliderect(check_rect):
                return False
        return True

    def find_overtake_lane(self, traffic_group):
        # Priority:
        # 1. Standard adjacent lanes (Left/Right)
        # 2. Split lanes (Left/Right dividers)
        
        current_lane = self.lane_idx
        options = []
        
        # Standard Lanes (Integers)
        # Check immediate integer neighbors
        # If we are integer, check -1, +1
        # If we are float (split), check floor and ceil (merge back)
        
        if isinstance(current_lane, int) or current_lane.is_integer():
            # We are in a lane
            c = int(current_lane)
            # Standard moves
            if c > 0: options.append(c - 1)
            if c < LANE_COUNT - 1: options.append(c + 1)
            
            # Splitting moves (only if standard failure, checked later? No, let's mix them or check logic order)
            # User implies usage "if enough space is there let my car go over the len deviding line"
            # It's a fallback.
            split_options = []
            if c > 0: split_options.append(c - 0.5)
            if c < LANE_COUNT - 1: split_options.append(c + 0.5)
            
            # First pass: Standard
            for lane in options:
                if self.is_lane_free(lane, traffic_group):
                    return lane
            
            # Second pass: Split
            for lane in split_options:
                if self.is_lane_free(lane, traffic_group):
                    return lane
                    
        else:
            # We are splitting (e.g. 1.5)
            # Priority: Merge back into 1 or 2
            floor_lane = int(current_lane)
            ceil_lane = floor_lane + 1
            
            # Try merging back first
            merge_options = [floor_lane, ceil_lane]
            for lane in merge_options:
                 if self.is_lane_free(lane, traffic_group):
                     return lane
            
            # If can't merge back, maybe switch to other split? (Unlikely to jump 1.5 -> 0.5 directly check dist)
            # Just stay or keep looking.
            
        return None

    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def drive(self, traffic_group):
        # Update Smoke
        self.smoke_group.update()

        # Default Acceleration
        accel_rate = self.acceleration

        # Sensor Logic (Raycast forward)
        self.detected_obj = None
        closest_dist = self.sensor_dist
        
        # Simple box cast ahead
        sensor_rect = pygame.Rect(self.rect.left, self.rect.top - self.sensor_dist, self.rect.width, self.sensor_dist)
        
        for car in traffic_group:
            if car != self and car.rect.colliderect(sensor_rect):
                dist = self.rect.top - car.rect.bottom
                if dist < closest_dist:
                    closest_dist = dist
                    self.detected_obj = car

        # State Machine logic to set target_speed
        if self.state == "CRUISE":
            self.target_speed = 10
            self.angle = 0 # Reset angle
            self.rotate()
            
            # Auto-Merge back if splitting (on float lane)
            if isinstance(self.lane_idx, float) and not self.lane_idx.is_integer():
                # We are splitting, try to merge back to standard lane
                floor_lane = int(self.lane_idx)
                ceil_lane = floor_lane + 1
                
                # Check possibilities (Prefer continuing straight-ish or just any empty one)
                if self.is_lane_free(floor_lane, traffic_group):
                    self.plan_overtake(floor_lane)
                elif self.is_lane_free(ceil_lane, traffic_group):
                    self.plan_overtake(ceil_lane)

            if self.detected_obj:
                # If too close, switch to FOLLOW or OVERTAKE
                if closest_dist < 150:
                    target_lane = self.find_overtake_lane(traffic_group)
                    
                    if target_lane is not None:
                        self.plan_overtake(target_lane)
                    else:
                        self.state = "FOLLOW"
        
        elif self.state == "FOLLOW":
            self.angle = 0 # Reset angle
            self.rotate()
            if self.detected_obj:
                # Safety Gaps
                safe_gap = 140
                critical_gap = 80
                
                if closest_dist < critical_gap:
                    # EMERGENCY BRAKING
                    self.target_speed = 0 # Aim for stop
                    accel_rate = 1.0 # Brake 5x harder than normal
                elif closest_dist < safe_gap:
                    self.target_speed = self.detected_obj.speed - 2 # Slow down to widen gap
                    accel_rate = 0.5 # Braking slightly harder
                else:
                    self.target_speed = self.detected_obj.speed # Match speed
                
                # Check for Overtake Opportunity
                target_lane = self.find_overtake_lane(traffic_group)
                
                if target_lane is not None:
                    # Only overtake if we aren't in critical danger
                    # (Avoid swerving while slamming brakes)
                    if closest_dist > 50:
                        self.plan_overtake(target_lane)
                else:
                    # Blocked: Strict braking handled above (target=0)
                    if closest_dist < critical_gap:
                         self.target_speed = 0
            else:
                self.state = "CRUISE"
                
        elif self.state == "OVERTAKE":
            self.target_speed = 12 # Speed up to overtake
            if self.path_index < len(self.path_points):
                target_pt = self.path_points[self.path_index]
                
                # Calculate simple rotation based on x movement
                dx = target_pt[0] - self.rect.centerx
                # Max angle clamp
                target_angle = -dx * 4.0 
                # Smoothing
                self.angle += (target_angle - self.angle) * 0.2
                self.rotate()

                # Spawn Smoke if drifting hard
                if abs(self.angle) > 10:
                    # Simple offset to rear tires (approx)
                    offset_x = -15 if self.angle > 0 else 15
                    spawn_x = self.rect.centerx + offset_x
                    spawn_y = self.rect.bottom - 10
                    self.smoke_group.add(Particle(spawn_x, spawn_y))

                self.rect.centerx = target_pt[0]
                self.path_index += 1
            else:
                self.state = "CRUISE"
                self.lane_idx = self.target_lane_idx # Update lane index

        # Apply Speed Update with dynamic accel_rate
        if self.current_speed < self.target_speed:
            self.current_speed += accel_rate
        elif self.current_speed > self.target_speed:
            self.current_speed -= accel_rate
        
        # Clamp speed
        if self.current_speed < 0: self.current_speed = 0

    def plan_overtake(self, target_lane):
        self.state = "OVERTAKE"
        self.target_lane_idx = target_lane
        start_x, start_y = self.rect.center
        end_x = self.get_lane_x(target_lane)
        # We want the lane change to happen over some distance 'd'
        # e.g. 300 pixels forward in "world space"
        # Since screen doesn't scroll PLAYER, but the WORLD scrolls, 
        # we can simulate the "time" of the maneuver.
        
        # B-Spline Control Points
        # 1. Current Pos
        # 2. Slightly forward in current lane
        # 3. Slightly backward from target in target lane
        # 4. Target pos
        
        # World Y coords (relative to start)
        # Note: In screen coords, Y is fixed for player mostly.
        # But for the curve generation, we treat Y as "forward distance".
        
        # Let's generate points in (x, t) where t is time/progress steps
        # Control points:
        y_dist = 40 # Reduced frametime for sharper, faster drift overtake
        
        # P0: Start
        p0 = (start_x, 0)
        # P1: Start Tangent (Straight ahead) -> Keep x same, advance t
        p1 = (start_x, y_dist * 0.3)
        # P2: End Tangent (Straight ahead at target) -> Target x, backward t
        p2 = (end_x, y_dist * 0.7)
        # P3: End
        p3 = (end_x, y_dist)
        
        x_pts = [p0[0], p1[0], p2[0], p3[0]]
        t_pts = [p0[1], p1[1], p2[1], p3[1]]
        
        # Fit B-Spline
        tck, u = splprep([x_pts, t_pts], k=3, s=0)
        new_points = splev(np.linspace(0, 1, y_dist), tck)
        
        # Store (x, y) relative to screen?
        # Actually we just need X for each frame of the maneuver.
        self.path_points = []
        for i in range(len(new_points[0])):
            px = new_points[0][i]
            # py = new_points[1][i] # Not needed if we just step through per frame
            self.path_points.append((px, start_y)) # Keep Y same on screen

        self.path_index = 0

def main():
    player = PlayerCar(LANE_CENTERS[1], 500)
    
    traffic_group = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()
    all_sprites.add(player)
    
    # Road scrolling vars
    road_y = 0
    bg_speed = 0
    
    running = True
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main() # Reset (simple)
                    return

    # Initial Traffic Initialization (Total 4 traffic cars + 1 Player = 5 cars)
    for _ in range(4):
        # Find valid spawn
        while True:
            lane = random.choice(LANE_CENTERS)
            spawn_y = random.randint(-1200, -100) # Spread out initially (More space for more cars)
            collision = False
            for t in traffic_group:
                if abs(t.rect.y - spawn_y) < 200 and abs(t.rect.centerx - lane) < 50:
                    collision = True
            if not collision:
                t_car = Car(lane, spawn_y, BLUE, random.randint(4, 9)) # Various velocity
                traffic_group.add(t_car)
                all_sprites.add(t_car)
                break
    
    running = True
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main() # Reset
                    return

        # Recycle Traffic (Keep same 4 cars)
        for car in traffic_group:
            # If car falls behind (goes off bottom of screen)
            if car.rect.top > HEIGHT:
                # Cycle it to top (ahead of player)
                # Find new valid spot
                reset_success = False
                attempts = 0
                while not reset_success and attempts < 20: 
                    new_lane = random.choice(LANE_CENTERS)
                    new_y = random.randint(-800, -100) # Expanded recycle range
                    collision = False
                    for other in traffic_group:
                        if other != car and abs(other.rect.y - new_y) < 200 and abs(other.rect.centerx - new_lane) < 50:
                            collision = True
                    
                    if not collision:
                        car.rect.center = (new_lane, new_y)
                        car.lane = 0 # Updates not needed as checks are rect based, but good practice
                        car.speed = random.randint(4, 9) # Various velocity on recycle
                        reset_success = True
                    attempts += 1
                
                # If crowded, just push further back
                if not reset_success:
                     car.rect.y = -1000

            # If car gets too far ahead (rect.bottom < -600)? 
            # In this logic (relative speed), if car is faster than player, it moves UP.
            # If it moves off TOP, it is "Gone". Recycle to BOTTOM?
            if car.rect.bottom < -600:
                car.rect.y = HEIGHT + 100 # Reset to behind? 
                pass

        # Update Logic
        player.drive(traffic_group)
        bg_speed = player.current_speed
        
        # Scroll Road (simulate movement)
        road_y += bg_speed
        if road_y >= HEIGHT:
            road_y = 0
            
        # Move Traffic (Relative speed)
        for car in traffic_group:
            car.rect.y += (player.current_speed - car.speed)
            
            # Legacy kill checks (should not trigger if recycle works, but safe to keep/modify)
            # Actually, recycle handles > HEIGHT. 
            # We must ensure we don't Kill() anymore if we want fixed set.
            # Removing kill logic.

        # Drawing
        screen.fill(GREEN) # Grass
        
        # Draw Road
        pygame.draw.rect(screen, GRAY, (ROAD_LEFT, 0, ROAD_WIDTH, HEIGHT))
        
        # Draw Lane Markers
        # Moving dashed line
        marker_y = road_y % 40
        for lane_i in range(1, LANE_COUNT):
            line_x = ROAD_LEFT + lane_i * LANE_WIDTH
            for i in range(-1, HEIGHT // 40 + 2):
                pygame.draw.rect(screen, WHITE, (line_x - 2, i * 40 + marker_y, 4, 20))
            
        # Draw Smoke under cars
        player.smoke_group.draw(screen)

        all_sprites.draw(screen)
        
        # Visualize BoxCast (Debug)
        if player.state in ["CRUISE", "FOLLOW"]:
             pygame.draw.rect(screen, (255, 255, 0), (player.rect.left, player.rect.top - player.sensor_dist, player.rect.width, player.sensor_dist), 1)

        # UI
        status_text = font.render(f"State: {player.state} | Speed: {player.current_speed:.1f}", True, BLACK)
        screen.blit(status_text, (10, 10))
        
        if player.detected_obj:
            warn_text = font.render("OBSTACLE DETECTED", True, RED)
            screen.blit(warn_text, (WIDTH//2 - 100, HEIGHT - 50))
            
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

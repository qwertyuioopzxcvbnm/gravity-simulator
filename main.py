import pygame
import math
import random
import asyncio

pygame.init()

# Screen setup
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gravity Simulator")

# Actual physical constants (SI units)
G = 6.67430e-11  # Gravitational constant (m³/(kg·s²))
C = 299792458  # Speed of light (m/s)
C_SQUARED = C ** 2  # Speed of light squared (m²/s²)

# Astronomical constants
SOLAR_MASS = 1.989e30  # kg
EARTH_MASS = 5.972e24  # kg
AU = 1.496e11  # meters (Earth-Sun distance)

# Display scaling factors
METERS_PER_PIXEL = 1e9  # 1 pixel = 1 billion meters (1 AU ≈ 150 pixels)
SECONDS_PER_STEP = 86400  # Each simulation step = 1 day of real time
DT = 1.0  # Simulation substep multiplier
TRAIL_LENGTH = 200

# Colors
COLORS = {
    'background': (5, 5, 15),
    'text': (200, 200, 200),
    'text_dim': (100, 100, 120),
    'trail_fade': 0.6,
}

# Fonts
font = pygame.font.SysFont('Monaco', 14)
font_large = pygame.font.SysFont('Monaco', 18)


class Camera:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2

    def world_to_screen(self, wx, wy):
        sx = wx - self.x + WIDTH // 2
        sy = wy - self.y + HEIGHT // 2
        return sx, sy

    def screen_to_world(self, sx, sy):
        wx = sx - WIDTH // 2 + self.x
        wy = sy - HEIGHT // 2 + self.y
        return wx, wy


class Star:
    """Background star for visual effect"""
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.brightness = random.randint(50, 150)
        self.size = random.choice([1, 1, 1, 2])

    def draw(self, surface):
        color = (self.brightness, self.brightness, self.brightness + 20)
        if self.size == 1:
            surface.set_at((self.x, self.y), color)
        else:
            pygame.draw.circle(surface, color, (self.x, self.y), self.size)


class Object:
    def __init__(self, position, mass, velocity, color=(255, 0, 0), name="", is_black_hole=False, display_radius=None):
        # Position in meters (world coordinates)
        self.x, self.y = position
        # Mass in kg
        self.mass = mass
        # Velocity in m/s
        self.dx, self.dy = velocity
        self.color = color
        self.name = name
        self.trail = []
        self.trail_color = tuple(max(30, c // 2) for c in color)
        self.is_black_hole = is_black_hole
        # Display radius in pixels (for visualization, not physics)
        self._display_radius = display_radius
        if is_black_hole:
            # Schwarzschild radius: r_s = 2GM/c² (in meters)
            self.schwarzschild_radius = (2 * G * mass) / C_SQUARED

    def draw(self, surface, camera):
        # Convert world position (meters) to screen position (pixels)
        screen_x = self.x / METERS_PER_PIXEL
        screen_y = self.y / METERS_PER_PIXEL
        sx, sy = camera.world_to_screen(screen_x, screen_y)

        if self.is_black_hole:
            # Black hole rendering with Schwarzschild radius (converted to pixels)
            event_horizon_radius = max(5, int(self.schwarzschild_radius / METERS_PER_PIXEL))

            # Draw trail
            if len(self.trail) > 1:
                trail_points = []
                for i, (tx, ty) in enumerate(self.trail):
                    # Convert trail positions from meters to pixels
                    tsx, tsy = camera.world_to_screen(tx / METERS_PER_PIXEL, ty / METERS_PER_PIXEL)
                    trail_points.append((tsx, tsy))
                for i in range(len(trail_points) - 1):
                    alpha = int(255 * (i / len(trail_points)) * COLORS['trail_fade'])
                    trail_color = (alpha // 4, alpha // 8, alpha // 2)
                    pygame.draw.line(surface, trail_color, trail_points[i], trail_points[i + 1], 1)

            # Draw gravitational lensing effect (outer distortion rings)
            for i in range(5, 0, -1):
                ring_radius = event_horizon_radius + i * 8
                ring_alpha = 60 - i * 10
                ring_color = (ring_alpha // 2, ring_alpha // 3, ring_alpha)
                pygame.draw.circle(surface, ring_color, (int(sx), int(sy)), int(ring_radius))

            # Draw accretion disk (orange/yellow ring around event horizon)
            accretion_outer = event_horizon_radius + 15
            accretion_inner = event_horizon_radius + 5
            for r in range(int(accretion_outer), int(accretion_inner), -1):
                intensity = int(255 * (r - accretion_inner) / (accretion_outer - accretion_inner))
                disk_color = (255, 150 - intensity // 3, intensity // 4)
                pygame.draw.circle(surface, disk_color, (int(sx), int(sy)), r, 1)

            # Draw photon sphere (bright ring at 1.5x Schwarzschild radius)
            photon_sphere_radius = int(event_horizon_radius * 1.5)
            pygame.draw.circle(surface, (255, 200, 100), (int(sx), int(sy)), photon_sphere_radius, 2)

            # Draw event horizon (pure black)
            pygame.draw.circle(surface, (0, 0, 0), (int(sx), int(sy)), event_horizon_radius)

            # Draw event horizon border
            pygame.draw.circle(surface, (80, 40, 120), (int(sx), int(sy)), event_horizon_radius, 2)
        else:
            # Use display radius if set, otherwise calculate from mass (log scale for visibility)
            if self._display_radius:
                radius = self._display_radius
            else:
                # Log scale for mass visualization (since masses span many orders of magnitude)
                radius = max(2, int(3 + 3 * math.log10(max(1, self.mass / EARTH_MASS))))

            # Draw trail
            if len(self.trail) > 1:
                trail_points = []
                for i, (tx, ty) in enumerate(self.trail):
                    # Convert trail positions from meters to pixels
                    tsx, tsy = camera.world_to_screen(tx / METERS_PER_PIXEL, ty / METERS_PER_PIXEL)
                    trail_points.append((tsx, tsy))

                # Draw trail segments with fading
                for i in range(len(trail_points) - 1):
                    alpha = int(255 * (i / len(trail_points)) * COLORS['trail_fade'])
                    trail_color = (
                        min(255, self.trail_color[0] + alpha // 4),
                        min(255, self.trail_color[1] + alpha // 4),
                        min(255, self.trail_color[2] + alpha // 4),
                    )
                    pygame.draw.line(surface, trail_color, trail_points[i], trail_points[i + 1], 1)

            # Draw glow effect
            for i in range(3, 0, -1):
                glow_radius = radius + i * 3
                glow_alpha = 30 - i * 8
                glow_color = (
                    min(255, self.color[0] // 4 + glow_alpha),
                    min(255, self.color[1] // 4 + glow_alpha),
                    min(255, self.color[2] // 4 + glow_alpha),
                )
                pygame.draw.circle(surface, glow_color, (int(sx), int(sy)), int(glow_radius))

            # Draw main body
            pygame.draw.circle(surface, self.color, (int(sx), int(sy)), radius)

            # Draw highlight
            highlight_offset = radius // 3
            highlight_radius = max(1, radius // 3)
            highlight_color = tuple(min(255, c + 80) for c in self.color)
            pygame.draw.circle(surface, highlight_color,
                              (int(sx - highlight_offset), int(sy - highlight_offset)),
                              highlight_radius)

    def radius(self):
        """Return collision radius in meters"""
        if self.is_black_hole:
            return self.schwarzschild_radius
        # Use a physical radius estimate based on mass (very rough approximation)
        # For visualization, use display radius scaled back to meters
        if self._display_radius:
            return self._display_radius * METERS_PER_PIXEL
        # Default: scale based on mass relative to Earth
        return max(1e7, (self.mass / EARTH_MASS) ** 0.33 * 6.371e6)  # Earth radius as reference

    def check_collision(self, other):
        rx = other.x - self.x
        ry = other.y - self.y
        distance = math.sqrt(rx**2 + ry**2)
        return distance < (self.radius() + other.radius())

    def merge(self, other):
        total_mass = self.mass + other.mass
        # Conserve momentum (p = mv)
        self.dx = (self.mass * self.dx + other.mass * other.dx) / total_mass
        self.dy = (self.mass * self.dy + other.mass * other.dy) / total_mass
        # Center of mass position
        self.x = (self.mass * self.x + other.mass * other.x) / total_mass
        self.y = (self.mass * self.y + other.mass * other.y) / total_mass
        # If either is a black hole, result is a black hole
        if self.is_black_hole or other.is_black_hole:
            self.is_black_hole = True
            self.color = (0, 0, 0)
            self._display_radius = None  # Black holes use Schwarzschild radius
        else:
            # Blend colors based on mass
            self.color = tuple(
                int((self.mass * self.color[i] + other.mass * other.color[i]) / total_mass)
                for i in range(3)
            )
            # Update display radius based on larger object
            if self._display_radius and other._display_radius:
                self._display_radius = max(self._display_radius, other._display_radius) + 1
        self.trail_color = tuple(max(30, c // 2) for c in self.color)
        self.mass = total_mass
        # Update Schwarzschild radius if black hole
        if self.is_black_hole:
            self.schwarzschild_radius = (2 * G * self.mass) / C_SQUARED
        # Combine trails
        self.trail.extend(other.trail[-50:])

    def apply_gravity(self, other, dt):
        rx = other.x - self.x
        ry = other.y - self.y
        distance = math.sqrt(rx**2 + ry**2)
        # Minimum distance to prevent numerical instability (in meters)
        min_distance = max(self.radius(), other.radius()) * 2
        if distance < min_distance:
            return
        # Newton's law: F = GMm/r², a = GM/r²
        acceleration = G * other.mass / (distance ** 2)
        self.dx += acceleration * rx / distance * dt
        self.dy += acceleration * ry / distance * dt

    def update(self, dt):
        self.x += self.dx * dt
        self.y += self.dy * dt
        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.dx**2 + self.dy**2)

    def speed(self):
        return math.sqrt(self.dx**2 + self.dy**2)


def create_binary_system(center_x, center_y):
    """Create a stable binary star system using real physics

    Two sun-like stars orbiting their common center of mass.
    Separation ~0.5 AU for a tight binary system.
    """
    # Convert screen center to world coordinates (meters)
    world_center_x = center_x * METERS_PER_PIXEL
    world_center_y = center_y * METERS_PER_PIXEL

    star_mass = 1.0 * SOLAR_MASS  # 1 solar mass each
    separation = 0.5 * AU  # 0.5 AU between stars

    # Orbital velocity for binary system with equal masses:
    # Each star orbits at r = separation/2 from center of mass
    # Gravitational force: F = G*M*M/d^2
    # Centripetal: F = M*v^2/r = 2*M*v^2/d
    # Solving: v = sqrt(G*M/(2*d))
    orbital_velocity = math.sqrt(G * star_mass / (2 * separation))

    star1 = Object(
        position=(world_center_x - separation / 2, world_center_y),
        mass=star_mass,
        velocity=(0, -orbital_velocity),
        color=(255, 200, 50),
        name="Star A",
        display_radius=12
    )
    star2 = Object(
        position=(world_center_x + separation / 2, world_center_y),
        mass=star_mass,
        velocity=(0, orbital_velocity),
        color=(100, 150, 255),
        name="Star B",
        display_radius=12
    )
    return [star1, star2]


def create_single_planet(center_x, center_y):
    """Create a Sun with Earth and Mars

    Uses actual solar system data:
    - Sun: 1 solar mass
    - Earth: 1 AU, ~30 km/s orbital velocity
    - Mars: 1.52 AU, ~24 km/s orbital velocity
    """
    world_center_x = center_x * METERS_PER_PIXEL
    world_center_y = center_y * METERS_PER_PIXEL

    sun_mass = SOLAR_MASS
    earth_distance = AU  # 1 AU
    mars_distance = 1.52 * AU  # 1.52 AU

    # Circular orbital velocity: v = sqrt(G * M / r)
    earth_velocity = math.sqrt(G * sun_mass / earth_distance)  # ~29.78 km/s
    mars_velocity = math.sqrt(G * sun_mass / mars_distance)  # ~24.1 km/s

    sun = Object(
        position=(world_center_x, world_center_y),
        mass=sun_mass,
        velocity=(0, 0),
        color=(255, 220, 50),
        name="Sun",
        display_radius=15
    )

    earth = Object(
        position=(world_center_x + earth_distance, world_center_y),
        mass=EARTH_MASS,
        velocity=(0, earth_velocity),
        color=(50, 120, 200),
        name="Earth",
        display_radius=6
    )

    mars = Object(
        position=(world_center_x - mars_distance, world_center_y),
        mass=0.107 * EARTH_MASS,  # Mars mass
        velocity=(0, -mars_velocity),
        color=(200, 100, 50),
        name="Mars",
        display_radius=4
    )

    return [sun, earth, mars]


def create_chaos(center_x, center_y):
    """Create a chaotic multi-body system

    8 star-like objects with slightly perturbed orbits around a common center.
    """
    world_center_x = center_x * METERS_PER_PIXEL
    world_center_y = center_y * METERS_PER_PIXEL

    # Total system mass for orbital calculations
    central_mass_equivalent = 8 * SOLAR_MASS

    objects = []
    for i in range(8):
        angle = (i / 8) * 2 * math.pi
        # Distance varies from ~0.8 to ~1.2 AU
        distance = AU * (1.0 + random.uniform(-0.2, 0.2))
        x = world_center_x + math.cos(angle) * distance
        y = world_center_y + math.sin(angle) * distance
        mass = SOLAR_MASS * random.uniform(0.5, 1.5)
        # Orbital speed with some randomness for chaos
        speed = math.sqrt(G * central_mass_equivalent / distance) * random.uniform(0.7, 1.3)
        vx = -math.sin(angle) * speed
        vy = math.cos(angle) * speed
        color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        objects.append(Object((x, y), mass, (vx, vy), color, f"Star {i+1}", display_radius=8))
    return objects


def create_black_hole_system(center_x, center_y):
    """Create a black hole with orbiting stars

    Uses a smaller black hole for simulation stability while still having
    a visible Schwarzschild radius at our scale.

    Schwarzschild radius: r_s = 2GM/c²
    For 100,000 solar masses: r_s ≈ 3 × 10^8 m ≈ visible at our scale
    """
    world_center_x = center_x * METERS_PER_PIXEL
    world_center_y = center_y * METERS_PER_PIXEL

    # Intermediate mass black hole: 1,000 solar masses
    # Smaller mass = more stable simulation and reasonable orbital velocities (~30-100 km/s)
    black_hole_mass = 1e3 * SOLAR_MASS

    black_hole = Object(
        position=(world_center_x, world_center_y),
        mass=black_hole_mass,
        velocity=(0, 0),
        color=(0, 0, 0),
        name="Black Hole",
        is_black_hole=True
    )

    # Add orbiting stars at visible distances (1-3 AU = 150-450 pixels on screen)
    objects = [black_hole]
    for i in range(4):
        angle = (i / 4) * 2 * math.pi
        # Orbit at 1-3 AU from center (visible on screen)
        distance = AU * (1.0 + i * 0.6)  # 1.0, 1.6, 2.2, 2.8 AU
        x = world_center_x + math.cos(angle) * distance
        y = world_center_y + math.sin(angle) * distance
        mass = SOLAR_MASS * (1 + i * 0.5)  # 1, 1.5, 2, 2.5 solar masses
        # Orbital velocity for circular orbit
        orbital_v = math.sqrt(G * black_hole_mass / distance)
        vx = -math.sin(angle) * orbital_v
        vy = math.cos(angle) * orbital_v
        colors = [
            (255, 200, 100),  # Yellow
            (100, 150, 255),  # Blue
            (255, 150, 100),  # Orange
            (200, 100, 255),  # Purple
        ]
        objects.append(Object((x, y), mass, (vx, vy), colors[i], f"Star {i+1}", display_radius=6))

    return objects


class Button:
    def __init__(self, x, y, width, height, text, color=(80, 80, 100)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = tuple(min(255, c + 40) for c in color)
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, (120, 120, 140), self.rect, 1, border_radius=4)
        text_surf = font.render(self.text, True, (220, 220, 220))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos, mouse_pressed):
        return self.rect.collidepoint(mouse_pos) and mouse_pressed


def draw_hud(surface, objects, paused, speed_mult, buttons):
    """Draw the heads-up display"""
    y_offset = 10

    # Title and status
    status = "PAUSED" if paused else "RUNNING"
    status_color = (255, 100, 100) if paused else (100, 255, 100)
    title = font_large.render(f"Gravity Simulator - {status}", True, status_color)
    surface.blit(title, (10, y_offset))
    y_offset += 25

    # Stats
    stats = [
        f"Objects: {len(objects)}",
        f"Speed: {speed_mult:.1f}x",
        f"Time step: {SECONDS_PER_STEP * speed_mult / 86400:.1f} days/frame",
    ]

    total_ke = sum(obj.kinetic_energy() for obj in objects)
    stats.append(f"Total KE: {total_ke:.2e} J")

    for stat in stats:
        text = font.render(stat, True, COLORS['text'])
        surface.blit(text, (10, y_offset))
        y_offset += 18

    # Speed control label
    speed_label = font.render("Speed:", True, COLORS['text_dim'])
    surface.blit(speed_label, (80, 133))

    # Draw buttons
    for button in buttons:
        button.draw(surface)

    # Object info on the right
    info_y = 10
    for obj in objects[:6]:  # Show max 6 objects
        # Display mass in appropriate units
        if obj.mass >= SOLAR_MASS * 0.1:
            mass_str = f"{obj.mass / SOLAR_MASS:.2f} M☉"
        else:
            mass_str = f"{obj.mass / EARTH_MASS:.2f} M⊕"
        # Display velocity in km/s
        vel_kms = obj.speed() / 1000
        info = f"{obj.name or 'Object'}: {mass_str} v={vel_kms:.1f} km/s"
        text = font.render(info, True, obj.color)
        surface.blit(text, (WIDTH - 300, info_y))
        info_y += 20


def draw_velocity_preview(surface, start_pos, end_pos, camera):
    """Draw velocity vector preview when adding new object"""
    pygame.draw.line(surface, (100, 255, 100), start_pos, end_pos, 2)
    # Draw arrowhead
    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
    arrow_len = 10
    for offset in [2.5, -2.5]:
        ax = end_pos[0] - arrow_len * math.cos(angle + offset)
        ay = end_pos[1] - arrow_len * math.sin(angle + offset)
        pygame.draw.line(surface, (100, 255, 100), end_pos, (ax, ay), 2)

    # Show velocity magnitude in km/s (1 pixel drag = 1 km/s)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    vel_kms = math.sqrt(dx**2 + dy**2)  # km/s
    vel_text = font.render(f"v={vel_kms:.1f} km/s", True, (100, 255, 100))
    surface.blit(vel_text, (end_pos[0] + 10, end_pos[1]))


async def main():
    # Initialize
    camera = Camera()
    background_stars = [Star() for _ in range(200)]
    objects = create_binary_system(WIDTH // 2, HEIGHT // 2)

    # Buttons
    clear_button = Button(10, HEIGHT - 40, 60, 28, "Clear", (120, 60, 60))
    binary_button = Button(80, HEIGHT - 40, 80, 28, "Binary", (60, 80, 120))
    planet_button = Button(170, HEIGHT - 40, 80, 28, "Planet", (60, 100, 80))
    chaos_button = Button(260, HEIGHT - 40, 80, 28, "Chaos", (100, 60, 100))
    blackhole_button = Button(350, HEIGHT - 40, 100, 28, "Black Hole", (20, 10, 40))
    # Speed control buttons (positioned under stats)
    speed_down_button = Button(10, 130, 30, 24, "-", (100, 70, 70))
    speed_up_button = Button(45, 130, 30, 24, "+", (70, 100, 70))
    buttons = [clear_button, binary_button, planet_button, chaos_button, blackhole_button, speed_down_button, speed_up_button]

    # State
    running = True
    paused = False
    speed_mult = 0.5  # Start at 0.5x speed for stability
    adding_object = False
    add_start_pos = None
    clock = pygame.time.Clock()

    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Update button hover states
        for button in buttons:
            button.update(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    speed_mult = min(50.0, speed_mult + 0.5)
                elif event.key == pygame.K_MINUS:
                    speed_mult = max(0.25, speed_mult - 0.5)
                elif event.key == pygame.K_c:
                    for obj in objects:
                        obj.trail.clear()
                elif event.key == pygame.K_1:
                    objects = create_binary_system(WIDTH // 2, HEIGHT // 2)
                elif event.key == pygame.K_2:
                    objects = create_single_planet(WIDTH // 2, HEIGHT // 2)
                elif event.key == pygame.K_3:
                    objects = create_chaos(WIDTH // 2, HEIGHT // 2)
                elif event.key == pygame.K_x or event.key == pygame.K_DELETE:
                    objects = []

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check if clicking on buttons
                    if clear_button.is_clicked(mouse_pos, True):
                        objects = []
                    elif binary_button.is_clicked(mouse_pos, True):
                        objects = create_binary_system(WIDTH // 2, HEIGHT // 2)
                    elif planet_button.is_clicked(mouse_pos, True):
                        objects = create_single_planet(WIDTH // 2, HEIGHT // 2)
                    elif chaos_button.is_clicked(mouse_pos, True):
                        objects = create_chaos(WIDTH // 2, HEIGHT // 2)
                    elif blackhole_button.is_clicked(mouse_pos, True):
                        objects = create_black_hole_system(WIDTH // 2, HEIGHT // 2)
                    elif speed_down_button.is_clicked(mouse_pos, True):
                        speed_mult = max(0.25, speed_mult - 0.5)
                    elif speed_up_button.is_clicked(mouse_pos, True):
                        speed_mult = min(50.0, speed_mult + 0.5)
                    else:
                        adding_object = True
                        add_start_pos = mouse_pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and adding_object:
                    adding_object = False
                    # Convert screen position to world coordinates (meters)
                    screen_x, screen_y = camera.screen_to_world(*add_start_pos)
                    wx = screen_x * METERS_PER_PIXEL
                    wy = screen_y * METERS_PER_PIXEL
                    # Convert pixel drag to velocity (m/s)
                    # Scale factor: pixels to km/s (1 pixel drag = 1 km/s)
                    velocity_scale = 1000  # m/s per pixel of drag
                    dx = (mouse_pos[0] - add_start_pos[0]) * velocity_scale
                    dy = (mouse_pos[1] - add_start_pos[1]) * velocity_scale
                    color = (
                        random.randint(150, 255),
                        random.randint(150, 255),
                        random.randint(150, 255),
                    )
                    new_obj = Object(
                        position=(wx, wy),
                        mass=EARTH_MASS,  # Default to Earth mass
                        velocity=(dx, dy),
                        color=color,
                        name=f"New {len(objects)+1}",
                        display_radius=5
                    )
                    objects.append(new_obj)

        screen.fill(COLORS['background'])

        for star in background_stars:
            star.draw(screen)

        if not paused:
            # Time step in seconds of simulation time
            # SECONDS_PER_STEP is how many real seconds pass per simulation step
            substeps = max(4, int(speed_mult * 8))  # More substeps for stability at higher speeds
            dt = SECONDS_PER_STEP * speed_mult / substeps

            for _ in range(substeps):
                for i, obj in enumerate(objects):
                    for j, other in enumerate(objects):
                        if i != j:
                            obj.apply_gravity(other, dt)

                # Handle collisions
                to_remove = set()
                for i, obj in enumerate(objects):
                    if obj in to_remove:
                        continue
                    for j, other in enumerate(objects):
                        if i < j and other not in to_remove and obj.check_collision(other):
                            # Bigger object absorbs the smaller one
                            if obj.mass >= other.mass:
                                obj.merge(other)
                                to_remove.add(other)
                            else:
                                other.merge(obj)
                                to_remove.add(obj)
                                break  # obj is gone, stop checking its collisions

                for obj in to_remove:
                    objects.remove(obj)

                # Update positions
                for obj in objects:
                    obj.update(dt)

        # Draw objects
        for obj in objects:
            obj.draw(screen, camera)

        if adding_object and add_start_pos:
            draw_velocity_preview(screen, add_start_pos, mouse_pos, camera)
            wx, wy = camera.screen_to_world(*add_start_pos)
            sx, sy = camera.world_to_screen(wx, wy)
            pygame.draw.circle(screen, (100, 255, 100), (int(sx), int(sy)), 15, 1)

        draw_hud(screen, objects, paused, speed_mult, buttons)

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    pygame.quit()

asyncio.run(main())

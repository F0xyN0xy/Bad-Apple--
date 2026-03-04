import sys
import os
import math
import time
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Must match exactly what you trained with ──────────────────────────────────
WIDTH      = 160
HEIGHT     = 120
FREQ_BANDS = 8
HIDDEN     = 128
FRAME_SIZE = WIDTH * HEIGHT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Positional encoding (identical to trainer) ────────────────────────────────
def pos_encode(coords: torch.Tensor, bands: int = FREQ_BANDS) -> torch.Tensor:
    out = [coords]
    for k in range(bands):
        freq = 2.0 ** k * math.pi
        out.append(torch.sin(freq * coords))
        out.append(torch.cos(freq * coords))
    return torch.cat(out, dim=-1)

INPUT_DIM = 3 + 3 * 2 * FREQ_BANDS

# ── Network (identical to trainer) ───────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
    def forward(self, x):
        return F.tanh(self.l2(F.tanh(self.l1(x)))) + x

class NeRFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([ResBlock(HIDDEN) for _ in range(3)])
        self.head   = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pos_encode(x)
        h = F.tanh(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        return torch.sigmoid(self.head(h))

# ── Args / file discovery ─────────────────────────────────────────────────────
def find_file(candidates, label):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

model_path = sys.argv[1] if len(sys.argv) > 1 else None
audio_path = sys.argv[2] if len(sys.argv) > 2 else None

model_path = find_file([model_path, "best_model.pt"], "model")
audio_path = find_file(
    [audio_path] + [f for f in os.listdir(".") if f.endswith((".mp3",".ogg",".wav",".flac"))],
    "audio"
)

if not model_path:
    print("[error] No model file found. Train first, or pass path as first argument.")
    print("        Example: python play_badapple.py best_model.pt bad_apple.mp3")
    sys.exit(1)

print(f"[info] Model : {model_path}")
print(f"[info] Audio : {audio_path or '(none — silent)'}")
print(f"[info] Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
model = NeRFNet().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print("[info] Model loaded.")

# ── Precompute pixel grid (constant across frames) ────────────────────────────
gx = np.linspace(0, 1, WIDTH,  dtype=np.float32)
gy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
mgx, mgy = np.meshgrid(gx, gy)
xy_flat = np.stack([mgx.flatten(), mgy.flatten()], axis=1)  # (H*W, 2)
xy_tensor = torch.FloatTensor(xy_flat).to(DEVICE)           # stays on device

def render_t(t_val: float) -> np.ndarray:
    """Render one frame for normalised time t_val ∈ [0,1]. Returns (H,W) uint8."""
    t_col = torch.full((FRAME_SIZE, 1), t_val, dtype=torch.float32, device=DEVICE)
    inp   = torch.cat([xy_tensor, t_col], dim=1)
    with torch.no_grad():
        out = model(inp).squeeze(1).cpu().numpy()
    return (out.reshape(HEIGHT, WIDTH) * 255).astype(np.uint8)

# ── Detect song duration ──────────────────────────────────────────────────────
# Try to read from frames.meta, fall back to pygame mixer, then hardcode
DURATION = 219.0   # Bad Apple default ~3:39

meta_path = "frames.meta"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        for line in f:
            if line.startswith("fps="):
                pass  # could calculate duration if frame count known

# We'll sync to audio position once mixer is running; duration mostly for
# the t-value when playing silent.

# ── Pygame init ───────────────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

info       = pygame.display.Info()
SCREEN_W   = info.current_w
SCREEN_H   = info.current_h
fullscreen = False

screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
pygame.display.set_caption("Bad Apple — Neural Network")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("Courier New", 14)

# ── Load audio ────────────────────────────────────────────────────────────────
audio_loaded = False
if audio_path:
    try:
        pygame.mixer.music.load(audio_path)
        # Get duration via a Sound object (mixer.music has no get_length before play)
        snd = pygame.mixer.Sound(audio_path)
        DURATION = snd.get_length()
        del snd
        print(f"[info] Duration: {DURATION:.1f}s")
        audio_loaded = True
    except Exception as e:
        print(f"[warn] Could not load audio: {e}")

# ── State ─────────────────────────────────────────────────────────────────────
paused       = False
play_start   = None      # wall-clock time when playback began (adjusted for seeks/pauses)
pause_start  = None      # wall-clock time when pause began
elapsed      = 0.0       # seconds into the animation
show_hud     = True

def current_elapsed():
    if paused:
        return elapsed
    return elapsed + (time.time() - play_start)

def seek(delta_sec):
    global elapsed, play_start
    new_elapsed = max(0.0, min(DURATION, current_elapsed() + delta_sec))
    elapsed     = new_elapsed
    play_start  = time.time()
    if audio_loaded:
        pygame.mixer.music.play(start=new_elapsed)
        if paused:
            pygame.mixer.music.pause()

def start_playback():
    global play_start, paused, elapsed
    play_start = time.time()
    paused     = False
    if audio_loaded:
        pygame.mixer.music.play(start=elapsed)

def pause_resume():
    global paused, elapsed, play_start, pause_start
    if paused:
        # Resume
        paused     = False
        play_start = time.time()
        if audio_loaded:
            pygame.mixer.music.unpause()
    else:
        # Pause
        elapsed     = current_elapsed()
        paused      = True
        if audio_loaded:
            pygame.mixer.music.pause()

# ── Start ─────────────────────────────────────────────────────────────────────
start_playback()

# Pre-render first frame so there's no black flash
last_frame = render_t(0.0)

print("[info] Playing. SPACE=pause  LEFT/RIGHT=seek  F=fullscreen  ESC=quit")

running = True
while running:

    # ── Events ────────────────────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_SPACE:
                pause_resume()
            elif event.key == pygame.K_RIGHT:
                seek(+5.0)
            elif event.key == pygame.K_LEFT:
                seek(-5.0)
            elif event.key == pygame.K_UP:
                seek(+30.0)
            elif event.key == pygame.K_DOWN:
                seek(-30.0)
            elif event.key == pygame.K_f:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
            elif event.key == pygame.K_h:
                show_hud = not show_hud

        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

    # ── Compute current t ─────────────────────────────────────────────────────
    t = current_elapsed()

    # Stop at end
    if t >= DURATION:
        if audio_loaded:
            pygame.mixer.music.stop()
        running = False
        break

    t_norm = t / DURATION

    # ── Render frame ──────────────────────────────────────────────────────────
    if not paused:
        last_frame = render_t(t_norm)

    sw, sh = screen.get_size()

    # Scale to fill screen, preserving aspect ratio
    aspect = WIDTH / HEIGHT
    if sw / sh > aspect:
        disp_h = sh
        disp_w = int(sh * aspect)
    else:
        disp_w = sw
        disp_h = int(sw / aspect)

    ox = (sw - disp_w) // 2
    oy = (sh - disp_h) // 2

    # Convert grayscale → RGB surface
    frame_rgb = np.stack([last_frame]*3, axis=-1)                   # (H,W,3)
    surf      = pygame.surfarray.make_surface(                       # needs (W,H,3)
        np.transpose(frame_rgb, (1,0,2))
    )
    surf_scaled = pygame.transform.scale(surf, (disp_w, disp_h))

    screen.fill((0,0,0))
    screen.blit(surf_scaled, (ox, oy))

    # ── HUD ───────────────────────────────────────────────────────────────────
    if show_hud:
        mm = int(t) // 60
        ss = int(t) % 60
        tm = int(DURATION) // 60
        ts = int(DURATION) % 60
        hud_text = f"{mm:02d}:{ss:02d} / {tm:02d}:{ts:02d}   {'PAUSED' if paused else '▶'}   H=hide"
        label = font.render(hud_text, True, (180,180,180))
        # Subtle background pill
        pw2, ph2 = label.get_width()+16, label.get_height()+8
        bg = pygame.Surface((pw2, ph2), pygame.SRCALPHA)
        bg.fill((0,0,0,120))
        screen.blit(bg, (12, sh - ph2 - 10))
        screen.blit(label, (20, sh - ph2 - 6))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("[done]")
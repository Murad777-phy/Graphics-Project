import sys
import math
import numpy as np
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.cluster import DBSCAN
import threading
import time
import torch

# --- Configuration ---
WINDOW_W, WINDOW_H = 1100, 700
CAM_W, CAM_H = 1280, 720
CAM_FPS      = 30
MIDAS_W, MIDAS_H = 640, 480

DEPTH_MAX = 10.0
DEPTH_MIN = 0.5
DEPTH_SCALE = 1.0
DEPTH_OFFSET = 0.0

CAM_FOV_DEG = 37.0
GRID_STEP  = 12
MAX_PTS    = 5000
DBSCAN_EPS     = 0.5
DBSCAN_MIN_PTS = 10
DBSCAN_EVERY   = 2

CLASS_COLOR = {
    "obstacle" : (1.0, 0.3, 0.2),
    "ground"   : (0.2, 0.9, 0.4),
    "boundary" : (0.2, 0.8, 1.0),
}

# --- UI Components ---
class Slider:
    def __init__(self, x, y, w, h, label, min_val, max_val, initial):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial
        self.grabbed = False

    def draw(self, surface):
        pygame.draw.rect(surface, (40, 45, 60), self.rect, border_radius=4)
        pos = (self.val - self.min_val) / (self.max_val - self.min_val + 1e-6)
        handle_x = self.rect.x + int(pos * self.rect.w)
        handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 4, 16, self.rect.h + 8)
        color = (255, 215, 0) if self.grabbed else (180, 180, 200)
        pygame.draw.rect(surface, color, handle_rect, border_radius=3)
        font = pygame.font.SysFont("monospace", 14, bold=True)
        txt = font.render(f"{self.label}: {self.val:.3f}", True, (255, 255, 255))
        surface.blit(txt, (self.rect.x, self.rect.y - 22))

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos): self.grabbed = True
        if event.type == MOUSEBUTTONUP: self.grabbed = False
        if event.type == MOUSEMOTION and self.grabbed:
            rel_x = max(0, min(event.pos[0] - self.rect.x, self.rect.w))
            self.val = self.min_val + (rel_x / self.rect.w) * (self.max_val - self.min_val)
            return True
        return False

# --- Shared pipeline state ---
class PipelineState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.frame       = None
        self.depth       = None
        self.pts         = []
        self.clusters    = []
        self.midas_fps   = 0.0
        self.depth_scale  = DEPTH_SCALE
        self.depth_offset = DEPTH_OFFSET

    def set_frame(self, f):
        with self._lock: self.frame = f

    def get_frame(self):
        with self._lock: return self.frame

    def set_depth(self, d, fps):
        with self._lock: self.depth, self.midas_fps = d, fps

    def set_pts(self, pts, clusters):
        with self._lock: self.pts, self.clusters = pts, clusters

    def get_render(self):
        with self._lock:
            return (self.frame, self.depth, self.pts, self.clusters,
                    self.midas_fps, self.depth_scale, self.depth_offset)

STATE = PipelineState()

# --- Threads ---
class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened(): sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if ret: STATE.set_frame(cv2.flip(frame, 1))

    def release(self): self._stop = True; self.cap.release()

class DepthThread:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = (self.device.type == "cuda")
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        if self.fp16: self.model = self.model.half()
        self.model.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
        self._last_frame, self._qlock, self._stop, self._fps = None, threading.Lock(), False, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def push(self, frame):
        with self._qlock: self._last_frame = frame

    def _run(self):
        while not self._stop:
            with self._qlock:
                frame, self._last_frame = self._last_frame, None

            if frame is None:
                time.sleep(0.004); continue

            t0 = time.perf_counter()
            small = cv2.resize(frame, (MIDAS_W, MIDAS_H))
            img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            inp = self.transform(img_rgb).to(self.device)
            if self.fp16: inp = inp.half()

            with torch.no_grad():
                pred = self.model(inp)
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(MIDAS_H, MIDAS_W), mode="bicubic").squeeze()

            disp = pred.float().cpu().numpy().astype(np.float32)
            disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
            disp = cv2.bilateralFilter(disp, d=9, sigmaColor=0.1, sigmaSpace=5)

            p5, p95 = np.percentile(disp, 5), np.percentile(disp, 95)
            range_val = p95 - p5
            norm = np.clip((disp - p5) / range_val, 0.0, 1.0) if range_val > 1e-4 else np.zeros_like(disp)

            depth_metric = DEPTH_MIN + (norm) * (DEPTH_MAX - DEPTH_MIN)
            depth_metric = cv2.medianBlur(depth_metric.astype(np.float32), 5)

            dt = time.perf_counter() - t0
            self._fps = 0.8 * self._fps + 0.2 * (1.0 / max(dt, 1e-6))
            STATE.set_depth(depth_metric, self._fps)

    def stop(self): self._stop = True

class VisionThread:
    def __init__(self):
        self._stop, self._bbox_ctr = False, 0
        self._last_clusters = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            with STATE._lock:
                depth, scale, offset = STATE.depth, STATE.depth_scale, STATE.depth_offset

            if depth is None:
                time.sleep(0.005); continue

            h, w = depth.shape
            vs, us = np.arange(0, h, GRID_STEP), np.arange(0, w, GRID_STEP)
            uu, vv = np.meshgrid(us, vs)
            uu, vv = uu.ravel(), vv.ravel()

            zz = (depth[vv, uu] * scale) + offset
            valid = (zz >= DEPTH_MIN) & (zz <= DEPTH_MAX)
            uu, vv, zz = uu[valid], vv[valid], zz[valid]

            if len(uu) == 0: continue

            fx = (w / 2.0) / math.tan(math.radians(CAM_FOV_DEG))
            cx, cy = w / 2.0, h / 2.0
            xx, yy = (uu - cx) / fx * zz, -((vv - cy) / fx * zz)

            pts = []
            for x, y, z in zip(xx, yy, zz):
                # Filter out extreme edges to prevent classifying walls as obstacles
                if y < -1.0:
                    label = "ground"
                elif abs(x) > 2.0 or z > 8.0:
                    label = "boundary"
                else:
                    label = "obstacle"
                pts.append((float(x), float(y), float(z), label))

            self._bbox_ctr += 1
            if self._bbox_ctr % DBSCAN_EVERY == 0:
                obs = np.array([p[:3] for p in pts if p[3] == "obstacle"], dtype=np.float32)
                new_clusters = []
                if len(obs) > DBSCAN_MIN_PTS:
                    lbls = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS).fit_predict(obs)
                    for cid in set(lbls):
                        if cid == -1: continue
                        grp = obs[lbls == cid]
                        mn, mx = grp.min(0), grp.max(0)

                        # --- WALL REJECTION FILTER ---
                        # If the object is massively wide or tall, it's likely a wall. Ignore it.
                        width = mx[0] - mn[0]
                        height = mx[1] - mn[1]
                        if width > 2.5 or height > 2.5:
                            continue

                        new_clusters.append({"box": (mn, mx)})
                self._last_clusters = new_clusters

            STATE.set_pts(pts, self._last_clusters)

    def stop(self): self._stop = True

# --- Rendering Functions ---
def draw_inset(tex_id, img, x, y, w=280, h=210):
    if img is None or not np.all(np.isfinite(img)): return
    img_rgb = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)

    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

    glEnable(GL_TEXTURE_2D); glDisable(GL_DEPTH_TEST)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(x, y)
    glTexCoord2f(1,0); glVertex2f(x+w, y)
    glTexCoord2f(1,1); glVertex2f(x+w, y+h)
    glTexCoord2f(0,1); glVertex2f(x, y+h)
    glEnd()

    glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()

def draw_grid():
    glLineWidth(1.0); glColor4f(0.15, 0.5, 0.25, 0.4); glBegin(GL_LINES)
    x = -4.0
    while x <= 4.01:
        glVertex3f(x, -1.2, 0.0); glVertex3f(x, -1.2, -DEPTH_MAX); x += 0.5
    z = 0.0
    while z >= -DEPTH_MAX - 0.01:
        glVertex3f(-4.0, -1.2, z); glVertex3f(4.0, -1.2, z); z -= 0.5
    glEnd()

# --- Main Execution ---
def main():
    cam, depth, vision = CameraThread(), DepthThread(), VisionThread()
    pygame.init()
    pygame.display.set_mode((WINDOW_W, WINDOW_H), DOUBLEBUF | OPENGL)
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    clock = pygame.time.Clock()

    s_scale = Slider(WINDOW_W//2 - 150, WINDOW_H - 100, 300, 12, "SCALE", 0.1, 5.0, DEPTH_SCALE)
    s_offset = Slider(WINDOW_W//2 - 150, WINDOW_H - 50, 300, 12, "OFFSET", -5.0, 5.0, DEPTH_OFFSET)

    cam_tex, depth_tex, ui_tex = glGenTextures(3)
    for t in [cam_tex, depth_tex, ui_tex]:
        glBindTexture(GL_TEXTURE_2D, t)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.05, 0.07, 0.12, 1.0)

    rot_x, rot_y, mouse_down_right, paused, fps = -20.0, 0.0, False, False, 0.0
    mono_font = pygame.font.SysFont("monospace", 15)

    while True:
        dt = clock.tick(60) / 1000.0
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        for event in pygame.event.get():
            if event.type == QUIT:
                vision.stop(); depth.stop(); cam.release(); pygame.quit(); sys.exit()

            if event.type == MOUSEBUTTONDOWN and event.button == 3: mouse_down_right = True
            if event.type == MOUSEBUTTONUP and event.button == 3: mouse_down_right = False

            if s_scale.handle_event(event): STATE.depth_scale = s_scale.val
            if s_offset.handle_event(event): STATE.depth_offset = s_offset.val

            if event.type == MOUSEMOTION and mouse_down_right:
                rot_y += event.rel[0] * 0.5; rot_x += event.rel[1] * 0.5

            if event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    vision.stop(); depth.stop(); cam.release(); pygame.quit(); sys.exit()
                if event.key == K_p: paused = not paused

        if not paused:
            f = STATE.get_frame()
            if f is not None: depth.push(f)

        frame, dmap, pts, clusters, midas_fps, d_scale, d_offset = STATE.get_render()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(45, WINDOW_W/WINDOW_H, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glTranslatef(0, -0.5, -5.0)
        glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0)

        # 3D Elements
        draw_grid()
        glPointSize(3.0); glBegin(GL_POINTS)
        for p in pts:
            c = CLASS_COLOR[p[3]]; shade = max(0, 1-(p[2]/DEPTH_MAX))
            glColor3f(c[0]*shade, c[1]*shade, c[2]*shade); glVertex3f(p[0], p[1], -p[2])
        glEnd()

        # 2D HUD and Overlays
        overlay.fill((0, 0, 0, 0))
        overlay.blit(mono_font.render(f"FPS: {fps:.1f} | MiDaS: {midas_fps:.1f} | Objects: {len(clusters)}", True, (180, 255, 180)), (10, 10))
        s_scale.draw(overlay); s_offset.draw(overlay)

        # Bounding Boxes (Full 3D Boxes, No Text)
        for c in clusters:
            mn, mx = c["box"]
            glColor3f(1.0, 0.85, 0.1); glLineWidth(2.4)
            # Bottom Loop
            glBegin(GL_LINE_LOOP); glVertex3f(mn[0], mn[1], -mn[2]); glVertex3f(mx[0], mn[1], -mn[2]); glVertex3f(mx[0], mn[1], -mx[2]); glVertex3f(mn[0], mn[1], -mx[2]); glEnd()
            # Top Loop
            glBegin(GL_LINE_LOOP); glVertex3f(mn[0], mx[1], -mn[2]); glVertex3f(mx[0], mx[1], -mn[2]); glVertex3f(mx[0], mx[1], -mx[2]); glVertex3f(mn[0], mx[1], -mx[2]); glEnd()
            # Vertical Pillars
            glBegin(GL_LINES)
            glVertex3f(mn[0], mn[1], -mn[2]); glVertex3f(mn[0], mx[1], -mn[2])
            glVertex3f(mx[0], mn[1], -mn[2]); glVertex3f(mx[0], mx[1], -mn[2])
            glVertex3f(mx[0], mn[1], -mx[2]); glVertex3f(mx[0], mx[1], -mx[2])
            glVertex3f(mn[0], mn[1], -mx[2]); glVertex3f(mn[0], mx[1], -mx[2])
            glEnd()

        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        pygame.draw.line(overlay, (0, 255, 0), (cx - 15, cy), (cx + 15, cy), 2)
        pygame.draw.line(overlay, (0, 255, 0), (cx, cy - 15), (cx, cy + 15), 2)

        # --- DRAW INSETS ---
        if frame is not None:
            draw_inset(cam_tex, cv2.resize(frame, (280, 210)), 15, WINDOW_H - 225)
        if dmap is not None:
            valid_dmap = np.nan_to_num(dmap, nan=0.0, posinf=0.0, neginf=0.0)
            d_min, d_max = valid_dmap.min(), valid_dmap.max()
            if d_max > d_min:
                d_viz = cv2.applyColorMap(((valid_dmap - d_min) / (d_max - d_min) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            else:
                d_viz = np.zeros((valid_dmap.shape[0], valid_dmap.shape[1], 3), dtype=np.uint8)
            draw_inset(depth_tex, cv2.resize(d_viz, (280, 210)), WINDOW_W - 295, WINDOW_H - 225)

        # Blit UI Overlay using the pre-allocated ui_tex
        raw = pygame.image.tostring(overlay, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, ui_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_W, WINDOW_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        glEnable(GL_TEXTURE_2D); glDisable(GL_DEPTH_TEST)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0)
        glTexCoord2f(1,0); glVertex2f(WINDOW_W,0)
        glTexCoord2f(1,1); glVertex2f(WINDOW_W,WINDOW_H)
        glTexCoord2f(0,1); glVertex2f(0,WINDOW_H)
        glEnd()

        glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST)
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()

        pygame.display.flip()

if __name__ == "__main__":
    main()
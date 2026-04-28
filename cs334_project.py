import sys
import math
import numpy as np
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.cluster import DBSCAN  # Added for bounding boxes

# --- Configuration ---
WINDOW_W, WINDOW_H = 1100, 700
DEPTH_MAX = 4.0
EDGE_STEP = 8
MAX_PTS   = 3000

CLASS_COLOR = {
    "obstacle" : (1.0, 0.3, 0.2),
    "ground"   : (0.2, 0.9, 0.4),
    "boundary" : (0.2, 0.8, 1.0),
}

# --- Camera State for Orbiting ---
cam_rot_x = -20
cam_rot_y = 0
mouse_down = False

def pixel_to_3d(u, v, img_w, img_h):
    # Pseudo-depth: lower pixels are closer. 
    # TODO: Replace with Real Stereo or Learned Monocular Depth (e.g., MiDaS)
    t = v / img_h
    z = DEPTH_MAX * (1.0 - t * 0.7 + 0.05)
    
    fx = (img_w / 2.0) / math.tan(math.radians(35))
    cx, cy = img_w / 2.0, img_h / 2.0
    x = (u - cx) / fx * z
    y = (v - cy) / fx * z
    return [x, -y, -z]

def classify(x, y, z):
    if y < -0.6: return "ground"
    if abs(x) > 1.8: return "boundary"
    return "obstacle"

def get_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 40, 120)

def edges_to_points(edges, img_w, img_h):
    ys, xs = np.where(edges > 0)
    if len(xs) == 0: return []
    
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    indices = indices[:MAX_PTS:EDGE_STEP]
    
    pts = []
    for i in indices:
        u, v = xs[i], ys[i]
        pos = pixel_to_3d(u, v, img_w, img_h)
        pts.append(pos + [classify(*pos)])
    return pts

# --- Rendering Helpers ---

def draw_grid():
    glLineWidth(1.0)
    glColor4f(0.2, 0.4, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in np.arange(-5, 5.1, 0.5):
        glVertex3f(i, -1.0, 0); glVertex3f(i, -1.0, -DEPTH_MAX)
        glVertex3f(-5, -1.0, -i); glVertex3f(5, -1.0, -i)
    glEnd()

def draw_bbox(pts):
    obs_pts = np.array([p[:3] for p in pts if p[3] == "obstacle"])
    if len(obs_pts) < 10: return

    clustering = DBSCAN(eps=0.3, min_samples=8).fit(obs_pts)
    labels = clustering.labels_

    glLineWidth(2.0)
    glColor3f(1.0, 0.8, 0.0) 
    
    for cluster_id in set(labels):
        if cluster_id == -1: continue
        group = obs_pts[labels == cluster_id]
        min_p = group.min(axis=0)
        max_p = group.max(axis=0)
        
        # Draw a simple 3D wireframe box
        glBegin(GL_LINE_LOOP) # Bottom face
        glVertex3f(min_p[0], min_p[1], min_p[2]); glVertex3f(max_p[0], min_p[1], min_p[2])
        glVertex3f(max_p[0], min_p[1], max_p[2]); glVertex3f(min_p[0], min_p[1], max_p[2])
        glEnd()
        glBegin(GL_LINE_LOOP) # Top face
        glVertex3f(min_p[0], max_p[1], min_p[2]); glVertex3f(max_p[0], max_p[1], min_p[2])
        glVertex3f(max_p[0], max_p[1], max_p[2]); glVertex3f(min_p[0], max_p[1], max_p[2])
        glEnd()
        glBegin(GL_LINES) # Vertical pillars
        glVertex3f(min_p[0], min_p[1], min_p[2]); glVertex3f(min_p[0], max_p[1], min_p[2])
        glVertex3f(max_p[0], min_p[1], min_p[2]); glVertex3f(max_p[0], max_p[1], min_p[2])
        glVertex3f(max_p[0], min_p[1], max_p[2]); glVertex3f(max_p[0], max_p[1], max_p[2])
        glVertex3f(min_p[0], min_p[1], max_p[2]); glVertex3f(min_p[0], max_p[1], max_p[2])
        glEnd()

def draw_camera_inset(tex_id, frame):
    # Convert CV2 BGR to RGB and flip for OpenGL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, 0) 
    
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
    
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_DEPTH_TEST)
    
    iw, ih = 280, 210
    x0, y0 = 15, WINDOW_H - ih - 15
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(x0, y0)
    glTexCoord2f(1,0); glVertex2f(x0+iw, y0)
    glTexCoord2f(1,1); glVertex2f(x0+iw, y0+ih)
    glTexCoord2f(0,1); glVertex2f(x0, y0+ih)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()

def main():
    global cam_rot_x, cam_rot_y, mouse_down
    cap = cv2.VideoCapture(0)
    pygame.init()
    pygame.display.set_mode((WINDOW_W, WINDOW_H), DOUBLEBUF | OPENGL)
    clock = pygame.time.Clock()

    # GL Setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    cam_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, cam_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    paused = False
    pts = []

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release(); pygame.quit(); sys.exit()
            
            # Orbit Controls: Right Click Drag
            if event.type == MOUSEBUTTONDOWN and event.button == 3: mouse_down = True
            if event.type == MOUSEBUTTONUP and event.button == 3: mouse_down = False
            if event.type == MOUSEMOTION and mouse_down:
                cam_rot_y += event.rel[0] * 0.5
                cam_rot_x += event.rel[1] * 0.5
            
            if event.type == KEYDOWN:
                if event.key == K_p: paused = not paused

        if not paused:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                edges = get_edges(frame)
                pts = edges_to_points(edges, frame.shape[1], frame.shape[0])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 3D View
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45, WINDOW_W/WINDOW_H, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        
        glTranslatef(0, -0.5, -5.0) # Move world back
        glRotatef(cam_rot_x, 1, 0, 0)
        glRotatef(cam_rot_y, 0, 1, 0)

        draw_grid()
        
        # Draw Points
        glPointSize(3)
        glBegin(GL_POINTS)
        for p in pts:
            col = CLASS_COLOR[p[3]]
            glColor3f(*col)
            glVertex3f(p[0], p[1], p[2])
        glEnd()

        draw_bbox(pts)

        if not paused:
            draw_camera_inset(cam_tex, frame)

        pygame.display.flip()
        clock.tick(30)
if __name__ == "__main__":
    main()
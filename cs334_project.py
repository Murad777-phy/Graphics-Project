"""
TODO:
  - LiDAR integration (ROS subscriber + point merge)
  - Manual camera orbit controls (mouse drag)
  - Bounding boxes around obstacle clusters
  - Depth estimation improvements (stereo or learned monocular)
"""

import sys
import math
import numpy as np
import cv2
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE, K_q, K_p
from OpenGL.GL import (
    glBegin, glEnd, glVertex3f, glColor3f, glColor4f,
    glPointSize, glLineWidth, glEnable, glClear, glClearColor,
    glLoadIdentity, glMatrixMode, glTranslatef, glRotatef,
    glBindTexture, glGenTextures, glTexImage2D, glTexParameteri,
    glBlendFunc, glDepthMask, glDisable,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_POINTS, GL_LINES, GL_QUADS,
    GL_MODELVIEW, GL_PROJECTION, GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_LINEAR, GL_RGB, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_DEPTH_TEST, GL_FALSE, GL_TRUE,
    glDeleteTextures,
)
from OpenGL.GLU import gluPerspective, gluOrtho2D


WINDOW_W  = 1100
WINDOW_H  = 700
DEPTH_MAX = 3.5
EDGE_STEP = 10
MAX_PTS   = 2000

CLASS_COLOR = {
    "obstacle" : (1.0, 0.3, 0.2),
    "ground"   : (0.2, 0.9, 0.4),
    "boundary" : (0.2, 0.8, 1.0),
}


def pixel_to_3d(u, v, img_w, img_h):
    t  = v / img_h
    z  = DEPTH_MAX * (1.0 - t * 0.65 + 0.1)
    fx = (img_w / 2.0) / math.tan(math.radians(30))
    cx, cy = img_w / 2.0, img_h / 2.0
    x = (u - cx) / fx * z
    y = (v - cy) / fx * z
    return (x, -y, z)


def classify(x, y, z):
    if y < -0.4:
        return "ground"
    if abs(x) > 1.5:
        return "boundary"
    return "obstacle"


def shade(color, z):
    b = 1.0 - (z / DEPTH_MAX) * 0.6
    return tuple(c * b for c in color)


def get_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med  = float(np.median(blur))
    return cv2.Canny(blur, max(0, int(0.5*med)), min(255, int(1.5*med)))


def edges_to_points(edges, img_w, img_h):
    ys, xs = np.where(edges > 0)
    xs, ys = xs[::EDGE_STEP], ys[::EDGE_STEP]
    if len(xs) > MAX_PTS:
        idx    = np.random.choice(len(xs), MAX_PTS, replace=False)
        xs, ys = xs[idx], ys[idx]
    pts = []
    for u, v in zip(xs, ys):
        x, y, z = pixel_to_3d(u, v, img_w, img_h)
        pts.append((x, y, z, classify(x, y, z)))
    return pts


def draw_grid():
    glLineWidth(1.0)
    glColor4f(0.15, 0.5, 0.25, 0.45)
    glBegin(GL_LINES)
    x = -3.0
    while x <= 3.01:
        glVertex3f(x, -1.2, 0.0); glVertex3f(x, -1.2, -DEPTH_MAX)
        x += 0.5
    z = 0.0
    while z >= -DEPTH_MAX - 0.01:
        glVertex3f(-3.0, -1.2, z); glVertex3f(3.0, -1.2, z)
        z -= 0.5
    glEnd()


def draw_axes():
    glLineWidth(2.5)
    glBegin(GL_LINES)
    glColor3f(1, 0.2, 0.2); glVertex3f(0,0,0); glVertex3f(1.2, 0, 0)
    glColor3f(0.2, 1, 0.2); glVertex3f(0,0,0); glVertex3f(0, 1.2, 0)
    glColor3f(0.3, 0.5, 1); glVertex3f(0,0,0); glVertex3f(0, 0,-1.2)
    glEnd()


def draw_points(pts):
    glPointSize(3.5)
    glBegin(GL_POINTS)
    for x, y, z, label in pts:
        r, g, b = shade(CLASS_COLOR[label], z)
        glColor4f(r, g, b, 1.0)
        glVertex3f(x, y, -z)
    glEnd()


# TODO: draw_lidar_points() — merge LiDAR scan with camera points


def draw_camera_inset(tex_id, frame, inset_w, inset_h):
    rgb = np.flipud(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    glBindTexture(GL_TEXTURE_2D, tex_id)
    h, w = rgb.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)

    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE)

    x0, y0 = 10, WINDOW_H - inset_h - 10
    x1, y1 = x0 + inset_w, y0 + inset_h
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glVertex3f(x0,y0,0); glVertex3f(x1,y0,0)
    glVertex3f(x1,y1,0); glVertex3f(x0,y1,0)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE)


def draw_hud(surface, fps, n_pts, paused):
    surface.fill((0, 0, 0, 0))
    mono = pygame.font.SysFont("monospace", 15)
    bold = pygame.font.SysFont("monospace", 17, bold=True)

    def t(msg, pos, col=(180,255,180), font=mono):
        surface.blit(font.render(msg, True, col), pos)

    t("CS 334 — Real-Time Vision System", (10, 10), (100, 210, 255), bold)
    t("PAUSED" if paused else f"FPS: {fps:4.1f}   pts: {n_pts}", (10, 33))

    lx, ly = WINDOW_W - 150, 10
    t("LEGEND", (lx, ly), (200, 200, 200), bold)
    for i, (label, (r, g, b)) in enumerate(CLASS_COLOR.items()):
        col = (int(r*255), int(g*255), int(b*255))
        pygame.draw.rect(surface, col, (lx, ly+22+i*20, 12, 12))
        t(label, (lx+18, ly+20+i*20), col)

    t("P-pause  Q-quit", (10, WINDOW_H - 25), (120, 120, 140))


def blit_hud(surface):
    raw = pygame.image.tostring(surface, "RGBA", True)
    tid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_W, WINDOW_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glVertex3f(0,0,0);              glVertex3f(WINDOW_W,0,0)
    glVertex3f(WINDOW_W,WINDOW_H,0); glVertex3f(0,WINDOW_H,0)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE)
    glDeleteTextures([tid])


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pygame.init()
    pygame.display.set_mode((WINDOW_W, WINDOW_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("CS 334 – Vision System")
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.05, 0.07, 0.12, 1.0)

    cam_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, cam_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    paused     = False
    pts        = []
    last_frame = None
    clock      = pygame.time.Clock()
    fps        = 0.0

    while True:
        dt  = clock.tick(60) / 1000.0
        fps = 0.9*fps + 0.1*(1.0/max(dt, 1e-6))

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release(); pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    cap.release(); pygame.quit(); sys.exit()
                if event.key == K_p:
                    paused = not paused

        if not paused:
            ret, frame = cap.read()
            if ret:
                frame      = cv2.flip(frame, 1)
                last_frame = frame.copy()
                edges      = get_edges(frame)
                pts        = edges_to_points(edges, 640, 480)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(50, WINDOW_W/WINDOW_H, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(0.0, -0.3, -4.5)
        glRotatef(-18, 1, 0, 0)   # fixed downward tilt, no spin

        draw_grid()
        draw_axes()
        draw_points(pts)

        if last_frame is not None:
            inset = cv2.resize(last_frame, (280, 210))
            draw_camera_inset(cam_tex, inset, 280, 210)

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        draw_hud(overlay, fps, len(pts), paused)
        blit_hud(overlay)

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
import pygame
import pygame_gui
import numpy as np
import moderngl
import argparse
from lenia_core import Lenia
import config

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Lenia Simulation')
parser.add_argument('--smoke-test', action='store_true', help='Run in a non-interactive mode for a few frames and exit.')
args = parser.parse_args()

# Initialize Pygame
pygame.init()

# Screen dimensions
UI_HEIGHT = 150
WINDOW_SIZE = (config.GRID_SIZE, config.GRID_SIZE + UI_HEIGHT)

# --- Pygame and OpenGL setup ---
pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Lenia (GPU Accelerated)")

# ModernGL context
ctx = moderngl.create_context()
ctx.viewport = (0, 0, *WINDOW_SIZE)

# --- Shaders (common for both Lenia and GUI) ---
vertex_shader = '''
    #version 330
    in vec2 in_vert; in vec2 in_uv;
    out vec2 v_uv; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); v_uv = in_uv; }
'''
fragment_shader = '''
    #version 330
    uniform sampler2D u_texture;
    in vec2 v_uv; out vec4 f_color;
    void main() { f_color = texture(u_texture, v_uv); }
'''

lenia_program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
lenia_program['u_texture'].value = 0

gui_program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
gui_program['u_texture'].value = 1

# --- Geometry and Textures ---
# Calculate the split point in Normalized Device Coordinates
y_split_ndc = (2 * UI_HEIGHT / WINDOW_SIZE[1]) - 1.0

# Vertices for the top (Lenia) and bottom (GUI) quads
vertices_lenia = np.array([
    -1.0, y_split_ndc, 0.0, 0.0,  # bottom left
     1.0, y_split_ndc, 1.0, 0.0,  # bottom right
    -1.0, 1.0,         0.0, 1.0,  # top left
     1.0, 1.0,         1.0, 1.0,  # top right
])
vertices_gui = np.array([
    -1.0, -1.0,        0.0, 0.0,  # bottom left
     1.0, -1.0,        1.0, 0.0,  # bottom right
    -1.0, y_split_ndc, 0.0, 1.0,  # top left
     1.0, y_split_ndc, 1.0, 1.0,  # top right
])

indices = np.array([0, 1, 2, 1, 2, 3])
ibo = ctx.buffer(indices.astype('i4').tobytes())

# VBOs and VAOs for each quad
vbo_lenia = ctx.buffer(vertices_lenia.astype('f4').tobytes())
vao_lenia = ctx.vertex_array(lenia_program, [(vbo_lenia, '2f 2f', 'in_vert', 'in_uv')], index_buffer=ibo)

vbo_gui = ctx.buffer(vertices_gui.astype('f4').tobytes())
vao_gui = ctx.vertex_array(gui_program, [(vbo_gui, '2f 2f', 'in_vert', 'in_uv')], index_buffer=ibo)

# Lenia texture
lenia_texture = ctx.texture((config.GRID_SIZE, config.GRID_SIZE), 3)
lenia_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

# GUI texture (sized to the UI panel)
gui_texture = ctx.texture((config.GRID_SIZE, UI_HEIGHT), 4) # RGBA
gui_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

# --- GUI Setup ---
manager = pygame_gui.UIManager(WINDOW_SIZE, enable_live_theme_updates=False)
# The offscreen surface for the GUI must be the full window size
gui_surface = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
lenia = Lenia()

# Create GUI elements. The panel is positioned at the bottom of the window.
ui_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, config.GRID_SIZE, config.GRID_SIZE, UI_HEIGHT)), manager=manager)

# First Column
pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 5, 100, 20)), text='Kernel Radius', manager=manager, container=ui_panel)
radius_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((120, 5, 120, 20)), start_value=lenia.kernel_radius, value_range=(1, 50), manager=manager, container=ui_panel)

pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 35, 100, 20)), text='Timestep', manager=manager, container=ui_panel)
timestep_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((120, 35, 120, 20)), start_value=lenia.timestep, value_range=(0.01, 0.2), manager=manager, container=ui_panel)

pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 65, 100, 20)), text='Kernel Shape', manager=manager, container=ui_panel)
shape_dropdown = pygame_gui.elements.UIDropDownMenu(options_list=lenia.kernel_shapes, starting_option=lenia.kernel_shape, relative_rect=pygame.Rect((120, 65, 120, 25)), manager=manager, container=ui_panel)

# Second Column
pygame_gui.elements.UILabel(relative_rect=pygame.Rect((250, 5, 100, 20)), text='Growth Mu', manager=manager, container=ui_panel)
mu_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((360, 5, 120, 20)), start_value=lenia.mu, value_range=(0.05, 0.3), manager=manager, container=ui_panel)

pygame_gui.elements.UILabel(relative_rect=pygame.Rect((250, 35, 100, 20)), text='Growth Sigma', manager=manager, container=ui_panel)
sigma_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((360, 35, 120, 20)), start_value=lenia.sigma, value_range=(0.005, 0.05), manager=manager, container=ui_panel)

randomize_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-160, 5, 150, 85)), text='Randomize World', manager=manager, container=ui_panel, anchors={'right': 'right'})

# Main loop
clock = pygame.time.Clock()
running = True
frame_count = 0
while running:
    time_delta = clock.tick(60) / 1000.0

    if args.smoke_test:
        if frame_count > 10: running = False
        frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        
        # Pass events to the manager. It will handle all GUI events correctly.
        manager.process_events(event)

        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == radius_slider: lenia.set_kernel_radius(event.value)
            elif event.ui_element == timestep_slider: lenia.set_timestep(event.value)
            elif event.ui_element == mu_slider: lenia.set_mu(event.value)
            elif event.ui_element == sigma_slider: lenia.set_sigma(event.value)
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == shape_dropdown: lenia.set_kernel_shape(event.text)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == randomize_button: lenia.randomize_world()

    manager.update(time_delta)
    lenia.update()

    # --- Drawing ---
    ctx.clear(0.1, 0.1, 0.1)

    # 1. Render Lenia to its texture
    world_cpu = lenia.get_world().get()
    texture_data = (np.stack([world_cpu] * 3, axis=-1) * 255).astype(np.uint8)
    lenia_texture.write(texture_data.tobytes())

    # 2. Render GUI to its texture
    gui_surface.fill((0, 0, 0, 0)) # Clear with transparent color
    manager.draw_ui(gui_surface)
    # Extract only the UI part of the surface for the texture
    gui_data = pygame.image.tostring(gui_surface.subsurface(pygame.Rect(0, config.GRID_SIZE, config.GRID_SIZE, UI_HEIGHT)), 'RGBA', True)
    gui_texture.write(gui_data)

    # 3. Render Lenia texture to its quad
    lenia_texture.use(location=0)
    vao_lenia.render()

    # 4. Blend GUI texture on its quad
    ctx.enable(moderngl.BLEND)
    gui_texture.use(location=1)
    vao_gui.render()
    ctx.disable(moderngl.BLEND)

    pygame.display.set_caption(f"Lenia - FPS: {clock.get_fps():.2f}")
    pygame.display.flip()

# Quit Pygame
pygame.quit()

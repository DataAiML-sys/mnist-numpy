import pygame
import numpy as np
from network import NeuralNetwork

WINDOW_SIZE = 560
CANVAS_SIZE = 28
SCALE = WINDOW_SIZE // CANVAS_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (50, 100, 200)

print("Loading trained model...")

net = NeuralNetwork()
data = np.load("data/model_weights.npz")
net.W1 = data['W1']
net.b1 = data['b1']
net.W2 = data['W2']
net.b2 = data['b2']

print("Model loaded. You can draw now.")

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 100))
pygame.display.set_caption("Draw a digit (0-9)")

font = pygame.font.SysFont("Arial", 24)

canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

def draw_on_canvas(mouse_x, mouse_y):
    if mouse_y >= WINDOW_SIZE:
        return
    cx = mouse_x // SCALE
    cy = mouse_y // SCALE
    if 0 <= cx < CANVAS_SIZE and 0 <= cy < CANVAS_SIZE:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny = cy + dy
                nx = cx + dx
                if 0 <= nx < CANVAS_SIZE and 0 <= ny < CANVAS_SIZE:
                    canvas[ny, nx] = 255

def canvas_to_surface():
    surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    for y in range(CANVAS_SIZE):
        for x in range(CANVAS_SIZE):
            v = canvas[y, x]
            surface.set_at((x, y), (v, v, v))
    return pygame.transform.scale(surface, (WINDOW_SIZE, WINDOW_SIZE))

def center_and_scale(img):
    """Crop digit to its bounding box, scale to fit in 20x20, place in center of 28x28."""
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img[rmin:rmax+1, cmin:cmax+1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    row_idx = (np.arange(new_h) * (h / new_h)).astype(int)
    col_idx = (np.arange(new_w) * (w / new_w)).astype(int)
    resized = cropped[row_idx][:, col_idx]

    result = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    result[top:top+new_h, left:left+new_w] = resized

    return result

def predict_digit():
    img = center_and_scale(canvas)
    x = img.astype(np.float32) / 255.0
    x = x.reshape(1, 784)
    probs = net.forward(x)
    pred = np.argmax(probs[0])
    confidence = probs[0][pred] * 100
    return pred, confidence

running = True
mouse_pressed = False
last_prediction = None
last_confidence = 0.0

print("\nControls:")
print("  Mouse: draw")
print("  C: clear canvas")
print("  Q: quit")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
                last_prediction = None
                last_confidence = 0.0
                print("Cleared")
            elif event.key == pygame.K_q:
                running = False

    if mouse_pressed:
        mx, my = pygame.mouse.get_pos()
        draw_on_canvas(mx, my)

    if canvas.sum() > 0:
        pred, confidence = predict_digit()
        last_prediction = pred
        last_confidence = confidence

    screen.fill(WHITE)

    canvas_surface = canvas_to_surface()
    screen.blit(canvas_surface, (0, 0))

    pygame.draw.rect(screen, GRAY, (0, 0, WINDOW_SIZE, WINDOW_SIZE), 2)

    if last_prediction is not None:
        text = f"Prediction: {last_prediction}  ({last_confidence:.1f}%)"
        text_surf = font.render(text, True, BLUE)
        screen.blit(text_surf, (20, WINDOW_SIZE + 20))

    hint = "C: clear    Q: quit"
    hint_surf = font.render(hint, True, GRAY)
    screen.blit(hint_surf, (20, WINDOW_SIZE + 60))

    pygame.display.flip()

pygame.quit()
print("Goodbye!")
import numpy as np
import cv2

class BoardState:

    RED_PLAYER = 1
    BLACK_PLAYER = -1
    EMPTY = 0
    
    def __init__(self, shape):
        self.state = np.array([
            [ 0,  1,  0,  1,  0,  1,  0,  1],
            [ 1,  0,  1,  0,  1,  0,  1,  0],
            [ 0,  1,  0,  1,  0,  1,  0,  1],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [-1,  0, -1,  0, -1,  0, -1,  0],
            [ 0, -1,  0, -1,  0, -1,  0, -1],
            [-1,  0, -1,  0, -1,  0, -1,  0]
        ], dtype=int)
        self.grabbing = False
        self.move_source = None
        self.move_dest = None
        self.turn = BoardState.BLACK_PLAYER
        self.shape = shape
        self.cell_side = self.shape[1] // 8

        board = cv2.imread("assets/board.png", cv2.IMREAD_COLOR)
        board = cv2.resize(board, self.shape)
        self.image = cv2.multiply(board, 0.5)
        red_piece = cv2.imread("assets/red_piece.png", cv2.IMREAD_UNCHANGED)
        self.red_piece = cv2.resize(red_piece, (self.cell_side, self.cell_side))
        black_piece = cv2.imread("assets/black_piece.png", cv2.IMREAD_UNCHANGED)
        self.black_piece = cv2.resize(black_piece, (self.cell_side, self.cell_side))


    def drawBoard(self, frame, blue_centroid, red_centroid):
        self.previous_state = self.state
        overlay = cv2.add(frame, self.image)
        for i, row in enumerate(self.state):
            for j, value in enumerate(row):
                if value == 1:
                    piece = self.red_piece
                elif value == -1:
                    piece = self.black_piece
                else:
                    continue
                
                position = (i * self.cell_side, j * self.cell_side)
                overlay = self.drawPiece(overlay, piece, position)
                
        if blue_centroid:
            overlay = cv2.circle(overlay, center=blue_centroid, radius=10, color=(255, 0, 0), thickness=5)
            overlay = self.handleGrab(overlay, blue_centroid, BoardState.BLACK_PLAYER)
        elif self.grabbing:
            overlay = self.handleRelease(overlay, BoardState.BLACK_PLAYER)

        if red_centroid:
            overlay = cv2.circle(overlay, center=red_centroid, radius=10, color=(0, 0, 255), thickness=5)
            overlay = self.handleGrab(overlay, red_centroid, BoardState.RED_PLAYER)
        elif self.grabbing:
            overlay = self.handleRelease(overlay, BoardState.RED_PLAYER)

        return overlay
    

    def drawPiece(self, overlay, piece, position):
        y, x = position
        h, w, _ = piece.shape

        y1 = max(0, y)
        y2 = min(overlay.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(overlay.shape[1], x + w)

        py1 = y1 - y
        py2 = py1 + (y2 - y1)
        px1 = x1 - x
        px2 = px1 + (x2 - x1)

        piece_rgb = piece[py1:py2, px1:px2, :3]
        alpha = piece[py1:py2, px1:px2, 3] / 255.0
        roi = overlay[y1:y2, x1:x2]

        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + piece_rgb[:, :, c] * alpha

        overlay[y1:y2, x1:x2] = roi
        return overlay


    def handleGrab(self, overlay, centroid, player):
        column = centroid[0] // self.cell_side
        row = centroid[1] // self.cell_side
        cell_value = self.state[row][column]
        if self.turn == player:
            self.move_dest = (row, column)
            if self.grabbing:
                piece = self.red_piece if player == BoardState.RED_PLAYER else self.black_piece
                origin = (centroid[1] - self.cell_side // 2, centroid[0] - self.cell_side // 2)
                overlay = self.drawPiece(overlay, piece, origin)
            elif player == cell_value:
                self.move_source = (row, column)
                self.state[row][column] = BoardState.EMPTY
                self.grabbing = True
        return overlay
    

    def handleRelease(self, overlay, player):
        if self.turn == player and self.moveIsLegal():
            row, column = self.move_dest
            piece = self.red_piece if self.turn == BoardState.RED_PLAYER else self.black_piece
            self.state[row][column] = self.turn
            position = (row * self.cell_side, column * self.cell_side)
            overlay = self.drawPiece(overlay, piece, position)
            self.grabbing = False
            self.move_source = None
            self.move_dest = None
            self.turn = BoardState.RED_PLAYER if self.turn == BoardState.BLACK_PLAYER else BoardState.BLACK_PLAYER
        return overlay
    

    def moveIsLegal(self):
        row, column = self.move_dest
        if self.state[row][column] == BoardState.EMPTY:
            return True
        return False
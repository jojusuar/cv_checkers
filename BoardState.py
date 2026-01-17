import numpy as np
import cv2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

class BoardState:

    RED_PLAYER = 1
    BLACK_PLAYER = -1
    RED_KING = 2
    BLACK_KING = -2
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
        self.source_value = None
        self.move_dest = None
        self.winner = None
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


    # Módulo de renderizado del tablero
    def drawBoard(self, frame, thumb, index):
        self.previous_state = self.state.copy()
        overlay = cv2.add(frame, self.image)
        
        for i, row in enumerate(self.state):
            for j, value in enumerate(row):
                position = (i, j)
                piece_value = self.state[position[0]][position[1]]
                if self.pieceIsRed(piece_value):
                    piece = self.red_piece
                elif self.pieceIsBlack(piece_value):
                    piece = self.black_piece
                else:
                    continue
                is_king = self.pieceIsKing(piece_value)
                pixel_position = (i * self.cell_side, j * self.cell_side)
                overlay = self.drawPiece(overlay, piece, pixel_position, is_king)

        if self.winner:
            overlay = self.drawVictoryBanner(overlay, self.winner)
               
        if thumb and index:
            distance_x = thumb.x - index.x
            distance_y = thumb.y - index.y
            distance_magnitude = np.sqrt(distance_x**2 + distance_y ** 2)

            thumb_position = self.unnormalizeLandmark(thumb)
            index_position = self.unnormalizeLandmark(index)
            midpoint = (
                (thumb_position[0] + index_position[0]) // 2,
                (thumb_position[1] + index_position[1]) // 2
            )

            if distance_magnitude < 8e-2:
                overlay = cv2.circle(overlay, center=thumb_position, radius=10, color=(0, 255, 0), thickness=5)
                overlay = cv2.circle(overlay, center=index_position, radius=10, color=(0, 255, 0), thickness=5)
                overlay = self.handleGrab(overlay, midpoint)
            else:
                overlay = cv2.circle(overlay, center=thumb_position, radius=10, color=(255, 0, 0), thickness=5)
                overlay = cv2.circle(overlay, center=index_position, radius=10, color=(255, 0, 0), thickness=5)
                if self.grabbing:
                    self.grabbing = False
                    overlay = self.handleRelease(overlay)

        return overlay
    

    def pieceIsBlack(self, value):
        return value == self.BLACK_PLAYER or value == self.BLACK_KING


    def pieceIsRed(self, value):
        return value == self.RED_PLAYER or value == self.RED_KING
    

    def pieceIsKing(self, value):
        return value == self.RED_KING or value == self.BLACK_KING
    

    def pieceCanBeMoved(self, position):
        cell_value = self.state[position[0]][position[1]]
        if self.isBlackTurn():
            return self.pieceIsBlack(cell_value)
        elif self.isRedTurn():
            return self.pieceIsRed(cell_value)
        return False
    

    def isRedTurn(self):
        return self.turn == self.RED_PLAYER
    

    def isBlackTurn(self):
        return self.turn == self.BLACK_PLAYER


    def drawPiece(self, overlay, piece, position, is_king=False):
        y, x = position
        h, w, _ = piece.shape
        
        y1 = max(0, y)
        y2 = min(overlay.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(overlay.shape[1], x + w)
        
        if y2 <= y1 or x2 <= x1:
            return overlay
        
        py1 = y1 - y
        py2 = py1 + (y2 - y1)
        px1 = x1 - x
        px2 = px1 + (x2 - x1)
        
        piece_rgb = piece[py1:py2, px1:px2, :3]
        alpha = piece[py1:py2, px1:px2, 3:4] / 255.0
        roi = overlay[y1:y2, x1:x2]
        
        blended = (roi * (1 - alpha) + piece_rgb * alpha).astype(np.uint8)
        overlay[y1:y2, x1:x2] = blended
        
        if is_king and y >= 0 and x >= 0:
            text = "K"
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2
            
            if 0 <= text_x < overlay.shape[1] and 0 <= text_y < overlay.shape[0]:
                cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return overlay
    

    # Módulo de intérprete de jugadas
    def handleGrab(self, overlay, centroid):
        column = np.clip(centroid[0] // self.cell_side, 0, 7)
        row = np.clip(centroid[1] // self.cell_side, 0, 7)
        position = (row, column)
        self.move_dest = position
        
        if self.grabbing:
            piece = self.red_piece if self.isRedTurn() else self.black_piece
            is_king = self.pieceIsKing(self.source_value)
            origin = (centroid[1] - self.cell_side // 2, centroid[0] - self.cell_side // 2)
            overlay = self.drawPiece(overlay, piece, origin, is_king)
        elif self.pieceCanBeMoved(position):
            self.move_source = position
            self.source_value = self.state[row][column]
            self.state[row][column] = BoardState.EMPTY
            self.grabbing = True
        return overlay
    

    def handleRelease(self, overlay):
        if self.grabbing:
            return overlay
        
        if self.moveIsLegal():
            self.commitMove(overlay, self.move_dest)
            self.turn = BoardState.RED_PLAYER if self.isBlackTurn() else BoardState.BLACK_PLAYER
            self.winner = self.checkVictoryConditions()
        else:
            self.commitMove(overlay, self.move_source)
        return overlay


    def drawVictoryBanner(self, overlay, winner):
        banner_height = 120
        banner_y = (self.shape[0] - banner_height) // 2
        
        overlay_copy = overlay.copy()
        cv2.rectangle(overlay_copy, 
                    (0, banner_y), 
                    (self.shape[1], banner_y + banner_height), 
                    (0, 0, 0), 
                    -1)
        alpha = 0.7
        overlay = cv2.addWeighted(overlay, 1 - alpha, overlay_copy, alpha, 0)
        
        if winner == self.BLACK_PLAYER:
            text = "BLACK WINS!"
            color = (255, 255, 255)
        elif winner == self.RED_PLAYER:
            text = "RED WINS!"
            color = (0, 0, 255)
        else:
            return overlay
        
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 2.5
        thickness = 5
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x = (self.shape[1] - text_width) // 2
        text_y = banner_y + (banner_height + text_height) // 2
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return overlay


    def checkVictoryConditions(self):
        red_count = np.sum(self.state > 0)
        black_count = np.sum(self.state < 0)
        
        if red_count == 0:
            return self.BLACK_PLAYER
        elif black_count == 0:
            return self.RED_PLAYER
        
        return None
    
    
    def checkKingConditions(self, target_position):
        row, column = target_position
        if self.isRedTurn():
            return row == 7
        if self.isBlackTurn():
            return row == 0
        return False
    

    def commitMove(self, overlay, target_position):
        row, column = target_position
        piece_value = self.source_value

        if self.checkKingConditions(target_position):
            piece_value = BoardState.RED_KING if self.isRedTurn() else BoardState.BLACK_KING
        
        self.state[row][column] = piece_value
        
        piece = self.red_piece if self.isRedTurn() else self.black_piece
        position = (row * self.cell_side, column * self.cell_side)
        overlay = self.drawPiece(overlay, piece, position, self.pieceIsKing(piece_value))
        
        self.grabbing = False
        self.move_source = None
        self.source_value = None
        self.move_dest = None
        return overlay
    

    def moveIsLegal(self):
        if self.move_source is None or self.move_dest is None:
            return False
        
        src_row, src_col = self.move_source
        dest_row, dest_col = self.move_dest
        
        if not (0 <= dest_row < 8 and 0 <= dest_col < 8):
            return False
        
        if self.state[dest_row][dest_col] != BoardState.EMPTY:
            return False
        
        if (dest_row + dest_col) % 2 == 0:
            return False
        
        row_diff = dest_row - src_row
        col_diff = abs(dest_col - src_col)
        
        is_king = self.pieceIsKing(self.source_value)
        
        if not is_king:
            if self.pieceIsBlack(self.source_value) and row_diff >= 0:
                return False
            if self.pieceIsRed(self.source_value) and row_diff <= 0:
                return False
        
        if abs(row_diff) == 1 and col_diff == 1:
            return True
        
        if abs(row_diff) == 2 and col_diff == 2:
            jumped_row = src_row + row_diff // 2
            jumped_col = src_col + (dest_col - src_col) // 2
            jumped_value = self.state[jumped_row, jumped_col]
            
            if self.isBlackTurn():
                is_opponent = self.pieceIsRed(jumped_value)
            else:
                is_opponent = self.pieceIsBlack(jumped_value)
            
            if not is_opponent:
                return False
            
            self.state[jumped_row][jumped_col] = BoardState.EMPTY
            return True
        
        if abs(row_diff) == 4 and col_diff == 4:
            first_jumped_row = src_row + (1 if row_diff > 0 else -1)
            first_jumped_col = src_col + (1 if dest_col > src_col else -1)
            first_jumped_value = self.state[first_jumped_row, first_jumped_col]
            
            second_jumped_row = src_row + (3 if row_diff > 0 else -3)
            second_jumped_col = src_col + (3 if dest_col > src_col else -3)
            second_jumped_value = self.state[second_jumped_row, second_jumped_col]
            
            middle_row = src_row + (2 if row_diff > 0 else -2)
            middle_col = src_col + (2 if dest_col > src_col else -2)
            
            if self.state[middle_row, middle_col] != BoardState.EMPTY:
                return False
            
            if self.isBlackTurn():
                first_is_opponent = self.pieceIsRed(first_jumped_value)
                second_is_opponent = self.pieceIsRed(second_jumped_value)
            else:
                first_is_opponent = self.pieceIsBlack(first_jumped_value)
                second_is_opponent = self.pieceIsBlack(second_jumped_value)
            
            if not (first_is_opponent and second_is_opponent):
                return False
            
            self.state[first_jumped_row, first_jumped_col] = BoardState.EMPTY
            self.state[second_jumped_row, second_jumped_col] = BoardState.EMPTY
            return True
        
        return False
    

    def unnormalizeLandmark(self, landmark: NormalizedLandmark):
        return (int(landmark.x * self.shape[1]), int(landmark.y * self.shape[1]))